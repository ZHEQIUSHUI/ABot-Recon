"""Torch-free ABot-Recon inference: onnxruntime graph + numpy pose composition.

Deps: numpy, onnxruntime, PIL (decode+resize). No torch anywhere.

Graph contract (export_full.py): imgs [1,N,3,280,504] ->
  local_points [1,N,280,504,3], conf [1,N,280,504,1], raw_delta [1,N-1,4,4], resid [1,N-1,3]
N is FIXED at export. Longer videos run as sliding windows with `warmup` frames of left
context; warmup outputs are discarded (validated: warmup=24 reproduces the full-sequence
result to cos=1.00000 / 0.15% trajectory error; warmup=16 -> ~1.3%).

Pose composition (validated bit-exact vs torch AdjacentPoseHead):
  pose[0]=I;  d=raw_delta[i];  d[:3,:3]=d[:3,:3]@rodrigues(resid[i]);  pose[i+1]=pose[i]@d.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

PAD_RGB = (0.485, 0.456, 0.406)


# ---------------- preprocessing (replicates abot_recon.preprocessing) ----------------
def preprocess_rgb(rgb: np.ndarray, height: int = 280, width: int = 504) -> np.ndarray:
    """HWC uint8/float RGB -> CHW float32 (bicubic overshoots preserved, like the reference:
    width-lock BICUBIC+antialias resize, center crop / ImageNet-mean pad, NO final clamp)."""
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() > 1.5:
            arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    sh, sw = arr.shape[:2]
    rh = max(1, round(sh * width / max(sw, 1)))
    # PIL 'F'-mode resize == antialiased float bicubic (what tvf.resize(antialias=True) matches)
    chans = [np.asarray(Image.fromarray(arr[:, :, c], mode="F").resize((width, rh), Image.BICUBIC))
             for c in range(3)]
    t = np.stack(chans, 0)                                     # [3, rh, W]
    if rh > height:
        top = round((rh - height) * 0.5)
        t = t[:, top:top + height]
    elif rh < height:
        pad_top = (height - rh) // 2
        canvas = np.empty((3, height, width), np.float32)
        for c in range(3):
            canvas[c] = PAD_RGB[c]
        canvas[:, pad_top:pad_top + rh] = t
        t = canvas
    return np.ascontiguousarray(t, np.float32)


# ---------------- pose math ----------------
def rodrigues(rv: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    th = float(np.linalg.norm(rv))
    if th < eps:
        return np.eye(3, dtype=np.float64)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], np.float64)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def compose_poses(raw_delta: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """[P,4,4],[P,3] -> camera poses [P+1,4,4] (c2w), pose[0]=I."""
    pose = np.eye(4, dtype=np.float64)
    out = [pose.copy()]
    for i in range(raw_delta.shape[0]):
        d = raw_delta[i].astype(np.float64).copy()
        d[:3, :3] = d[:3, :3] @ rodrigues(resid[i].astype(np.float64))
        pose = pose @ d
        out.append(pose.copy())
    return np.stack(out).astype(np.float32)


def local_to_world(local_points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """[N,H,W,3] local + [N,4,4] c2w -> world [N,H,W,3]."""
    R = poses[:, :3, :3]; t = poses[:, :3, 3]
    return np.einsum("nij,nhwj->nhwi", R, local_points) + t[:, None, None, :]


# ---------------- chunked ORT inference ----------------
class OrtAbotRecon:
    def __init__(self, onnx_path: str, providers=None, warmup: int = 24):
        import onnxruntime as ort, json, os
        so = ort.SessionOptions()
        # CPU arena hoards freed blocks across the multi-GB attention transients of
        # successive windows (fragmentation -> unbounded growth -> OOM). Disable by default;
        # ORT then returns memory to the OS between nodes. ORT_ARENA=1 re-enables.
        so.enable_cpu_mem_arena = os.environ.get("ORT_ARENA", "0") == "1"
        # CUDA EP: the extended-level BiasSoftmax fusion trips CUDNN_STATUS_NOT_SUPPORTED on
        # our huge attention tensors -> BASIC skips that fusion (plain Add+Softmax kernels).
        lvl = os.environ.get("ORT_OPT_LEVEL", "basic").lower()
        so.graph_optimization_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }.get(lvl, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        provs = list(providers or ["CPUExecutionProvider"])
        # CUDA EP on unified-memory boxes (GB10): cap the arena and grow it exactly as
        # requested — the default BFC greedy growth + decomposed-attention transients
        # (6.4GB logits per chunk x 36 layers) OOMs the whole machine. HEURISTIC conv
        # search avoids exhaustive-autotune workspace spikes.
        cuda_opts = {
            "gpu_mem_limit": str(int(float(os.environ.get("ORT_CUDA_MEM_GB", "64")) * (1 << 30))),
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
        }
        provs = [(pv, cuda_opts) if pv == "CUDAExecutionProvider" else pv for pv in provs]
        self.sess = ort.InferenceSession(onnx_path, so, providers=provs)
        self.N = int(self.sess.get_inputs()[0].shape[1])       # fixed window length
        self.warmup = min(int(warmup), self.N - 1)
        # sidecar json: normalization constants (graph input is pre-normalized)
        side = onnx_path.replace(".onnx", ".json")
        if os.path.exists(side):
            meta = json.load(open(side))
            self.mean = np.asarray(meta["image_mean"], np.float32).reshape(1, 3, 1, 1)
            self.std = np.asarray(meta["image_std"], np.float32).reshape(1, 3, 1, 1)
        else:                                                  # ImageNet defaults
            self.mean = np.asarray([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
            self.std = np.asarray([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
        self.input_name = self.sess.get_inputs()[0].name

    def _run(self, imgs: np.ndarray):
        x = ((imgs - self.mean) / self.std).astype(np.float32)  # runtime-side normalization
        lp, cf, rd, rs = self.sess.run(["local_points", "conf", "raw_delta", "resid"],
                                       {self.input_name: x[None]})
        return lp[0], cf[0], rd[0], rs[0]

    def infer(self, frames: list[np.ndarray], progress=None):
        """frames: list of CHW float32 (preprocess_rgb). Returns camera_poses [T,4,4],
        local_points [T,280,504,3], conf [T,280,504], raw_delta/resid."""
        T = len(frames)
        N, W = self.N, self.warmup
        X = np.stack(frames).astype(np.float32)
        if T <= N:                                             # single (padded) window
            pad = np.repeat(X[-1:], N - T, 0) if T < N else X[:0]
            lp, cf, rd, rs = self._run(np.concatenate([X, pad]) if T < N else X)
            lps, cfs, rds, rss = lp[:T], cf[:T], rd[:max(T - 1, 0)], rs[:max(T - 1, 0)]
        else:
            lps = np.empty((T, 280, 504, 3), np.float32)
            cfs = np.empty((T, 280, 504, 1), np.float32)
            rds = np.empty((T - 1, 4, 4), np.float32)
            rss = np.empty((T - 1, 3), np.float32)
            produced = 0
            while produced < T:
                if produced == 0:
                    st, e = 0, N
                else:
                    e = min(produced + (N - W), T)
                    st = e - N                                 # ≥0 because T > N
                lp, cf, rd, rs = self._run(X[st:e])
                lf = produced - st                             # local index of first kept frame
                lps[produced:e] = lp[lf:]
                cfs[produced:e] = cf[lf:]
                pair_from = max(produced - 1, 0)               # connector pair from fresh window
                rds[pair_from:e - 1] = rd[pair_from - st:]
                rss[pair_from:e - 1] = rs[pair_from - st:]
                if progress:
                    progress(e, T)
                produced = e
        poses = compose_poses(rds, rss)
        # graph emits conf LOGITS; api.infer applies sigmoid -> replicate (confidence in 0..1)
        conf = 1.0 / (1.0 + np.exp(-cfs[..., 0].astype(np.float32)))
        return {"camera_poses": poses, "local_points": lps, "conf": conf,
                "raw_delta": rds, "resid": rss}
