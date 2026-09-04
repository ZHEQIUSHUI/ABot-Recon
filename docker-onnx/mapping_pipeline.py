#!/usr/bin/env python3
"""ABot-Recon 建图流水线 — torch-free(onnxruntime + numpy)版。

与 docker/mapping_pipeline.py 同接口(service.py 零改动复用),但推理走导出的 ONNX 图
(abot_recon_nXX.onnx,固定 N 滑窗)+ numpy 位姿组合,整条链路无 torch:
  依赖仅 onnxruntime, numpy, opencv, PIL, open3d, matplotlib。
数值上与 torch 版对齐(位姿/点图 cos≈1,warmup=24 滑窗 vs 全序列 0.15% 轨迹误差)。

产物与 torch 版一致:recon.npz / cloud.ply / cloud_viz.ply / floorplan.png / view_*.png / meta.json
CLI:  mapping_pipeline.py <video> <out_dir> [--fps 8] [--ceiling-cut] [--ceiling-keep 0.85]
"""
import os, sys, time, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = os.environ.get("ABOT_MODEL_ID", "/app/model/abot_recon.onnx")  # .onnx path
ORT_PROVIDERS = [p for p in os.environ.get(
    "ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider").split(",") if p]
WARMUP = int(os.environ.get("ABOT_WARMUP", "24"))


def loop_enabled():
    return False                                  # loop closure是torch侧功能,ONNX版不带


def load_ready_model(model_id=None, use_sdpa=True, device="cuda"):
    """Load the ONNX recon graph, ready to infer. model_id = path to .onnx (or a dir
    containing abot_recon.onnx). Provider fallback: CUDA EP -> CPU EP."""
    from ort_recon import OrtAbotRecon
    p = model_id or MODEL_ID
    if os.path.isdir(p):
        cands = [f for f in sorted(os.listdir(p)) if f.endswith(".onnx")]
        if not cands:
            raise FileNotFoundError(f"no .onnx in {p}")
        p = os.path.join(p, cands[0])
    return OrtAbotRecon(p, providers=ORT_PROVIDERS, warmup=WARMUP)


def _extract_frames(video, fps, out_dir):
    import cv2
    cap = cv2.VideoCapture(video)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(src / max(fps, 1)))
    idx, saved = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            p = os.path.join(out_dir, f"{len(saved):06d}.jpg")
            cv2.imwrite(p, frame)
            saved.append(p)
        idx += 1
    cap.release()
    return saved


def _grav_basis(cams):
    import numpy as np
    cc = cams - cams.mean(0)
    _, _, Vt = np.linalg.svd(cc, full_matrices=False)
    up = Vt[2]; e1 = Vt[0]
    e2 = np.cross(up, e1); e2 /= np.linalg.norm(e2) + 1e-9
    return np.stack([e1, e2, up], 1).astype(np.float32)


def run(video, out_dir, fps=8, ceiling_cut=False, ceiling_keep=0.85,
        model=None, model_id=None, device="cuda"):
    import numpy as np, cv2, tempfile, shutil
    from ort_recon import preprocess_rgb, local_to_world
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    meta = {"video": os.path.basename(video), "fps": fps, "engine": "ABot-Recon-ONNX"}

    fdir = tempfile.mkdtemp(prefix="abot_frames_")
    try:
        fpaths = _extract_frames(video, fps, fdir)
        meta["frames"] = len(fpaths)
        if not fpaths:
            raise RuntimeError("no frames decoded from video")

        if model is None:
            model = load_ready_model(model_id)
        frames, col = [], []
        for p in fpaths:
            bgr = cv2.imread(p)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = preprocess_rgb(rgb)                                    # CHW float
            frames.append(t)
            col.append((np.clip(t, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0))
        col = np.stack(col)                                            # [T,H,W,3] uint8

        t_inf = time.time()
        out = model.infer(frames)
        meta["infer_s"] = round(time.time() - t_inf, 1)
    finally:
        shutil.rmtree(fdir, ignore_errors=True)

    poses = out["camera_poses"]                                        # [T,4,4] c2w
    wp = local_to_world(out["local_points"], poses)                    # [T,H,W,3]
    conf = out["conf"]                                                  # [T,H,W]
    cams = poses[:, :3, 3].astype(np.float32)

    M, H, W = wp.shape[0], wp.shape[1], wp.shape[2]
    st = 2
    P = wp[:, ::st, ::st, :].reshape(-1, 3).astype(np.float32)
    C = (col[:, ::st, ::st, :].reshape(-1, 3).astype(np.float32) / 255.0).clip(0, 1)
    hh, ww = (H + st - 1) // st, (W + st - 1) // st
    T = np.repeat(np.arange(M, dtype=np.float32), hh * ww)
    F = conf[:, ::st, ::st].reshape(-1)

    m = np.isfinite(P).all(1)
    pos = F > 0
    m &= pos & (F >= np.percentile(F[pos], 45))
    P, C, T = P[m], C[m], T[m]
    meta["points_raw"] = int(len(P))

    Bg = _grav_basis(cams); P = (P @ Bg).astype(np.float32); cams = cams @ Bg
    # upright via camera down-axis (c2w col1)
    if float(poses[:, :3, 1].mean(0) @ Bg[:, 2]) > 0:
        P[:, 1] *= -1; P[:, 2] *= -1; cams[:, 1] *= -1; cams[:, 2] *= -1

    # walked-area clip (drop far global-drift geometry)
    if len(P):
        cxy0, cxy1 = cams[:, :2].min(0), cams[:, :2].max(0)
        mrg = 0.35 * float((cxy1 - cxy0).max()) + 1e-6
        inb = ((P[:, 0] > cxy0[0] - mrg) & (P[:, 0] < cxy1[0] + mrg) &
               (P[:, 1] > cxy0[1] - mrg) & (P[:, 1] < cxy1[1] + mrg))
        meta["walked_area_kept"] = [int(inb.sum()), int(len(inb))]
        P, C, T = P[inb], C[inb], T[inb]

    if ceiling_cut and len(P):
        up_h = P[:, 2]; lo, hi = np.percentile(up_h, 1), np.percentile(up_h, 99)
        keep = up_h <= lo + float(ceiling_keep) * (hi - lo)
        meta["ceiling_cut"] = {"keep_frac": round(float(ceiling_keep), 2),
                               "kept": int(keep.sum()), "of": int(len(keep))}
        P, C, T = P[keep], C[keep], T[keep]

    np.savez_compressed(os.path.join(out_dir, "recon.npz"),
                        cams=cams.astype(np.float32), poses=poses)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    tnorm = (T - T.min()) / max(1.0, float(T.max() - T.min()))
    Ctime = cm.turbo(tnorm)[:, :3].astype(np.float32)

    import open3d as o3d
    span = float((np.percentile(P, 99, 0) - np.percentile(P, 1, 0)).mean())
    vox = max(1e-3, span / 550)
    def _cloud(colors, path):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(P.astype(np.float64))
        p.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        p = p.voxel_down_sample(voxel_size=vox)
        o3d.io.write_point_cloud(path, p); return p
    pcd = _cloud(C, os.path.join(out_dir, "cloud.ply"))
    _cloud(Ctime, os.path.join(out_dir, "cloud_viz.ply"))
    meta["cloud_points"] = len(pcd.points)

    _render(out_dir, P, C, Ctime, cams, meta)

    meta["total_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print("products:", sorted(os.listdir(out_dir)))
    print("meta:", json.dumps(meta, ensure_ascii=False))
    return meta


def _render(out_dir, P, Crgb, Ctime, cams, meta):
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BG = "#0a0e13"
    x, y = P[:, 0], P[:, 1]; cx_, cy_ = cams[:, 0], cams[:, 1]
    xl, xh = np.percentile(x, 1), np.percentile(x, 99)
    yl, yh = np.percentile(y, 1), np.percentile(y, 99)
    k = (x > xl) & (x < xh) & (y > yl) & (y < yh)
    fig = plt.figure(figsize=(13, 9)); ax = fig.add_subplot(111); ax.set_facecolor(BG)
    ax.hexbin(x[k], y[k], gridsize=300, cmap="magma", bins="log", linewidths=0)
    ax.plot(cx_, cy_, color="#35d0c0", lw=2.4)
    ax.scatter(cx_[0], cy_[0], c="#7CFF6B", s=90, edgecolors="white", linewidths=1.2, zorder=5)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); fig.patch.set_facecolor(BG)
    fig.savefig(os.path.join(out_dir, "floorplan.png"), dpi=130, facecolor=BG,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    sel = np.random.default_rng(0).choice(len(P), min(500000, len(P)), replace=False)
    Q, QC = P[sel], np.clip(Crgb[sel], 0, 1)
    lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c=QC, s=.45, linewidths=0)
    ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], color="#e6edf3", lw=1.1)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]))
    ax.view_init(elev=32, azim=-110); ax.axis("off")
    fig.savefig(os.path.join(out_dir, "view_ob.png"), dpi=120, facecolor=BG,
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    fig = plt.figure(figsize=(11, 9)); ax = fig.add_subplot(111); ax.set_facecolor(BG)
    ax.scatter(x[k], y[k], c=Ctime[k], s=.4, linewidths=0)
    ax.plot(cx_, cy_, color="#e6edf3", lw=1.0)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); fig.patch.set_facecolor(BG)
    fig.savefig(os.path.join(out_dir, "view_top.png"), dpi=120, facecolor=BG,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("out_dir")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--model", default=None, help="path to abot_recon .onnx")
    ap.add_argument("--ceiling-cut", action="store_true")
    ap.add_argument("--ceiling-keep", type=float, default=0.85)
    args = ap.parse_args()
    run(args.video, args.out_dir, fps=args.fps, model_id=args.model,
        ceiling_cut=args.ceiling_cut, ceiling_keep=args.ceiling_keep)
