"""End-to-end validation: torch-free (ORT + numpy) pipeline vs torch ABotRecon.infer.

Checks, on real IMG_0754 frames:
  A. torch m.infer (streaming, GPU bf16)  -> reference camera_poses / local_points / conf
  B. OrtAbotRecon chunked (CPU fp32) + preprocess_rgb + numpy compose -> same outputs
Report cosine / max|delta| / trajectory-center error.
"""
import os, sys, numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from pathlib import Path
from PIL import Image

FR = "/home/qiushui/workspace/projects/pycode/walking_alarm/testdata/home_reconstruction/IMG_0754_frames"
ONNX = os.environ.get("ONNX", f"{SP}/abot_recon_n32.onnx")
T = int(os.environ.get("T", "64"))
WARMUP = int(os.environ.get("WARMUP", "24"))

paths = sorted(Path(FR).glob("*.jpg"))[:T]

# ---------- B: torch-free ----------
from ort_recon import OrtAbotRecon, preprocess_rgb, local_to_world
frames = [preprocess_rgb(np.asarray(Image.open(str(p)).convert("RGB"))) for p in paths]
ort_model = OrtAbotRecon(ONNX, warmup=WARMUP)
print(f"ORT graph N={ort_model.N}, warmup={ort_model.warmup}; running {T} frames ...", flush=True)
out = ort_model.infer(frames, progress=lambda e, t: print(f"  ort {e}/{t}", flush=True))
poses_b = out["camera_poses"]; lp_b = out["local_points"]; cf_b = out["conf"]

# ---------- A: torch reference ----------
import torch
from abot_recon import ABotRecon
m = ABotRecon.from_pretrained("acvlab/ABot-Recon", device="cuda",
                              attention_backend="sdpa", loop_closure=False)
r = m.infer(paths, output_local_points=True, output_confidence=True, loop_closure=False)
poses_a = r.camera_poses.float().cpu().numpy()
lp_a = r.local_points.float().cpu().numpy()
cf_a = r.confidence.float().cpu().numpy()

def cos(a, b):
    a = np.asarray(a, np.float64).flatten(); b = np.asarray(b, np.float64).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print("shapes:", poses_a.shape, poses_b.shape, lp_a.shape, lp_b.shape, cf_a.shape, cf_b.shape, flush=True)
c_a = poses_a[:, :3, 3]; c_b = poses_b[:, :3, 3]
scale = np.linalg.norm(c_a.max(0) - c_a.min(0)) + 1e-9
print(f"camera_poses : cos={cos(poses_a, poses_b):.6f}  center_err/scale={np.linalg.norm(c_a-c_b,axis=1).max()/scale:.5f}")
print(f"local_points : cos={cos(lp_a, lp_b):.6f}")
print(f"conf         : cos={cos(cf_a, cf_b):.6f}")
w_a = local_to_world(lp_a, poses_a); w_b = local_to_world(lp_b, poses_b)
print(f"world_points : cos={cos(w_a, w_b):.6f}")
print("VALIDATION DONE", flush=True)
