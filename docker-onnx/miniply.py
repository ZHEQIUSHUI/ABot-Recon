"""Tiny numpy-only replacements for the three things open3d did here:
binary PLY write/read (xyz float32 + rgb uint8) and voxel downsampling (per-cell mean).
Dropping open3d frees the container from the py<=3.11 pin (aarch64 open3d 0.18) so we can
run py3.12 + onnxruntime-gpu."""
from __future__ import annotations
import numpy as np

_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def write_ply(path: str, points: np.ndarray, colors01: np.ndarray) -> int:
    """points [N,3] float, colors01 [N,3] in 0..1 -> binary_little_endian PLY."""
    n = len(points)
    rec = np.empty(n, dtype=_DTYPE)
    rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
    c = (np.clip(colors01, 0, 1) * 255).round().astype(np.uint8)
    rec["red"], rec["green"], rec["blue"] = c[:, 0], c[:, 1], c[:, 2]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)
    return n


def read_ply(path: str):
    """Read a PLY we wrote (or any binary_little_endian xyz-f4 + rgb-u1 vertex layout).
    Returns (points [N,3] float32, colors01 [N,3] float32)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"not a PLY: {path}")
        n = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"bad PLY header: {path}")
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s == "end_header":
                break
        rec = np.fromfile(f, dtype=_DTYPE, count=n)
    pts = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float32)
    col = np.stack([rec["red"], rec["green"], rec["blue"]], 1).astype(np.float32) / 255.0
    return pts, col


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float):
    """Per-voxel MEAN of points and colors (same semantics as open3d voxel_down_sample)."""
    if voxel <= 0 or not len(points):
        return points, colors
    cells = np.floor(points / voxel).astype(np.int64)
    # row-wise unique via void view (fast, no python loop)
    key = np.ascontiguousarray(cells).view([("", cells.dtype)] * 3).ravel()
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    m = len(cnt)
    psum = np.zeros((m, 3), np.float64)
    csum = np.zeros((m, 3), np.float64)
    np.add.at(psum, inv, points.astype(np.float64))
    np.add.at(csum, inv, colors.astype(np.float64))
    w = cnt[:, None].astype(np.float64)
    return (psum / w).astype(np.float32), (csum / w).astype(np.float32)
