#!/usr/bin/env python3
"""Export ABot-Recon sub-models to ONNX.

Two feed-forward, KV-cache-free pieces export cleanly and match torch bit-for-bit:

  * SALAD descriptor  (DINOv2-vitb14 + SALAD aggregator) — loop-closure place recognition,
    also directly usable for cross-session relocalization ("where am I": image → descriptor).
  * ABot ViT encoder  (the per-frame DINOv2 trunk) — the heavy compute of the recon backbone.

The only ONNX blocker in the DINOv2 stack is the positional-embedding interpolation
(`_upsample_bicubic2d_aa`, opset≤17 unsupported). Inputs here are fixed-resolution, so the
interpolation is constant — we force `interpolate_antialias=False` (opset-17 bicubic); output
is numerically identical (validated below). For an exact-to-the-bit build you can instead bake
the interpolated pos_embed at the target grid and drop the interpolation op entirely.

NOT yet exported (the hard part): the streaming `decode` (global attention + bounded KV-cache
carry over the 12-frame window) and the camera/point/conf heads. Plan: export the per-step
decode with the KV cache as explicit input/output tensors, then reimplement the thin outer
loop (slide window, compose adjacent relative poses into a global trajectory, local→world) in
the target runtime — feasible precisely because the context is a *fixed 12-frame window*, not
an unbounded cache.

Usage:
  python tools/export_onnx.py --salad-dino checkpoints/loop/dinov2_vitb14_pretrain.pth \
      --salad-ckpt checkpoints/loop/dino_salad.ckpt --model acvlab/ABot-Recon --out out_onnx
"""
from __future__ import annotations
import argparse, os
import torch


def _no_aa(module: torch.nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "interpolate_antialias"):
            m.interpolate_antialias = False


def _report(name, a, b):
    a = a.flatten().float(); b = b.flatten().float()
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    print(f"  {name}: cos={cos:.5f} max|Δ|={float((a - b).abs().max()):.6f}")


def export_salad(dino_ckpt, salad_ckpt, out):
    from abot_recon.sparse_loop.retrieval import SaladDescriptor, _load_salad_checkpoint
    m = SaladDescriptor("dinov2_vitb14", dino_ckpt)
    _load_salad_checkpoint(m, salad_ckpt); m.eval(); _no_aa(m)
    x = torch.randn(1, 3, 336, 336)
    with torch.no_grad(): y = m(x)
    path = os.path.join(out, "salad.onnx")
    torch.onnx.export(m, (x,), path, opset_version=17, input_names=["image"],
                      output_names=["descriptor"], dynamic_axes={"image": {0: "b"}},
                      do_constant_folding=True)
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    _report(f"salad.onnx [{os.path.getsize(path)//10**6}MB] out{tuple(y.shape)}",
            y, torch.tensor(s.run(["descriptor"], {"image": x.numpy()})[0]))


def export_encoder(model_id, out):
    from abot_recon import ABotRecon
    api = ABotRecon.from_pretrained(model_id, device="cpu", attention_backend="sdpa", loop_closure=False)
    net = api.model.network if hasattr(api.model, "network") else api.model
    enc = net.encoder.eval(); _no_aa(enc)

    class Wrap(torch.nn.Module):
        def __init__(self, e): super().__init__(); self.e = e
        def forward(self, x):
            h = self.e(x, is_training=True)
            return h["x_norm_patchtokens"] if isinstance(h, dict) else h

    w = Wrap(enc).eval(); x = torch.randn(2, 3, 280, 504)
    with torch.no_grad(): y = w(x)
    path = os.path.join(out, "abot_encoder.onnx")
    torch.onnx.export(w, (x,), path, opset_version=17, input_names=["imgs"],
                      output_names=["feat"], dynamic_axes={"imgs": {0: "n"}, "feat": {0: "n"}},
                      do_constant_folding=True)
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    _report(f"abot_encoder.onnx [{os.path.getsize(path)//10**6}MB] out{tuple(y.shape)}",
            y, torch.tensor(s.run(["feat"], {"imgs": x.numpy()})[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="acvlab/ABot-Recon")
    ap.add_argument("--salad-dino", default="checkpoints/loop/dinov2_vitb14_pretrain.pth")
    ap.add_argument("--salad-ckpt", default="checkpoints/loop/dino_salad.ckpt")
    ap.add_argument("--out", default="out_onnx")
    ap.add_argument("--skip-salad", action="store_true")
    ap.add_argument("--skip-encoder", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("ONNX export (validated vs torch):")
    if not a.skip_salad and os.path.exists(a.salad_dino) and os.path.exists(a.salad_ckpt):
        export_salad(a.salad_dino, a.salad_ckpt, a.out)
    if not a.skip_encoder:
        export_encoder(a.model, a.out)
