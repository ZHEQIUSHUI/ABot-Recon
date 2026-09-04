"""Export the full ABot recon as ONNX (fixed-N chunk graph, torch-free runtime friendly).

Graph inputs : imgs [1, N, 3, 280, 504]  (N fixed at export; runtime slides chunks w/ overlap)
Graph outputs: local_points [1,N,280,504,3], conf [1,N,280,504,1],
               raw_delta [1,N-1,4,4], resid [1,N-1,3]
The sequential pose composition + local->world stays in numpy (validated exact).

The original AdjacentPoseHead is stubbed out of net.forward (its python loop + in-place
index_put is untraceable); the vectorized ParallelPoseHead (bit-exact, validated) runs on
the captured head input features instead.
"""
import os, sys, traceback
import torch
import torch.nn as nn

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from parallel_head import ParallelPoseHead

N = int(os.environ.get("EXPORT_N", "32"))
OUT = os.environ.get("EXPORT_OUT", f"{SP}/abot_recon_n{N}.onnx")


class HeadStub(nn.Module):
    """Captures the pose-head input feat; returns identity poses (net.forward only uses
    them for world-points we discard)."""
    def __init__(self):
        super().__init__()
        self.feat = None

    def forward(self, feat, camera_state=None, return_state=False):
        self.feat = feat
        B, n = feat.shape[0], feat.shape[1]
        eye = torch.eye(4, dtype=feat.dtype, device=feat.device)[None, None].expand(B, n, 4, 4)
        return (eye, {}) if return_state else eye


class ExportRecon(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.stub = HeadStub()
        self.phead = ParallelPoseHead(net.camera_head)
        net.camera_head = self.stub          # detach sequential head from the traced path

    def forward(self, imgs):
        out = self.net.forward(imgs, causal_global_attn=True, streaming_inference=False,
                               stream_use_cache=False, long_sequence_parallel=True)
        raw_delta, resid = self.phead(self.stub.feat)
        return out["local_points"], out["conf"], raw_delta, resid


def _install_chunked_sdpa(q_chunk_tokens: int = 2900):
    """CPU sdpa materializes the full [L,L] attention matrix (34GB/layer at N=32).
    Chunk the QUERY dim (math identical, mask rows sliced along) so peak memory drops
    ~10x; the export unrolls the fixed-count loop into the graph — fine for fixed N."""
    import torch.nn.functional as F
    orig = F.scaled_dot_product_attention

    def chunked(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        L = q.shape[-2]
        if L <= q_chunk_tokens or is_causal:
            return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, **kw)
        outs = []
        for s in range(0, L, q_chunk_tokens):
            e = min(s + q_chunk_tokens, L)
            m = attn_mask[..., s:e, :] if (attn_mask is not None and attn_mask.dim() >= 2
                                           and attn_mask.shape[-2] == L) else attn_mask
            outs.append(orig(q[..., s:e, :], k, v, attn_mask=m,
                             dropout_p=dropout_p, is_causal=False, **kw))
        return torch.cat(outs, dim=-2)

    F.scaled_dot_product_attention = chunked


def _install_bias_memo():
    """streaming_chunk_attention_bias is a pure function of scalar args, but every layer
    recomputes the same (Lq,Lk) matrix (2.1GB at N=32) and the tracer keeps ALL of them
    (36x -> 77GB OOM). Memoize per-args so all layers reuse ONE traced tensor."""
    import abot_recon.modeling.streaming.core.window_mask as wm
    import abot_recon.modeling.streaming.attention as att
    import abot_recon.modeling.streaming.core.chunk_attention as ca
    orig = wm.streaming_chunk_attention_bias
    cache = {}

    def memo(num_past_frames, num_new_frames, tokens_per_frame, dtype, device, *,
             temporal_window_frames=12, num_reference_frames=0):
        key = (int(num_past_frames), int(num_new_frames), int(tokens_per_frame),
               str(dtype), str(device), int(temporal_window_frames), int(num_reference_frames))
        if key not in cache:
            cache[key] = orig(num_past_frames, num_new_frames, tokens_per_frame, dtype,
                              device, temporal_window_frames=temporal_window_frames,
                              num_reference_frames=num_reference_frames)
        return cache[key]

    wm.streaming_chunk_attention_bias = memo
    att.streaming_chunk_attention_bias = memo
    ca.streaming_chunk_attention_bias = memo


def main():
    from abot_recon import ABotRecon
    print(f"load model (cpu), export N={N} ...", flush=True)
    api = ABotRecon.from_pretrained("acvlab/ABot-Recon", device="cpu",
                                    attention_backend="sdpa", loop_closure=False)
    net = api.model.network if hasattr(api.model, "network") else api.model
    for m in net.modules():
        if hasattr(m, "interpolate_antialias"):
            m.interpolate_antialias = False
    _install_chunked_sdpa()
    _install_bias_memo()
    # normalization moves to the runtime (graph input = normalized imgs); record the
    # constants in a sidecar json the ORT runtime loads.
    os.environ["ABOT_PRENORMALIZED"] = "1"
    os.environ["ABOT_EXPORT_STATIC_ROPE"] = "1"
    os.environ["ABOT_EXPORT_REAL_ROPE"] = "1"
    if getattr(net, "rope3d_embed", None) is not None:
        # precompute EAGERLY so torch.angle (complex op) never enters the export graph
        net.rope3d_embed._freqs_angle = torch.angle(net.rope3d_embed.freqs).to(torch.float32)
        print("rope3d angle buffer precomputed:", tuple(net.rope3d_embed._freqs_angle.shape), flush=True)
    mean = net.image_mean.flatten().tolist()
    std = net.image_std.flatten().tolist()
    import json
    json.dump({"image_mean": mean, "image_std": std, "N": N, "input": "prenormalized"},
              open(OUT.replace(".onnx", ".json"), "w"), indent=1)
    print("sidecar:", {"image_mean": mean, "image_std": std}, flush=True)
    w = ExportRecon(net).eval()
    x = torch.randn(1, N, 3, 280, 504)
    try:
        use_dynamo = os.environ.get("EXPORT_DYNAMO", "1") == "1"
        if use_dynamo:
            # two-step: torch.export first, then convert. The converter lowers some view ops
            # to prims.* which have no builtin ONNX mapping — supply custom translations
            # (both are pure Reshapes).
            from onnxscript import opset18 as oop

            def prims_collapse_view(x, start: int, end: int):
                shape = oop.Shape(x)
                pre = oop.Slice(shape, oop.Constant(value_ints=[0]),
                                oop.Constant(value_ints=[int(start)]), oop.Constant(value_ints=[0]))
                post = oop.Slice(shape, oop.Constant(value_ints=[int(end) + 1]),
                                 oop.Constant(value_ints=[2 ** 31 - 1]), oop.Constant(value_ints=[0]))
                new_shape = oop.Concat(pre, oop.Constant(value_ints=[-1]), post, axis=0)
                return oop.Reshape(x, new_shape)

            def prims_split_dim(x, dim: int, outer_length: int):
                shape = oop.Shape(x)
                pre = oop.Slice(shape, oop.Constant(value_ints=[0]),
                                oop.Constant(value_ints=[int(dim)]), oop.Constant(value_ints=[0]))
                post = oop.Slice(shape, oop.Constant(value_ints=[int(dim) + 1]),
                                 oop.Constant(value_ints=[2 ** 31 - 1]), oop.Constant(value_ints=[0]))
                mid = oop.Constant(value_ints=[int(outer_length), -1])
                return oop.Reshape(x, oop.Concat(pre, mid, post, axis=0))

            table = {
                torch.ops.prims.collapse_view.default: prims_collapse_view,
                torch.ops.prims.split_dim.default: prims_split_dim,
            }
            ep = torch.export.export(w, (x,))
            print("torch.export OK; converting to ONNX ...", flush=True)
            torch.onnx.export(ep, (x,), OUT, input_names=["imgs"],
                              output_names=["local_points", "conf", "raw_delta", "resid"],
                              dynamo=True, external_data=True, optimize=True,
                              custom_translation_table=table)
        else:
            torch.onnx.export(w, (x,), OUT, opset_version=17, input_names=["imgs"],
                              output_names=["local_points", "conf", "raw_delta", "resid"],
                              do_constant_folding=False)
        total = os.path.getsize(OUT)
        for f in os.listdir(os.path.dirname(OUT) or "."):
            p = os.path.join(os.path.dirname(OUT) or ".", f)
            if f.startswith(os.path.basename(OUT)) and f != os.path.basename(OUT):
                total += os.path.getsize(p)
        print(f"ONNX EXPORT OK: {OUT} total≈{total/1e9:.2f}GB", flush=True)
    except Exception:
        print("EXPORT FAILED:", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
