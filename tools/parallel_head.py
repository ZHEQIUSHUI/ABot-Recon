"""Vectorized (parallel) reimplementation of AdjacentPoseHead's per-pair computation.

Reuses the trained submodules of the original head (shared weights), but computes all
adjacent pairs (i-1, i) as a batch instead of a sequential python loop, and emits:
  raw_delta  [B, N-1, 4, 4]  (the per-pair relative pose, BEFORE rotation correction)
  resid      [B, N-1, 3]     (rotation-correction rotvec from TemporalRotationRefiner)
The sequential composition (pose[i] = pose[i-1] @ corrected_delta) is left to the runtime
(numpy) — validated to reproduce torch camera_poses exactly.

The sliding-window part of TemporalRotationRefiner is replicated with explicit per-pair
windows over the fused-feature sequence (left zero padding + age embedding on valid
positions), exactly mirroring the reference feature_buffer semantics.
"""
import torch
import torch.nn as nn


class ParallelPoseHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head          # the trained AdjacentPoseHead (weights shared)
        self.rc = head.rot_correction

    def forward(self, feat):      # feat: [B, N, tokens, C]
        head, rc = self.head, self.rc
        B, N = feat.shape[0], feat.shape[1]
        P = N - 1
        desc = head._describe_frames(feat)                    # [B, N, 512]
        frame_tokens = feat[:, :, head.num_pose_tokens:]      # [B, N, T, C]

        prev_d, curr_d = desc[:, :-1], desc[:, 1:]            # [B, P, 512]

        # ---- raw delta (_predict_delta), pairs as batch ----
        pair = torch.cat([prev_d, curr_d, curr_d - prev_d, curr_d * prev_d], -1)
        hidden = head.pair_mlp(pair)
        t = head.delta_t_head(hidden.float())                 # [B, P, 3]
        q = head.delta_q_head(hidden.float())
        R = head._quat_to_mat(q.float())                      # [B, P, 3, 3]
        top = torch.cat([R.to(t.dtype), t[..., None]], -1)    # [B, P, 3, 4]
        bot = torch.tensor([0.0, 0, 0, 1], dtype=t.dtype, device=t.device).view(1, 1, 1, 4).expand(B, P, 1, 4)
        raw_delta = torch.cat([top, bot], -2)                 # [B, P, 4, 4]

        # ---- TemporalRotationRefiner, per-pair part as batch ----
        pd = prev_d.reshape(B * P, -1).float()
        cd = curr_d.reshape(B * P, -1).float()
        desc_feature = rc.desc_proj(torch.cat([pd, cd, cd - pd, cd * pd], -1))

        pt = frame_tokens[:, :-1].reshape(B * P, *frame_tokens.shape[2:]).float()  # [BP, T, C]
        ct = frame_tokens[:, 1:].reshape(B * P, *frame_tokens.shape[2:]).float()
        pm, cm = pt.mean(1), ct.mean(1)
        frame_query = rc.frame_query_proj(torch.cat([pm, cm, cm - pm, cm * pm], -1))[:, None, :]
        memory = rc.frame_proj(torch.cat([pt, ct], 1))
        role = rc.frame_role_embed.to(device=memory.device, dtype=memory.dtype)
        T = pt.shape[1]
        memory = torch.cat([memory[:, :T] + role[0], memory[:, T:] + role[1]], 1)
        frame_out, _ = rc.frame_attn(frame_query, memory, memory, need_weights=False)
        temporal_context = rc.frame_norm(frame_query + rc.frame_dropout(frame_out)).squeeze(1)
        fused = rc.fuse_proj(torch.cat([desc_feature, temporal_context], -1))
        fused = fused.reshape(B, P, -1)                       # [B, P, 512] pair sequence

        # ---- sliding windows (feature_buffer semantics): window_j = fused[j-K+1 .. j] ----
        K = rc.kernel_size
        pad = fused.new_zeros(B, K - 1, fused.shape[-1])
        padded = torch.cat([pad, fused], 1)                   # [B, P+K-1, 512]
        wins = torch.stack([padded[:, j:j + K] for j in range(P)], 1)   # [B, P, K, 512]
        if rc.age_embed is not None:
            age_ids = torch.arange(K - 1, -1, -1, device=fused.device)
            age = rc.age_embed(age_ids).to(dtype=wins.dtype)            # [K, 512]
            # valid mask per pair j: first max(0, K-1-j) positions are padding
            valid = torch.ones(P, K, device=fused.device, dtype=wins.dtype)
            for j in range(min(P, K - 1)):
                valid[j, : K - 1 - j] = 0
            wins = wins + age[None, None] * valid[None, :, :, None]

        temporal = wins.reshape(B * P, K, -1).transpose(1, 2).float()   # [BP, 512, K]
        h = (rc.conv(temporal) * torch.sigmoid(rc.gate(temporal))).squeeze(-1)
        resid = (rc.max_rad * torch.tanh(rc.out(h))).reshape(B, P, 3)

        return raw_delta, resid
