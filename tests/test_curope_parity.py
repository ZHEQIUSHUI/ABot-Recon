import os
from pathlib import Path

import pytest
import torch


def reference_rope2d(tokens, positions, base=100.0):
    dim = tokens.shape[-1] // 2
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, device=tokens.device).float() / dim)
    )
    steps = torch.arange(
        int(positions.max()) + 1, device=tokens.device, dtype=inv_freq.dtype
    )
    freqs = torch.einsum("i,j->ij", steps, inv_freq).to(tokens.dtype)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos, sin = freqs.cos(), freqs.sin()

    def apply(values, indices):
        embedded_cos = torch.nn.functional.embedding(indices, cos)[:, None]
        embedded_sin = torch.nn.functional.embedding(indices, sin)[:, None]
        left, right = values.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return values * embedded_cos + rotated * embedded_sin

    y, x = tokens.chunk(2, dim=-1)
    return torch.cat(
        (apply(y, positions[:, :, 0]), apply(x, positions[:, :, 1])), dim=-1
    )


def import_curope_or_skip():
    try:
        from abot_recon.modeling.pi3.models.curope import curope2d
    except ImportError as exc:
        if os.environ.get("ABOT_RECON_REQUIRE_CUROPE") == "1":
            pytest.fail(f"required cuRoPE2D extension cannot be loaded: {exc}")
        pytest.skip(f"cuRoPE2D is not compiled: {exc}")
    return curope2d


def test_curope_uses_the_repository_local_extension():
    curope2d = import_curope_or_skip()
    assert Path(curope2d._kernels.__file__).resolve().parent == Path(
        curope2d.__file__
    ).resolve().parent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_curope_matches_reference_exactly(dtype):
    cuRoPE2D = import_curope_or_skip().cuRoPE2D
    torch.manual_seed(0)
    tokens = torch.randn(1, 16, 725, 64, device="cuda", dtype=dtype)
    positions = torch.randint(0, 37, (1, 725, 2), device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        expected = reference_rope2d(tokens.clone(), positions)
    module = cuRoPE2D()

    for _ in range(5):
        actual = module(tokens.clone(), positions)
        difference = (actual.float() - expected.float()).abs()
        assert torch.equal(actual, expected), (
            f"cuRoPE2D differs from the autocast reference: "
            f"max={difference.max().item()}, mean={difference.mean().item()}"
        )
