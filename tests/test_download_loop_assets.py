import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_loop_assets.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("download_loop_assets", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_loop_asset_manifest_uses_official_sources():
    module = _load_script()
    assets = {asset.filename: asset.url for asset in module.ASSETS}

    assert set(assets) == {"dino_salad.ckpt", "dinov2_vitb14_pretrain.pth"}
    assert assets["dino_salad.ckpt"].startswith("https://github.com/serizba/salad/")
    assert assets["dinov2_vitb14_pretrain.pth"].startswith(
        "https://dl.fbaipublicfiles.com/dinov2/"
    )
