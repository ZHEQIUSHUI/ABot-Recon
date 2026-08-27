from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_python_has_no_experiment_machine_paths():
    forbidden = (
        "/" + "mnt/",
        "source" + "_root",
        "baseline_" + "rot_" + "corr",
        "eval_" + "3R",
    )
    violations = []
    for path in (ROOT / "abot_recon").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert not violations, "\n".join(violations)


def test_release_contains_no_training_entrypoints():
    names = {path.name for path in (ROOT / "abot_recon").rglob("*.py")}
    assert "pi3_training.py" not in names
    assert "loss.py" not in names


def test_release_contains_only_the_final_streaming_architecture():
    modeling = ROOT / "abot_recon" / "modeling"
    layers = modeling / "pi3" / "models" / "layers"
    assert not (modeling / ("long_" + "pi3")).exists()
    assert not (modeling / ("hybrid_" + "long_" + "pi3")).exists()
    assert (modeling / "streaming" / "network.py").is_file()
    assert (layers / "adjacent_pose_head.py").is_file()
    assert not (layers / "relative_camera_head.py").exists()
    assert not (layers / "relative_pi3_camera_head.py").exists()

    forbidden = (
        "anchor",
        "compact",
        "hybrid_" + "long_" + "pi3",
        "long_" + "pi3",
        "motion_mode",
        "use_residual_reference",
        "use_role_embed",
    )
    violations = []
    for path in (ROOT / "abot_recon").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert not violations, "\n".join(violations)


def test_release_loop_backend_is_self_contained():
    forbidden = ("horizon" + "stream", "loop_" + "horizon_root")
    violations = []
    for path in (ROOT / "abot_recon").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert not violations, "\n".join(violations)
    assert (ROOT / "abot_recon" / "sparse_loop" / "gpu_pgo.py").is_file()
