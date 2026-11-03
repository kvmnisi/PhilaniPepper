import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "frontend" / "assets" / "pepper3d" / "pepper-model.json"


def test_model_is_a_single_tree_with_all_visual_meshes():
    model = json.loads(MODEL.read_text())
    assert model["root"] == "base_link" and model["up_axis"] == "Z"
    children = [j["child"] for j in model["joints"]]
    assert len(children) == len(set(children))                      # every link has one parent
    assert set(children) | {model["root"]} == set(model["links"])   # and every link is reachable
    meshes = {v["mesh"] for visuals in model["links"].values() for v in visuals}
    assert len(meshes) == 47 and "Torso.dae" in meshes and "HeadPitch.dae" in meshes
    joint_names = {j["name"] for j in model["joints"] if j["type"] == "revolute"}
    assert {"HeadYaw", "HeadPitch", "LShoulderPitch", "RElbowRoll", "HipPitch"} <= joint_names


def test_licensed_meshes_are_never_committed():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert not [f for f in tracked if f.lower().endswith((".dae", ".stl"))]
    ignored = subprocess.run(["git", "check-ignore", "frontend/assets/pepper3d/meshes/Torso.dae"], cwd=ROOT)
    assert ignored.returncode == 0
    assert "assets/pepper3d/meshes/" in (ROOT / "frontend" / ".vercelignore").read_text()


def test_installer_requires_typed_acceptance(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("installer", ROOT / "scripts" / "install_pepper_meshes.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    monkeypatch.setattr(installer, "DEST", tmp_path / "meshes")
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert installer.main(["--from", str(tmp_path)]) == 1
    assert not (tmp_path / "meshes").exists()
