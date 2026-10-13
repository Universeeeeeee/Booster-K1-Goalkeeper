"""Export only official K1 walk weights and their Python dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/BoosterRobotics/booster_deploy.git"
REVISION = "7bb1462e2ac742eba5c01659b815e2e99fbe617f"
DEST = ROOT / "assets/booster_deploy"
FILES = [
    "LICENSE", "README.md",
    "booster_deploy/__init__.py",
    "booster_deploy/controllers/__init__.py",
    "booster_deploy/controllers/base_controller.py",
    "booster_deploy/controllers/controller_cfg.py",
    "booster_deploy/controllers/mujoco_controller.py",
    "booster_deploy/robots/k1.py",
    "booster_deploy/utils/__init__.py",
    "booster_deploy/utils/isaaclab/array.py",
    "booster_deploy/utils/isaaclab/configclass.py",
    "booster_deploy/utils/isaaclab/dict.py",
    "booster_deploy/utils/isaaclab/math.py",
    "booster_deploy/utils/isaaclab/string.py",
    "booster_deploy/utils/policy_runner.py",
    "booster_deploy/utils/registry.py",
    "tasks/__init__.py",
    "tasks/locomotion/__init__.py",
    "tasks/locomotion/locomotion.py",
    "tasks/locomotion/robots/__init__.py",
    "tasks/locomotion/robots/k1/__init__.py",
    "tasks/locomotion/robots/k1/models/k1_walk.pt",
]


def export(source):
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": {}}
    blobs = {}
    for name in FILES:
        content = subprocess.check_output(["git", "-C", str(source), "show", f"{REVISION}:{name}"])
        path = DEST / name
        if path.exists() and path.read_bytes() != content:
            raise SystemExit(f"Existing file differs from pinned upstream: {path}")
        blobs[name] = content
        manifest["files"][name] = hashlib.sha256(content).hexdigest()
    for name, content in blobs.items():
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(FILES)} official files (K1 only) to {DEST} @ {REVISION}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Reuse a local upstream Git repository")
    args = parser.parse_args()
    if args.source:
        export(args.source)
    else:
        with tempfile.TemporaryDirectory(prefix="k1-walk-") as directory:
            checkout = Path(directory) / "upstream"
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            REPOSITORY, str(checkout)], check=True)
            export(checkout)
