"""Generate and compile a complete MJCF scene."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from k1_goalkeeper.scene import build_scene, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-mode", choices=["free", "fixed", "absent"], default="free")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(build_scene(args.output, args.robot_mode, load_config(args.config)))
