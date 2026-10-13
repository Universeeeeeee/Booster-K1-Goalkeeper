#!/usr/bin/env python3
"""Run a preset suite, seeded random shots, or one explicitly configured shot."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from k1_goalkeeper.scene import build_scene, load_config
from k1_goalkeeper.simulation import Shot, preset_shots, random_shots, run_shot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preset", choices=tuple(preset_shots()), help="Named shot (default: center_low)")
    source.add_argument("--suite", action="store_true", help="Run all named shots")
    source.add_argument("--random", type=int, metavar="COUNT", help="Generate COUNT seeded random shots")
    parser.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "VZ"))
    parser.add_argument("--spin", type=float, nargs=3, metavar=("WX", "WY", "WZ"), help="Initial angular velocity, rad/s")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot-mode", choices=("free", "fixed", "absent"), default="free")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "simulation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "shots.json")
    args = parser.parse_args()
    custom = any(value is not None for value in (args.position, args.velocity, args.spin))
    if custom and (args.suite or args.random is not None or args.preset):
        parser.error("Custom position/velocity/spin cannot be combined with preset/suite/random")
    if args.random is not None and args.random < 1:
        parser.error("Random count must be positive")
    if args.duration <= 0 or not np.isfinite(args.duration):
        parser.error("Duration must be positive and finite")
    if custom:
        default = Shot()
        shots = [Shot(position=tuple(args.position or default.position),
                      velocity=tuple(args.velocity or default.velocity),
                      angular_velocity=tuple(args.spin or default.angular_velocity),
                      duration=args.duration, seed=args.seed)]
    elif args.random is not None:
        shots = random_shots(args.random, args.seed, args.duration)
    else:
        presets = preset_shots(args.seed, args.duration)
        shots = list(presets.values()) if args.suite else [presets[args.preset or "center_low"]]
    config = load_config(args.config)
    scene = build_scene(robot_mode=args.robot_mode, config=config)
    model = mujoco.MjModel.from_xml_path(str(scene))
    commit = subprocess.run(["git", "-C", str(ROOT / "assets" / "booster_assets"), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    output = {"environment": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                              "numpy": np.__version__, "platform": platform.platform(),
                              "asset_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                              "config_path": str(args.config.resolve()),
                              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                              "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest()},
              "results": [run_shot(model, shot, args.robot_mode) for shot in shots]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for result in output["results"]:
        print(f"{result['shot']['name']}: {result['outcome']} "
              f"(robot_contact={result['robot_contact']}, goal_contact={result['goal_contact']})")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
