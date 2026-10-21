"""Reuse an existing physical shot and standing controller; record 0ms/exposed stereo videos."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
sys.dont_write_bytecode = True

from k1_goalkeeper.simulation import Shot, preset_shots
from robot_perception_sim.camera import DEFAULT_CONFIG, load_camera_config
from robot_perception_sim.recording import record_shot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(preset_shots()), default="right")
    parser.add_argument("--duration", type=float, default=1.0, help="Shot duration after the existing 2s settle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/moving_stereo")
    parser.add_argument("--ffmpeg", type=Path, help="Existing FFmpeg executable; defaults to PATH or beside Python")
    args = parser.parse_args()
    try:
        Shot(duration=args.duration)  # Reuse existing shot validation before constructing presets.
        shot = preset_shots(args.seed, args.duration)[args.preset]
        result = record_shot(shot, args.output_dir, load_camera_config(args.camera_config), args.fps, args.ffmpeg)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Saved stereo videos, timestamps, truth and diagnostics: {args.output_dir.resolve()}")
    print(result)


if __name__ == "__main__":
    main()
