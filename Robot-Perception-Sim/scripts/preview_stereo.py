"""Add nominal ZED X stereo cameras to the existing K1 scene and save a snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
sys.dont_write_bytecode = True

import mujoco
import numpy as np
from PIL import Image

from k1_goalkeeper.scene import K1_XML, load_config
from robot_perception_sim.camera import (
    CAMERA_NAMES, DEFAULT_CONFIG, intrinsics, load_camera_config, optical_pose,
    project_world, vertical_fov_deg,
)
from robot_perception_sim.scene import build_stereo_scene
from robot_perception_sim.stereo import StereoCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/stereo")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-config", type=Path, help="Existing simulation.json format")
    parser.add_argument("--head-yaw", type=float, default=0, help="Radians")
    parser.add_argument("--head-pitch", type=float, default=0, help="Radians")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("--output-dir must stay inside Robot-Perception-Sim")
    config = load_camera_config(args.camera_config)
    scene_path = build_stereo_scene(output / "scene.xml", config, load_config(args.scene_config))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    for name, angle in (("aahead_yaw_joint", args.head_yaw), ("aahead_pitch_joint", args.head_pitch)):
        joint = model.joint(name)
        if not np.isfinite(angle) or not joint.range[0] <= angle <= joint.range[1]:
            parser.error(f"{name}: angle must be finite and inside {joint.range.tolist()}")
        data.qpos[joint.qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    with StereoCamera(model, config) as camera:
        frame = camera.capture(data)
    for side in ("left", "right"):
        Image.fromarray(frame[side]).save(output / f"{side}.png")
    with mujoco.Renderer(model, height=600, width=960) as renderer:
        renderer.update_scene(data, camera="goal_front")
        Image.fromarray(renderer.render()).save(output / "overview.png")
        closeup = mujoco.MjvCamera()
        closeup.type = mujoco.mjtCamera.mjCAMERA_FREE
        closeup.lookat[:] = data.body("zedx_mount").xpos
        closeup.distance, closeup.azimuth, closeup.elevation = 0.45, 30, -10
        renderer.update_scene(data, camera=closeup)
        Image.fromarray(renderer.render()).save(output / "mount.png")
    calibration = intrinsics(config)
    poses = {side: optical_pose(data, name) for side, name in zip(("left", "right"), CAMERA_NAMES)}
    ball_world = data.body("ball").xpos.copy()
    projections = {}
    for side, pose in poses.items():
        try:
            uv, depth = project_world(ball_world, calibration, pose)
            projections[side] = {"uv_px": uv.tolist(), "optical_depth_m": depth}
        except ValueError:
            projections[side] = {"uv_px": None, "optical_depth_m": None}
    metadata = {
        "schema_version": 1,
        "backend": "MuJoCo stereo RGB, not Stereolabs SDK",
        "mujoco_version": mujoco.__version__,
        "git_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "camera_config": config,
        "camera_config_sha256": hashlib.sha256(args.camera_config.read_bytes()).hexdigest(),
        "robot_model_revision": "12a97516f57469078707afb587a5f84c1cb110d1",
        "robot_model_sha256": hashlib.sha256(K1_XML.read_bytes()).hexdigest(),
        "camera_asset_revision": "22468247c1653d08a1e213809f3e9ce76291487d",
        "camera_mesh_sha256": hashlib.sha256((ROOT / "models/zedx.stl").read_bytes()).hexdigest(),
        "image_size_px": [config["image_width_px"], config["image_height_px"]],
        "K": calibration.tolist(),
        "derived_vertical_fov_deg": vertical_fov_deg(config),
        "timestamp_s": frame["timestamp_s"],
        "left_timestamp_s": frame["timestamp_s"], "right_timestamp_s": frame["timestamp_s"],
        "T_world_from_left_optical": poses["left"].tolist(),
        "T_world_from_right_optical": poses["right"].tolist(),
        "T_left_from_right_optical": (np.linalg.inv(poses["left"]) @ poses["right"]).tolist(),
        "measured_baseline_m": float(np.linalg.norm(poses["right"][:3, 3] - poses["left"][:3, 3])),
        "head_yaw_pitch_rad": [args.head_yaw, args.head_pitch],
        "diagnostic_ground_truth": {"ball_world_m": ball_world.tolist(), "ball_projection": projections},
        "assumptions": "Static mj_forward; ideal instantaneous stereo, zero distortion; no hardware calibration or realtime FPS claim",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved stereo RGB, overview and geometry metadata: {output}")


if __name__ == "__main__":
    main()
