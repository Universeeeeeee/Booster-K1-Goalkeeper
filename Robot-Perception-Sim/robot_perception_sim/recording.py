"""Record one existing physical shot with paired exposure renders and passive truth."""

from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image

from acquisition.timing import exposure_window
from .camera import CAMERA_NAMES, calibration_record, intrinsics, load_camera_config, optical_pose
from .scene import ROOT, build_stereo_scene
from .stereo import StereoCamera


def capture_schedule(physics_times, start, duration, fps):
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("duration and fps must be finite and positive")
    times = np.asarray(physics_times)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("physics timestamps must be finite and strictly increasing")
    targets = start + np.arange(max(1, math.ceil(duration * fps - 1e-9))) / fps
    indices = np.searchsorted(times, targets - 1e-9)
    if start < times[0] - 1e-9 or indices[-1] >= len(times):
        raise ValueError("physics trace does not cover the capture interval")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("fps exceeds the available physical state sampling rate")
    return targets, indices


def find_ffmpeg(path=None):
    candidate = str(path) if path else shutil.which("ffmpeg")
    if candidate is None:
        sibling = Path(sys.executable).parent / "ffmpeg"
        candidate = str(sibling) if sibling.is_file() else None
    executable = shutil.which(candidate) if candidate else None
    if not executable:
        raise ValueError("FFmpeg not found: use the existing robot environment or pass --ffmpeg PATH")
    return executable


def encode_video(ffmpeg, frames, output, fps, count):
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-n", "-framerate", str(fps),
                    "-i", str(frames / "%06d.png"), "-frames:v", str(count),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(output)], check=True)


def record_shot(shot, output, camera_config=None, fps=60, ffmpeg=None):
    config = load_camera_config() if camera_config is None else camera_config
    exposure = config["exposure_time_ms"] / 1000
    exposure_window(0, config["exposure_time_ms"], fps)
    exposure_window(0, config["exposure_time_ms"], config["reference_fps"])
    # Reuse the existing settle/launch/physics/control/contact recorder as a whole.
    from k1_goalkeeper.impact_validation import ImpactCase, run_impact
    from k1_goalkeeper.scene import K1_XML
    from k1_goalkeeper.walk_controller import DEPLOY_ROOT
    import torch

    ffmpeg = find_ffmpeg(ffmpeg)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Recording output must stay inside Robot-Perception-Sim/results")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output-dir")
    if config["image_width_px"] % 2 or config["image_height_px"] % 2:
        raise ValueError("H.264 yuv420p output requires even image dimensions")
    output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(build_stereo_scene(output / "scene.xml", config)))
    if fps > 1 / model.opt.timestep:
        raise ValueError("fps cannot exceed the physics sampling rate")
    print("Simulating once with the existing K1 standing controller (2 s settle)...", flush=True)
    case = ImpactCase(shot.name, duration=shot.duration + exposure, seed=shot.seed)
    launch = {"position": list(shot.position), "velocity": list(shot.velocity),
              "angular_velocity": list(shot.angular_velocity)}
    summary, trace = run_impact(model, case, launch=launch)
    (output / "simulation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not summary["completed"] or summary["numerical_error"] or summary["policy_stop"] or summary["fall_events"]:
        raise RuntimeError(f"Physical run did not complete safely; inspect {output / 'simulation.json'}")
    start = summary["launch"]["time"]
    targets, indices = capture_schedule(trace["time"], start, shot.duration, fps)
    data = mujoco.MjData(model)
    rows, poses, visible, diagnostics = [], [], [], []
    subframe_poses = {"0ms": [], "exposed": []}
    with ExitStack() as stack:
        cameras = {
            "0ms": stack.enter_context(StereoCamera(model, {**config, "exposure_time_ms": 0})),
            "exposed": stack.enter_context(StereoCamera(model, config)),
        }
        for variant in cameras:
            for side in ("left", "right"):
                (output / variant / side).mkdir(parents=True)
        for frame_id, (target, index) in enumerate(zip(targets, indices)):
            data.time = trace["time"][index]
            data.qpos[:] = trace["qpos"][index]
            data.qvel[:] = trace["qvel"][index]
            mujoco.mj_forward(model, data)
            paths = {}
            exposures = {}
            for variant, camera in cameras.items():
                frame = camera.capture(data)
                subframe_poses[variant].append(frame["sample_T_world_from_optical"])
                exposures[variant] = {key: value for key, value in frame.items()
                                      if key not in ("left", "right", "sample_T_world_from_optical")}
                paths[variant] = {}
                for side in ("left", "right"):
                    path = Path(variant) / side / f"{frame_id:06d}.png"
                    Image.fromarray(frame[side]).save(output / path)
                    paths[variant][side] = path.as_posix()
            # Diagnostic visibility uses segmentation only; it never enters RGB observations.
            renderer = cameras["0ms"].renderer
            counts = []
            renderer.enable_segmentation_rendering()
            try:
                for name in CAMERA_NAMES:
                    renderer.update_scene(data, camera=name)
                    labels = renderer.render()
                    counts.append(int(np.count_nonzero((labels[:, :, 0] == model.geom("ball_collision").id)
                                  & (labels[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM))))
            finally:
                renderer.disable_segmentation_rendering()
            visible.append(counts)
            poses.append([optical_pose(data, name) for name in CAMERA_NAMES])
            end = data.time + exposure
            end_index = int(np.searchsorted(trace["time"], end - 1e-9))
            if end_index >= len(trace["time"]):
                raise RuntimeError("Physical trace does not cover the full exposure")
            actual_end = trace["time"][end_index]
            predicted = trace["ball_position"][index] + trace["ball_velocity"][index] * (actual_end - data.time)
            error = float(np.linalg.norm(predicted - trace["ball_position"][end_index]))
            contacts = trace["contacts"]
            during = (contacts[:, 0] < end - 1e-9) & (contacts[:, 1] > data.time + 1e-9)
            diagnostics.append({"frame_id": frame_id, "ball_visible_pixels_left_right": counts,
                                "exposure_has_ball_contact": bool(during.any()),
                                "exposure_end_reference_time_s": float(actual_end),
                                "constant_velocity_endpoint_error_m": error})
            rows.append({"frame_id": frame_id, "target_timestamp_s": float(target),
                         "timestamp_s": float(data.time), "shot_time_s": float(data.time - start),
                         "left_timestamp_s": float(data.time), "right_timestamp_s": float(data.time),
                         "sampling_error_s": float(data.time - target), "ground_truth_index": frame_id,
                         "exposure_end_s": {"0ms": float(data.time), "exposed": float(end)},
                         "exposures": exposures,
                         "images": paths})
            if (frame_id + 1) % 15 == 0 or frame_id + 1 == len(indices):
                print(f"Rendered {frame_id + 1}/{len(indices)} stereo exposure pairs", flush=True)
    (output / "frames.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    physics_start = int(np.searchsorted(trace["time"], start - 1e-9))
    np.savez_compressed(output / "ground_truth.npz", timestamp_s=trace["time"][indices],
                        ball_position_world_m=trace["ball_position"][indices],
                        ball_velocity_world_mps=trace["ball_velocity"][indices],
                        T_world_from_optical=np.asarray(poses),
                        physics_time_s=trace["time"][physics_start:],
                        physics_qpos=trace["qpos"][physics_start:],
                        physics_qvel=trace["qvel"][physics_start:],
                        exposure_0ms_T_world_from_optical=np.asarray(subframe_poses["0ms"]),
                        exposure_exposed_T_world_from_optical=np.asarray(subframe_poses["exposed"]),
                        physics_ball_position_world_m=trace["ball_position"][physics_start:],
                        physics_ball_velocity_world_mps=trace["ball_velocity"][physics_start:],
                        contacts=trace["contacts"])
    for variant in cameras:
        for side in ("left", "right"):
            encode_video(ffmpeg, output / variant / side, output / f"{side}_{variant}.mp4", fps, len(indices))
    diagnostics_summary = {
        "frames_visible_in_both_eyes": int(np.all(np.asarray(visible) > 0, axis=1).sum()),
        "frames_not_visible_in_either_eye": int(np.all(np.asarray(visible) == 0, axis=1).sum()),
        "frames_with_ball_contact_during_exposure": sum(row["exposure_has_ball_contact"] for row in diagnostics),
        "max_constant_velocity_endpoint_error_m": max(row["constant_velocity_endpoint_error_m"] for row in diagnostics),
        "max_sampling_error_s": max(row["sampling_error_s"] for row in rows),
    }
    (output / "diagnostics.json").write_text(json.dumps({"summary": diagnostics_summary, "frames": diagnostics}, indent=2) + "\n")
    metadata = {
        "schema_version": 2, "shot": asdict(shot), "launch_time_s": start,
        "fps": fps, "frame_count": len(indices), "video_duration_s": len(indices) / fps,
        "physics_dt_s": float(model.opt.timestep), "camera_config": config,
        "K": intrinsics(config).tolist(), "timestamp_reference": "exposure_start",
        "calibration": calibration_record(config),
        "pose_source": "sim_oracle", "source_kind": "mujoco_rgb_only", "depth_source": "missing",
        "acquisition_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "robot_perception_sim/stereo.py",
                         ROOT / "robot_perception_sim/camera.py", ROOT / "acquisition/timing.py")},
        "exposures_ms": {"0ms": 0, "exposed": config["exposure_time_ms"]},
        "control": "existing run_impact / SceneSimulation / k1_walk; zero commanded velocity",
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "git_revision": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "robot_model_sha256": hashlib.sha256(K1_XML.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256((DEPLOY_ROOT / "tasks/locomotion/robots/k1/models/k1_walk.pt").read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256((output / "scene.xml").read_bytes()).hexdigest(),
        "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        "coordinates": "world X/Y horizontal, Z up, metres/seconds; optical X right/Y down/Z forward; T uses column vectors",
        "camera_order": ["left", "right"], "camera_pose_time_reference": "exposure_start",
        "limitations": "Offline rendering; 0ms and exposed share one physical trajectory. Exposure uses constant qvel, not contact/gravity integration. MP4 uses nominal CFR; frames.jsonl records actual <=1ms sampling jitter. Visibility is instantaneous diagnostic segmentation, not RGB detection. No real-camera latency/performance claim.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return diagnostics_summary
