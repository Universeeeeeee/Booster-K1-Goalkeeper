"""Read back G3 PNGs and state archives without loading MuJoCo or a renderer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def calibration_digest(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def red_profile(rgb):
    signal = np.maximum(0, rgb[..., 0].astype(float) - np.maximum(rgb[..., 1], rgb[..., 2]))
    signal[signal <= 5] = 0
    weights = signal.sum(axis=0)
    xs = np.flatnonzero(weights)
    require(len(xs) > 0, "No visible red target in image")
    return float(np.dot(np.arange(len(weights)), weights) / weights.sum()), int(xs[0]), int(xs[-1])


def verify_capture(root):
    root = Path(root).resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    require(metadata["source"] == "mujoco_existing_stereo_capture", "Unexpected acquisition provenance")
    require(metadata["exposure_model"] == "constant_qvel_encoded_rgb_average",
            "Exposure model must describe the implemented RGB approximation")
    calibration = metadata["calibration"]
    digest = calibration_digest(calibration)
    require(metadata["calibration_sha256"] == digest, "Calibration digest mismatch")
    K = np.asarray(calibration["K"], dtype=float)
    require(K.shape == (3, 3) and np.isfinite(K).all() and K[0, 0] > 0,
            "Invalid calibration intrinsics")
    require(np.isfinite(calibration["baseline_m"]) and calibration["baseline_m"] > 0,
            "Invalid stereo baseline")
    width, height = calibration["resolution"]
    dt = float(metadata["physics_dt_s"])
    require(np.isfinite(dt) and dt > 0, "Invalid physics interval")
    rows = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    require(bool(rows), "Empty frame recording")
    archives, case_indices, metrics = {}, {}, []

    def local_path(name):
        path = (root / name).resolve()
        require(path.is_relative_to(root), "Artifact path escapes recording")
        return path

    try:
        for row in rows:
            case = f"{row['fps']}hz_{row['exposure_ms']}ms_{row['motion']}"
            index = row["frame_index"]
            require(index == case_indices.get(case, 0), "Frame indices must be contiguous per case")
            case_indices[case] = index + 1
            fps, exposure_ms = float(row["fps"]), float(row["exposure_ms"])
            times = np.array([row[key] for key in ("scheduled_time_s", "sim_state_time_s",
                             "sampling_error_s", "exposure_start_s", "exposure_mid_s", "exposure_end_s")], dtype=float)
            require(np.isfinite(times).all() and np.isfinite([fps, exposure_ms]).all() and fps > 0,
                    "Nonfinite frame timing")
            scheduled, state_time, error, start, mid, end = times
            duration = exposure_ms / 1000
            require(0 <= duration < 1 / fps, "Exposure window must be shorter than frame period")
            require(abs(scheduled - index / fps) <= 1e-10, "Capture schedule accumulates drift")
            require(-1e-10 <= error < dt and abs(error - (state_time - scheduled)) <= 1e-10,
                    "Sampling error exceeds physics interval or disagrees with state")
            require(abs(start - state_time) <= 1e-10 and abs(end - start - duration) <= 1e-10
                    and abs(mid - (start + end) / 2) <= 1e-10, "Invalid exposure window")
            samples = np.asarray(row["sample_timestamps_s"], dtype=float)
            require(samples.ndim == 1 and len(samples) > 0 and np.isfinite(samples).all(),
                    "Invalid exposure samples")
            expected = start + (np.arange(len(samples)) + 0.5) * duration / len(samples)
            require(np.allclose(samples, expected, rtol=0, atol=1e-10), "Subframe timestamps disagree with exposure")
            require(row["calibration_sha256"] == digest, "Frame calibration identity mismatch")
            state_file = row["state_file"]
            if state_file not in archives:
                archives[state_file] = np.load(local_path(state_file), allow_pickle=False)
            archive = archives[state_file]
            state_index = row["state_index"]
            require(0 <= state_index < len(archive["time"]), "State index outside archive")
            require(abs(float(archive["time"][state_index]) - state_time) <= 1e-10,
                    "Archived state timestamp mismatch")
            for key in ("qpos", "qvel"):
                require(len(archive[key]) == len(archive["time"]) and np.isfinite(archive[key][state_index]).all(),
                        "Invalid archived state")
            images = {}
            for variant in ("instant_start", "instant_mid", "exposed"):
                images[variant] = {}
                for side in ("left", "right"):
                    with Image.open(local_path(row["images"][variant][side])) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    require(rgb.shape == (height, width, 3), "PNG resolution disagrees with calibration")
                    images[variant][side] = rgb
            motion = row["motion"]
            require(motion in ("static", "ball", "camera"), "Unknown motion case")
            velocity = float(row["relative_velocity_m_s"])
            depth = float(row["optical_depth_m"])
            require(np.isfinite([velocity, depth]).all() and depth > 0, "Invalid motion geometry")
            require((motion == "static" and velocity == 0) or (motion == "ball" and velocity > 0)
                    or (motion == "camera" and velocity < 0), "Motion provenance disagrees with velocity")
            for side in ("left", "right"):
                instant, centre, exposed = (images[key][side] for key in ("instant_start", "instant_mid", "exposed"))
                if motion == "static" or duration == 0:
                    require(np.array_equal(instant, centre) and np.array_equal(instant, exposed),
                            "Static/zero-exposure image changed")
                    continue
                c0, lo0, hi0 = red_profile(instant)
                cm, _, _ = red_profile(centre)
                ce, lo, hi = red_profile(exposed)
                displacement = K[0, 0] * velocity * duration / depth
                extension = hi - hi0 if velocity > 0 else lo0 - lo
                require(abs(ce - cm) <= 1, "Exposure centroid does not match midpoint image")
                require(abs((cm - c0) - displacement / 2) <= 1.5,
                        "Midpoint image motion disagrees with geometry")
                require(abs(extension - abs(displacement)) <= 2,
                        "Delivered blur extent disagrees with exposure geometry")
                if exposure_ms >= 10:
                    require((hi - lo) - (hi0 - lo0) > 6,
                            "Claimed exposure has no meaningful delivered blur")
                metrics.append({"case_id": case, "frame_index": index, "side": side,
                                "centroid_midpoint_error_px": ce - cm,
                                "extension_px": extension, "expected_extension_px": abs(displacement)})
    finally:
        for archive in archives.values():
            archive.close()
    return {"verified": True, "frames": len(rows), "cases": case_indices,
            "calibration_sha256": digest, "exposure_model": metadata["exposure_model"],
            "scope": "Offline constant-qvel encoded-RGB approximation; no SDK exposure claim",
            "image_metrics": metrics}



def verify_replay(root):
    """Reconstruct each archived kinematic subframe without rendering or stepping."""
    import mujoco

    root = Path(root).resolve()
    model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
    data = mujoco.MjData(model)
    camera_ids = [model.camera(name).id for name in ("zedx_left_camera", "zedx_right_camera")]
    maximum_error, count = 0.0, 0
    rows = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    for row in rows:
        state_path = (root / row["state_file"]).resolve()
        require(state_path.is_relative_to(root), "State path escapes recording")
        with np.load(state_path, allow_pickle=False) as states:
            index = row["state_index"]
            samples = np.asarray(row["sample_timestamps_s"])
            require(np.allclose(states["sample_time"][index], samples, rtol=0, atol=1e-10),
                    "Archived subframe timestamps mismatch")
            archived = states["sample_T_world_from_optical"][index]
            require(archived.shape == (len(samples), 2, 4, 4) and np.isfinite(archived).all(),
                    "Invalid archived optical poses")
            for sample_index, sample_time in enumerate(samples):
                data.qpos[:] = states["qpos"][index]
                data.qvel[:] = states["qvel"][index]
                data.time = sample_time
                mujoco.mj_integratePos(model, data.qpos, data.qvel,
                                      sample_time - float(states["time"][index]))
                mujoco.mj_forward(model, data)
                for side, camera_id in enumerate(camera_ids):
                    transform = np.eye(4)
                    transform[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) * [1, -1, -1]
                    transform[:3, 3] = data.cam_xpos[camera_id]
                    error = float(np.max(np.abs(transform - archived[sample_index, side])))
                    maximum_error = max(maximum_error, error)
                    require(error <= 1e-10, "Replayed optical pose disagrees with archive")
                count += 1
    return {"samples": count, "maximum_pose_matrix_error": maximum_error,
            "method": "independent XML/NPZ readback; mj_integratePos plus mj_forward; no render or physics step"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--verify-replay", action="store_true", help="Also reconstruct archived subframe optical poses")
    args = parser.parse_args()
    report = verify_capture(args.capture)
    if args.verify_replay:
        report["replay"] = verify_replay(args.capture)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
