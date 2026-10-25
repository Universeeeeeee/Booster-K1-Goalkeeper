"""Join saved SDK pixels to authored instantaneous G3 state and clock records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .verify_zed_sdk_stream import read_capture, associate_render_observations
except ImportError:
    from verify_zed_sdk_stream import read_capture, associate_render_observations


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_timing(capture, source, expected_frames):
    capture, source = Path(capture), Path(source)
    payload = read_capture(capture, expected_frames)
    frames = read_rows(capture / "frames.jsonl")
    source_frames = read_rows(source / "source_frames.jsonl")
    metadata = json.loads((source / "metadata.json").read_text())
    sdk_metadata = json.loads((capture / "metadata.json").read_text())
    config = metadata["acquisition"]
    require(config["realized_exposure_s"] == 0 and config["finite_exposure_supported"] is False
            and config["exposure_model"] == "instantaneous_render_baseline",
            "Unsupported finite SDK exposure claim")
    require(config["rolling_skew_s"] == 0, "Unexpected rolling shutter claim")
    require(metadata["sdk_timestamp_source"] == "paired_render_sync_rational_seconds",
            "SDK timestamp must use paired render reference")
    calibration = {"cameras": metadata["cameras"], "resolution": metadata["resolution"],
                   "right_shift_m": metadata["right_shift_m"]}
    calibration_version = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    require(config["calibration_version"] == calibration_version, "Sender calibration version mismatch")
    require(config["camera_usd_sha256"] == metadata["camera_usd_sha256"], "Camera asset version mismatch")
    observations = read_rows(source / metadata["render_observations_file"])
    association = associate_render_observations(capture, frames, source_frames,
                                              metadata["frame_id_marker"], observations)
    authored = {row["source_frame"]: row for row in source_frames}
    require(len(authored) == len(source_frames), "Duplicate source author IDs")
    require(all(b > a for a, b in zip(authored, list(authored)[1:])), "Source author IDs must increase")
    sdk_calibration = {key: sdk_metadata[key] for key in
                       ("calibration", "baseline_m", "width", "height", "serial_number")}
    sdk_version = hashlib.sha256(json.dumps(sdk_calibration, sort_keys=True).encode()).hexdigest()
    joined, received = [], set()
    previous_receive = previous_saved = None
    for frame, match in zip(frames, association["per_frame"]):
        source_id = match["image_source_frame"]
        author = authored[source_id]
        state = author[config["state_key"]]
        timestamps = np.array([state[key] for key in ("t_sim_state_s", "t_exposure_start_s",
                              "t_exposure_mid_s", "t_exposure_end_s")], dtype=float)
        require(np.isfinite(timestamps).all() and np.max(timestamps) == np.min(timestamps),
                "Instantaneous SDK images require shared zero exposure window")
        require(state["exposure_model"] == "instantaneous_render_baseline"
                and state["rolling_skew_s"] == 0, "State exposure provenance mismatch")
        require(abs(timestamps[0] - author["marker_authored_sim_time_s"]) <= 1e-10,
                "Acquisition time disagrees with authored visual identity")
        rational = state["motion_target_s"]
        if isinstance(rational, dict):
            target = rational["numerator"] / rational["denominator"]
        else:
            target = float(rational)
        require(abs(target - source_id / metadata["configured_fps"]) <= 1e-10,
                "Motion schedule accumulates drift")
        phase = np.sin(np.pi * target)
        expected_ball = 0.3 * phase if config["scenario"] == "ball_motion" else 0
        expected_camera = 0.015 * phase if config["scenario"] == "camera_motion" else 0
        require(abs(state["ball_optical_x_m"] - expected_ball) <= 1e-10
                and abs(state["camera_optical_x_m"] - expected_camera) <= 1e-10,
                "Archived motion cannot replay the declared trajectory")
        ball = np.asarray(state["ball_world_position_m"], dtype=float)
        require(ball.shape == (3,) and np.isfinite(ball).all() and state["ball_radius_m"] > 0,
                "Invalid replayable ball state")
        reference_pose = np.asarray(metadata["cameras"]["left"]["world_from_usd_camera_row_matrix"], dtype=float)
        expected_ball_world = np.array([expected_ball, -0.15, -1.2, 1]) @ reference_pose
        require(np.allclose(ball, expected_ball_world[:3], rtol=0, atol=1e-8),
                "Ball world state disagrees with analytic replay")
        displacement = np.array([expected_camera, 0, 0]) @ reference_pose[:3, :3]
        for side in ("left", "right"):
            pose = np.asarray(state["world_from_usd_camera_row_matrix"][side], dtype=float)
            require(pose.shape == (4, 4) and np.isfinite(pose).all()
                    and np.allclose(pose[:, 3], [0, 0, 0, 1], atol=1e-8)
                    and np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-7),
                    "Invalid replayable camera pose")
            expected_pose = np.asarray(metadata["cameras"][side]["world_from_usd_camera_row_matrix"], dtype=float).copy()
            expected_pose[3, :3] += displacement
            require(np.allclose(pose, expected_pose, rtol=0, atol=1e-8),
                    "Camera world state disagrees with analytic replay")
        received_ns, saved_ns = frame["host_monotonic_ns"], frame["host_saved_monotonic_ns"]
        require(received_ns <= saved_ns, "Processing completion precedes arrival")
        require(previous_receive is None or received_ns > previous_receive,
                "Arrival monotonic clock regressed")
        require(previous_saved is None or saved_ns > previous_saved,
                "Processing completion clock regressed")
        require(author["host_before_step_ns"] <= received_ns,
                "SDK arrival precedes the associated authored state")
        previous_receive, previous_saved = received_ns, saved_ns
        received.add(source_id)
        joined.append({**match, "acquisition": state, "source_calibration_version": calibration_version,
                       "sdk_calibration_version": sdk_version, "host_receive_monotonic_ns": received_ns,
                       "host_saved_monotonic_ns": saved_ns,
                       "retrieve_and_save_duration_ns": saved_ns - received_ns,
                       "author_to_receive_same_host_ns": received_ns - author["host_before_step_ns"]})
    return {"verified": True, "scope": "SDK instantaneous pixel-ID/authored-state association; finite exposure unsupported",
            "delivered_scene_geometry_verified": False,
            "scenario": config["scenario"], "sdk_payload": payload, "association": association,
            "source_calibration_version": calibration_version, "sdk_calibration_version": sdk_version,
            "source_author_ids_not_received": sorted(set(authored) - received),
            "source_author_frames": len(authored), "received_frames": len(joined),
            "paired_invalid_observations": sum(not row.get("observation_valid", False) for row in observations),
            "timing_semantics": "World exposure time selected by delivered pixel identity; SDK clock remains paired render reference; host arrival and save are same-host monotonic bounds",
            "joined_frames": joined}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--joined", type=Path, required=True)
    args = parser.parse_args()
    report = verify_timing(args.capture, args.source, args.frames)
    args.joined.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in report.pop("joined_frames")))
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("verified", "scenario", "received_frames", "paired_invalid_observations")}))


if __name__ == "__main__":
    main()
