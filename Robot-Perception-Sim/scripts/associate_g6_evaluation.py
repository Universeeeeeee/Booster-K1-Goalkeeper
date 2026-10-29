"""Join G6 independent truth only after full rendered/SDK acquisition verification."""

import argparse
import json
from pathlib import Path

from verify_g6_capture import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.source, args.capture, args.trajectory)
    states = {row["visual_frame_id"]: row["acquisition"] for row in map(
        json.loads, (args.source / "source_frames.jsonl").read_text().splitlines())}
    intended = {row["source_frame"]: row for row in map(
        json.loads, (args.source / "source_observations.jsonl").read_text().splitlines())}
    keys = ("source_frame", "sequence_id", "trajectory_name", "head_pose", "phase", "target_index",
            "exposure_ms", "trial_exposure_ms", "source_time_s", "trajectory_time_s",
            "desired_physical_midpoint_s", "trajectory_requested_time_s", "trajectory_realized_time_s",
            "time_start_s", "time_end_s", "subframe_times_s", "subframe_realized_times_s",
            "subframe_quantization_errors_s", "maximum_subframe_time_error_s",
            "observation_interval_end_s", "observation_interval_basis", "ball_world_position_m",
            "ball_center_camera_m", "ball_radius_m", "T_world_camera", "T_world_left_optical",
            "T_world_right_optical", "T_left_from_right", "K_per_eye", "expected_visible",
            "visibility_reason", "visibility_basis", "visibility_evidence", "any_visible", "all_visible",
            "any_partial_occlusion", "source_trajectory_sha256", "ball_appearance")
    rows = []
    for joined in report["joined"]:
        state = states[joined["visual_id"]]
        row = {key: state[key] for key in keys}
        if state["phase"] == "target":
            row.update(intended[state["source_frame"]])
        row.update(frame_index=joined["frame_index"], visual_id=joined["visual_id"], segment=state["phase"],
                   timing_valid=True, identity_verified_from_both_sdk_images=True,
                   timing_validation_basis="rendered output/subframe IDs, immutable publication, SDK timestamp and frozen-nearest-state verification")
        rows.append(row)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({"delivered_frames": len(rows), "target_frames": sum(r["phase"] == "target" for r in rows)}))


if __name__ == "__main__":
    main()
