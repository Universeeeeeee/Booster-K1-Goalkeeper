"""Join independent authored GT to actual SDK pixels, for offline evaluation only."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from verify_zed_sdk_stream import decode_frame_id


def associate(sender, capture):
    metadata = json.loads((sender / "metadata.json").read_text())
    if metadata.get("acquisition", {}).get("scenario") != "g5_static":
        raise ValueError("Evaluation requires the g5_static scenario")
    source, source_indices = {}, set()
    for row in map(json.loads, (sender / "source_frames.jsonl").read_text().splitlines()):
        visual = row["visual_frame_id"]
        if visual in source or row["source_frame"] in source_indices:
            raise ValueError("Source frame and visual identities must be unique")
        source[visual] = row
        source_indices.add(row["source_frame"])
    rows, decoded_sources = [], set()
    previous_index = previous_stamp = -1
    for frame in map(json.loads, (capture / "frames.jsonl").read_text().splitlines()):
        if frame["frame_index"] <= previous_index or frame["sdk_timestamp_ns"] <= previous_stamp:
            raise ValueError("Receiver frame indices and SDK timestamps must strictly increase")
        previous_index, previous_stamp = frame["frame_index"], frame["sdk_timestamp_ns"]
        ids = [decode_frame_id(np.asarray(Image.open(capture / frame[f"{side}_path"]).convert("RGB")),
                               metadata["frame_id_marker"]["pixel_centres"][side])
               for side in ("left", "right")]
        if ids[0] != ids[1]:
            raise ValueError(f"Stereo frame identity differs: {ids}")
        if ids[0] in decoded_sources:
            raise ValueError("Decoded source identity repeated within SDK capture")
        decoded_sources.add(ids[0])
        if ids[0] not in source:
            raise ValueError("Decoded source identity missing from authored states")
        state = source[ids[0]]["acquisition"]
        if abs(frame["sdk_timestamp_ns"] - state["sdk_composite_timestamp_ns"]) > 2:
            raise ValueError("Pixel identity disagrees with SDK timestamp")
        rows.append({"frame_index": frame["frame_index"], "source_frame": source[ids[0]]["source_frame"],
                     "identity_verified_from_both_sdk_images": True,
                     **{key: state[key] for key in ("depth_label_m", "angle_label", "ball_center_camera_m",
                                                    "ball_world_position_m", "ball_radius_m",
                                                    "T_world_camera", "expected_visible")}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = associate(args.sender, args.capture)
    with args.output.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps({"verified_frames": len(rows), "case_counts": {
        str(case): sum((row["depth_label_m"], row["angle_label"]) == case for row in rows)
        for case in sorted({(row["depth_label_m"], row["angle_label"]) for row in rows})}}, indent=2))


if __name__ == "__main__":
    main()
