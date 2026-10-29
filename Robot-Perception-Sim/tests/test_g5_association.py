"""Synthetic identity failures; these fixtures are not SDK capture evidence."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("associate_g5", scripts / "associate_g5_evaluation.py")
association = importlib.util.module_from_spec(spec)
spec.loader.exec_module(association)


def save_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def recording(tmp_path):
    sender, capture = tmp_path / "sender", tmp_path / "capture"
    sender.mkdir()
    capture.mkdir()
    centres = [[cell * 10 + 5, 6] for cell in range(23)]
    metadata = {"acquisition": {"scenario": "g5_static"}, "frame_id_marker": {
        "pixel_centres": {side: centres for side in ("left", "right")}}}
    (sender / "metadata.json").write_text(json.dumps(metadata))
    sources, frames = [], []
    for index in range(2):
        payload = [(index >> bit) & 1 for bit in range(15, -1, -1)]
        bits = [1, 0, 1, 0] + payload + [sum(payload) % 2] + [0, 1]
        rgb = np.repeat(np.array([225 if bit else 35 for bit in bits], dtype=np.uint8), 10)
        rgb = np.repeat(rgb[None, :, None], 12, axis=0).repeat(3, axis=2)
        stamp = (index + 1) * 1000000000
        state = {"sdk_composite_timestamp_ns": stamp, "depth_label_m": 1,
                 "angle_label": "center", "ball_center_camera_m": [0, 0, 1],
                 "ball_world_position_m": [1, 0, 0], "ball_radius_m": .11,
                 "T_world_camera": np.eye(4).tolist(), "expected_visible": True}
        sources.append({"source_frame": index, "visual_frame_id": index, "acquisition": state})
        frame = {"frame_index": index, "sdk_timestamp_ns": stamp}
        for side in ("left", "right"):
            frame[f"{side}_path"] = f"{side}_{index}.png"
            Image.fromarray(rgb).save(capture / frame[f"{side}_path"])
        frames.append(frame)
    save_rows(sender / "source_frames.jsonl", sources)
    save_rows(capture / "frames.jsonl", frames)
    return sender, capture, sources, frames


def test_associates_both_delivered_images_and_timestamp(recording):
    sender, capture, _, _ = recording
    rows = association.associate(sender, capture)
    assert [row["source_frame"] for row in rows] == [0, 1]
    assert all(row["identity_verified_from_both_sdk_images"] for row in rows)


@pytest.mark.parametrize("key", ["source_frame", "visual_frame_id"])
def test_duplicate_source_identity_rejected(recording, key):
    sender, capture, sources, _ = recording
    sources[1][key] = sources[0][key]
    save_rows(sender / "source_frames.jsonl", sources)
    with pytest.raises(ValueError, match="unique"):
        association.associate(sender, capture)


@pytest.mark.parametrize("key", ["frame_index", "sdk_timestamp_ns"])
@pytest.mark.parametrize("offset", [0, -1])
def test_duplicate_or_regressing_receiver_sequence_rejected(recording, key, offset):
    sender, capture, _, frames = recording
    frames[1][key] = frames[0][key] + offset
    save_rows(capture / "frames.jsonl", frames)
    with pytest.raises(ValueError, match="strictly increase"):
        association.associate(sender, capture)


def test_repeated_decoded_pixels_rejected_even_with_new_timestamp(recording):
    sender, capture, _, frames = recording
    for side in ("left", "right"):
        frames[1][f"{side}_path"] = frames[0][f"{side}_path"]
    save_rows(capture / "frames.jsonl", frames)
    with pytest.raises(ValueError, match="identity repeated"):
        association.associate(sender, capture)


def test_wrong_scenario_rejected(recording):
    sender, capture, _, _ = recording
    path = sender / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["acquisition"]["scenario"] = "static"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="g5_static"):
        association.associate(sender, capture)


def test_timestamp_disagrees_with_pixel_identity(recording):
    sender, capture, _, frames = recording
    frames[1]["sdk_timestamp_ns"] += 100
    save_rows(capture / "frames.jsonl", frames)
    with pytest.raises(ValueError, match="Pixel identity disagrees"):
        association.associate(sender, capture)
