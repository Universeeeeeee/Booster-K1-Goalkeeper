"""Synthetic contract fixtures; these tests are not evidence of SDK execution."""

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from scripts.verify_g3_sdk_timing import verify_timing


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def recording(tmp_path):
    source, capture = tmp_path / "source", tmp_path / "capture"
    source.mkdir()
    capture.mkdir()
    cameras = {side: {"fx_px": 100, "world_from_usd_camera_row_matrix": np.eye(4).tolist()}
               for side in ("left", "right")}
    calibration = {"cameras": cameras, "resolution": [230, 64], "right_shift_m": 0}
    digest = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    metadata = {**calibration, "configured_fps": 30, "camera_usd_sha256": "asset-version",
                "sdk_timestamp_source": "paired_render_sync_rational_seconds",
                "render_observations_file": "render_observations.jsonl",
                "frame_id_marker": {"pixel_centres": {side: [[i * 10 + 5, 6] for i in range(23)]
                                                        for side in ("left", "right")}},
                "acquisition": {"realized_exposure_s": 0, "finite_exposure_supported": False,
                                "exposure_model": "instantaneous_render_baseline", "rolling_skew_s": 0,
                                "calibration_version": digest, "camera_usd_sha256": "asset-version",
                                "state_key": "acquisition", "scenario": "static"}}
    (source / "metadata.json").write_text(json.dumps(metadata))
    sdk_metadata = {"depth_source": "zed_stereo_sdk", "depth_units": "m", "calibration": cameras,
                    "baseline_m": 0.12, "width": 230, "height": 64, "serial_number": 123}
    (capture / "metadata.json").write_text(json.dumps(sdk_metadata))
    authors, observations, frames = [], [], []
    for i in range(3):
        state = {key: i / 30 for key in ("t_sim_state_s", "t_exposure_start_s",
                                       "t_exposure_mid_s", "t_exposure_end_s")}
        state.update(exposure_model="instantaneous_render_baseline", rolling_skew_s=0,
                     motion_target_s={"numerator": i, "denominator": 30},
                     ball_optical_x_m=0, camera_optical_x_m=0,
                     ball_world_position_m=[0, -0.15, -1.2], ball_radius_m=0.09,
                     world_from_usd_camera_row_matrix={side: np.eye(4).tolist() for side in ("left", "right")})
        authors.append({"source_frame": i, "visual_frame_id": i, "sim_time_s": (i + 1) / 30,
                        "marker_authored_sim_time_s": i / 30, "host_before_step_ns": 1000 + i * 100,
                        "acquisition": state})
        observations.append({"observation_index": i, "reference_time": [100 + i, 30],
                             "observation_valid": True, "observed_left_visual_id": i,
                             "observed_right_visual_id": i})
        if i == 1:
            continue
        payload = [(i >> bit) & 1 for bit in range(15, -1, -1)]
        bits = [1, 0, 1, 0] + payload + [sum(payload) % 2] + [0, 1]
        rgb = np.random.default_rng(i).integers(0, 256, (64, 230, 3), dtype=np.uint8)
        for cell, bit in enumerate(bits):
            rgb[:12, cell * 10:(cell + 1) * 10] = 225 if bit else 35
        stamp = int((100 + i) / 30 * 1e9)
        frame = {"frame_index": len(frames), "sdk_timestamp_ns": stamp,
                 "camera_timestamp_after_retrieval_ns": stamp, "width": 230, "height": 64,
                 "valid_depth_count": 230 * 64, "host_monotonic_ns": 2000 + i * 100,
                 "host_saved_monotonic_ns": 2050 + i * 100}
        for side in ("left", "right"):
            frame[f"{side}_path"] = f"{side}_{i}.png"
            Image.fromarray(rgb).save(capture / frame[f"{side}_path"])
        for kind, value in (("depth", 2), ("confidence", 50)):
            frame[f"{kind}_path"] = f"{kind}_{i}.f32"
            np.full((64, 230), value, dtype="<f4").tofile(capture / frame[f"{kind}_path"])
        frames.append(frame)
    observations.append({"observation_valid": False})
    write_rows(source / "source_frames.jsonl", authors)
    write_rows(source / "render_observations.jsonl", observations)
    write_rows(capture / "frames.jsonl", frames)
    return capture, source


def test_links_delivered_pixels_to_replayable_state_and_clocks(recording):
    report = verify_timing(*recording, 2)
    assert report["verified"]
    assert report["source_author_ids_not_received"] == [1]
    assert report["paired_invalid_observations"] == 1
    assert [row["image_source_frame"] for row in report["joined_frames"]] == [0, 2]
    assert report["joined_frames"][1]["acquisition"]["t_exposure_mid_s"] == 2 / 30


@pytest.mark.parametrize("field,value,match", [
    ("realized_exposure_s", 0.005, "Unsupported finite"),
    ("finite_exposure_supported", True, "Unsupported finite"),
    ("calibration_version", "corrupt", "calibration version"),
])
def test_reject_false_metadata_claim(recording, field, value, match):
    capture, source = recording
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["acquisition"][field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        verify_timing(capture, source, 2)


@pytest.mark.parametrize("field,value,match", [
    ("t_exposure_end_s", 0.005, "zero exposure window"),
    ("ball_optical_x_m", 0.1, "replay"),
    ("motion_target_s", 0.1, "drift"),
    ("world_from_usd_camera_row_matrix", {"left": [[0]], "right": [[0]]}, "camera pose"),
])
def test_reject_corrupt_replay_state(recording, field, value, match):
    capture, source = recording
    path = source / "source_frames.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["acquisition"][field] = value
    write_rows(path, rows)
    with pytest.raises(ValueError, match=match):
        verify_timing(capture, source, 2)


def test_reject_processing_before_arrival(recording):
    capture, source = recording
    path = capture / "frames.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["host_saved_monotonic_ns"] = 0
    write_rows(path, rows)
    with pytest.raises(ValueError, match="completion precedes"):
        verify_timing(capture, source, 2)


def test_reject_paired_image_mismatch(recording):
    capture, source = recording
    (capture / "right_0.png").write_bytes((capture / "right_2.png").read_bytes())
    with pytest.raises(ValueError, match="pixels disagree"):
        verify_timing(capture, source, 2)
