"""Recorded-file validation tests; these synthetic fixtures are not SDK evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


spec = importlib.util.spec_from_file_location(
    "verify_zed_sdk_stream", Path(__file__).parents[1] / "scripts/verify_zed_sdk_stream.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def capture(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(
        {"depth_source": "zed_stereo_sdk", "depth_units": "m"}))
    random = np.random.default_rng(9)
    left = random.integers(0, 256, (64, 192, 3), dtype=np.uint8)
    right = np.roll(left, -4, axis=1)
    frames = []
    for index in range(2):
        frame = {"frame_index": index, "sdk_timestamp_ns": (index + 1) * 33333333,
                 "camera_timestamp_after_retrieval_ns": (index + 1) * 33333333,
                 "width": 192, "height": 64, "valid_depth_count": 64 * 192}
        for side, array in (("left", left), ("right", right)):
            frame[f"{side}_path"] = f"{side}_{index}.png"
            Image.fromarray(array).save(tmp_path / frame[f"{side}_path"])
        for kind, value in (("depth", 2), ("confidence", 50)):
            frame[f"{kind}_path"] = f"{kind}_{index}.f32"
            np.full((64, 192), value, dtype="<f4").tofile(tmp_path / frame[f"{kind}_path"])
        frames.append(frame)
    (tmp_path / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in frames))
    return tmp_path


def test_reads_actual_payload_and_independent_disparity(capture):
    result = verifier.read_capture(capture, 2)
    assert result["plane_median_m"] == 2
    assert result["minimum_plane_valid_fraction"] == 1
    assert all(row["integer_disparity_px"] == 4 for row in result["disparity_checks"])
    assert result["provenance_verified_by_this_script"] is False


def test_duplicate_sdk_timestamp_rejected(capture):
    path = capture / "frames.jsonl"
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    frames[1]["sdk_timestamp_ns"] = frames[0]["sdk_timestamp_ns"]
    path.write_text("".join(json.dumps(row) + "\n" for row in frames))
    with pytest.raises(ValueError, match="strictly increase"):
        verifier.read_capture(capture, 2)


def test_truncated_depth_rejected(capture):
    (capture / "depth_0.f32").write_bytes(b"\0" * 12)
    with pytest.raises(ValueError, match="payload size"):
        verifier.read_capture(capture, 2)


def test_renderer_depth_not_accepted_as_sdk_capture(capture):
    (capture / "metadata.json").write_text(json.dumps(
        {"depth_source": "renderer_gt_via_sdk", "depth_units": "m"}))
    with pytest.raises(ValueError, match="SDK stereo"):
        verifier.read_capture(capture, 2)


def test_simulation_clock_matching_counts_skipped_source_frames():
    source = [{"source_frame": i, "sim_time_s": i / 30,
               "host_monotonic_ns": 10**12 + i * 33333333} for i in range(10)]
    frames = [{"sdk_timestamp_ns": int(source[i]["sim_time_s"] * 1e9),
               "host_monotonic_ns": source[i]["host_monotonic_ns"] + 5000000}
              for i in (1, 4, 7)]
    result = verifier.associate_source_frames(frames, source)
    assert result["matched_source_frame_ids"] == [1, 4, 7]
    assert result["unobserved_source_frames"] == 4
    assert result["grab_minus_post_step_source_record_ms"]["median"] == 5
    assert result["timestamp_mapping_verified"] is False
    assert result["pixel_identity_verified"] is False


@pytest.mark.parametrize("error_ns", [1000, 10**18])
def test_wrong_clock_domain_or_offset_is_not_fitted_away(error_ns):
    source = [{"source_frame": i, "sim_time_s": i / 30,
               "host_monotonic_ns": 10**12 + i * 33333333} for i in range(10)]
    frames = [{"sdk_timestamp_ns": int(source[i]["sim_time_s"] * 1e9) + error_ns,
               "host_monotonic_ns": source[i]["host_monotonic_ns"] + 5000000}
              for i in (1, 4, 7)]
    with pytest.raises(ValueError, match="timestamp"):
        verifier.associate_source_frames(frames, source)


def test_bridge_clock_is_distinct_from_post_step_world_clock():
    source = [{"source_frame": i, "sim_time_s": (i + 1) / 30,
               "zed_input_sim_time_s": i / 30, "og_sim_time_s": i / 30,
               "host_monotonic_ns": 10**12 + i * 33333333} for i in range(10)]
    frames = [{"sdk_timestamp_ns": int(source[i]["zed_input_sim_time_s"] * 1e9),
               "host_monotonic_ns": source[i]["host_monotonic_ns"] - 1000000}
              for i in (1, 4, 7)]
    result = verifier.associate_source_frames(frames, source)
    assert result["timestamp_mapping_verified"] is False  # Only pixels can establish this.
    assert result["matched_source_frame_ids"] == [0, 3, 6]
    assert result["graph_observation_source_frame_candidates"] == [[1], [4], [7]]
    assert result["post_step_world_minus_bridge_s"] == pytest.approx(1 / 30)
    assert result["grab_minus_post_step_source_record_ms"]["median"] == pytest.approx(32.333333)


def marker_image(value):
    payload = [(value >> bit) & 1 for bit in range(15, -1, -1)]
    bits = [1, 0, 1, 0] + payload + [sum(payload) % 2] + [0, 1]
    rgb = np.zeros((12, 230, 3), dtype=np.uint8)
    centres = []
    for i, bit in enumerate(bits):
        rgb[:, i*10:(i+1)*10] = 225 if bit else 35
        centres.append([i*10+5, 6])
    return rgb, centres, bits


@pytest.mark.parametrize("value", [0, 1, 256, 32768, 65535])
def test_decodes_visual_identity_and_rejects_corrupt_parity(value):
    rgb, centres, bits = marker_image(value)
    assert verifier.decode_frame_id(rgb, centres) == value
    rgb[:, 200:210] = 35 if bits[20] else 225
    with pytest.raises(ValueError, match="parity"):
        verifier.decode_frame_id(rgb, centres)


def test_two_valid_but_different_eye_ids_are_rejected(tmp_path):
    centres = None
    for side, value in (("left", 14), ("right", 15)):
        rgb, centres, _ = marker_image(value)
        Image.fromarray(rgb).save(tmp_path / f"{side}.png")
    frames = [{"frame_index": 0, "left_path": "left.png", "right_path": "right.png"}]
    source = [{"source_frame": 14, "visual_frame_id": 14, "sim_time_s": 14/30}]
    marker = {"pixel_centres": {"left": centres, "right": centres}}
    with pytest.raises(ValueError, match="disagree"):
        verifier.verify_visual_ids(tmp_path, frames, source, marker, {"matched_source_frame_ids": [14]})


def test_visual_identity_exposes_render_queue_lag_instead_of_hiding_it(tmp_path):
    rgb, centres, _ = marker_image(14)
    Image.fromarray(rgb).save(tmp_path / "both.png")
    frames = [{"frame_index": 0, "left_path": "both.png", "right_path": "both.png",
               "sdk_timestamp_ns": int(14/30 * 1e9)}]
    source = [{"source_frame": i, "visual_frame_id": i, "sim_time_s": i/30}
              for i in (14, 15, 16)]
    marker = {"pixel_centres": {"left": centres, "right": centres}}
    result = verifier.verify_visual_ids(tmp_path, frames, source, marker,
                                        {"matched_source_frame_ids": [14],
                                         "graph_observation_source_frame_candidates": [[16]]})
    assert result["pixel_identity_verified"] is True
    assert result["sampled_graph_minus_image_source_frames"] == {"min": 2, "median": 2, "max": 2}
    assert result["per_frame"][0]["image_source_frame"] == 14
    assert result["per_frame"][0]["timestamp_world_source_frame"] == 14
    assert result["max_sdk_image_time_residual_ns"] == 0
    frames[0]["sdk_timestamp_ns"] += 33333333
    with pytest.raises(ValueError, match="visually identified"):
        verifier.verify_visual_ids(tmp_path, frames, source, marker,
                                   {"matched_source_frame_ids": [16]})


def test_cached_graph_observations_do_not_claim_unique_dispatch():
    source = [{"source_frame": i, "sim_time_s": i / 30,
               "zed_input_sim_time_s": (i // 2) / 30, "og_sim_time_s": (i // 2) / 30,
               "host_monotonic_ns": 10**12 + i * 33333333} for i in range(8)]
    frames = [{"sdk_timestamp_ns": int(i / 30 * 1e9),
               "host_monotonic_ns": 10**12 + (i + 8) * 33333333} for i in (1, 2, 3)]
    result = verifier.associate_source_frames(frames, source)
    assert result["graph_observation_source_frame_candidates"] == [[2, 3], [4, 5], [6, 7]]
    assert result["repeated_graph_input_observations"] == 4
    assert result["timestamp_mapping_verified"] is False


def test_paired_render_reference_maps_actual_pixels_without_world_offset_fit(tmp_path):
    source, frames, centres = [], [], None
    for i in range(3):
        rgb, centres, _ = marker_image(i)
        Image.fromarray(rgb).save(tmp_path / f"{i}.png")
        source.append({"source_frame": i, "visual_frame_id": i, "sim_time_s": i / 30,
                       "observation_valid": True, "observed_ref_before": [100 + i, 30],
                       "observed_ref_after": [100 + i, 30],
                       "observed_left_visual_id": i, "observed_right_visual_id": i})
        frames.append({"frame_index": i, "sdk_timestamp_ns": int((100 + i) / 30 * 1e9),
                       "host_monotonic_ns": 10**12 + i * 33333333,
                       "left_path": f"{i}.png", "right_path": f"{i}.png"})
    marker = {"pixel_centres": {"left": centres, "right": centres}}
    result = verifier.associate_render_observations(tmp_path, frames, source, marker)
    assert result["matched_frames"] == 3
    assert result["max_reference_residual_ns"] == 0
    assert result["per_frame"][0]["source_world_state_time_s"] == 0
    observations = [{"observation_index": i + 10, "reference_time": row["observed_ref_before"],
                     "observation_valid": True, "observed_left_visual_id": i,
                     "observed_right_visual_id": i} for i, row in enumerate(source)]
    authored_only = [{key: row[key] for key in ("source_frame", "visual_frame_id", "sim_time_s")}
                     for row in source]
    event_result = verifier.associate_render_observations(tmp_path, frames, authored_only, marker, observations)
    assert event_result["per_frame"][0]["source_observation_indices"] == [10]
    assert event_result["matched_frames"] == 3
    frames[0]["right_path"] = "1.png"
    with pytest.raises(ValueError, match="pixels disagree"):
        verifier.associate_render_observations(tmp_path, frames, source, marker)
    frames[0]["right_path"] = "0.png"
    source[1]["observed_ref_before"] = source[1]["observed_ref_after"] = [100, 30]
    with pytest.raises(ValueError, match="conflicting source"):
        verifier.associate_render_observations(tmp_path, frames, source, marker)
