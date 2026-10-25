"""Synthetic controlled-publication fixtures, not SDK runtime evidence."""

import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from scripts.verify_g3_sdk_composite import verify_composites


def write_rows(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def marker(rgb, value, y0):
    payload = [(value >> bit) & 1 for bit in range(15, -1, -1)]
    for cell, bit in enumerate([1, 0, 1, 0] + payload + [sum(payload) % 2] + [0, 1]):
        rgb[y0:y0 + 12, cell * 10:(cell + 1) * 10, :3] = 225 if bit else 35


@pytest.fixture
def controlled(tmp_path):
    source, capture = tmp_path / "source", tmp_path / "capture"
    source.mkdir()
    capture.mkdir()
    intrinsics = {"fx": 100, "fy": 100, "cx": 115, "cy": 32}
    cameras = {side: {"fx_px": 100, "world_from_usd_camera_row_matrix": np.eye(4).tolist()} for side in ("left", "right")}
    cal = {"cameras": cameras, "resolution": [230, 64], "right_shift_m": 0}
    digest = hashlib.sha256(json.dumps(cal, sort_keys=True).encode()).hexdigest()
    centres = {side: [[i * 10 + 5, 6] for i in range(23)] for side in cameras}
    secondary = {side: [[i * 10 + 5, 22] for i in range(23)] for side in cameras}
    (source / "producer_snapshot.py").write_text("# synthetic fixture, no SDK runtime\n")
    source_files = {"synthetic": {"path": "producer_snapshot.py", "sha256": hashlib.sha256((source / "producer_snapshot.py").read_bytes()).hexdigest()}}
    config = {"source_files": source_files, "realized_exposure_s": 0, "finite_exposure_supported": True,
              "exposure_model": "instantaneous_render_baseline", "rolling_skew_s": 0,
              "calibration_version": digest, "samples_per_exposure": 1,
              "scenario": "static", "motion_frequency_hz": 0.5, "ball_motion_amplitude_m": 0.3,
              "camera_motion_amplitude_m": 0.015, "subframe_marker_pixel_centres": secondary}
    (source / "metadata.json").write_text(json.dumps({**cal, "configured_fps": 30,
        "acquisition": config, "render_observations_file": "render_observations.jsonl",
        "sdk_timestamp_source": "immutable_composite_exposure_midpoint", "transport": "ipc",
        "frame_id_marker": {"pixel_centres": centres}}))
    (capture / "metadata.json").write_text(json.dumps({"depth_source": "zed_stereo_sdk", "depth_units": "m", "sdk_version": "5.0.7", "baseline_m": 0.12,
        "calibration": {side: intrinsics for side in cameras}}))
    authors, observations, frames = [], [], []
    for i in range(2):
        t = 1 + i / 30
        exact = {"numerator": 30 + i, "denominator": 30}
        rgb = np.random.default_rng(i).integers(0, 255, (64, 230, 4), dtype=np.uint8)
        rgb[..., 3] = 255
        marker(rgb, i, 0)
        marker(rgb, i, 16)
        yy, xx = np.indices(rgb.shape[:2])
        rgb[(xx - 115) ** 2 + (yy - 44.5) ** 2 <= 7.5 ** 2, :3] = [225, 142, 66]
        Image.fromarray(rgb).save(source / f"sub_{i}.png")
        Image.fromarray(rgb).save(source / f"mean_{i}.png")
        Image.fromarray(rgb[..., :3]).save(capture / f"sdk_{i}.png")
        pose = {"world_from_usd_camera_row_matrix": {side: np.eye(4).tolist() for side in cameras},
                "ball_world_position_m": [0, -0.15, -1.2], "ball_radius_m": 0.09, "motion_target_s": exact,
                "ball_optical_x_m": 0, "camera_optical_x_m": 0, "native_world_time_s": 1}
        state = {"window_rational": {key: exact for key in ("start_s", "mid_s", "end_s")},
                 "t_exposure_start_s": t, "t_exposure_mid_s": t, "t_exposure_end_s": t,
                 "sdk_composite_timestamp_ns": int(t * 1e9),
                 "publication": {"timestamp_s": t, "bytes_per_eye": rgb.nbytes,
                    "host_armed_monotonic_ns": 900 + i * 100, "host_after_render_monotonic_ns": 950 + i * 100,
                    "source_sha256_left_right": [hashlib.sha256(rgb.tobytes()).hexdigest()] * 2},
                 "subframes": [{"subframe_id": i, "target_s": exact, "state": pose,
                    "reference_time": [100 + i, 30], "host_authored_monotonic_ns": 800 + i * 100,
                    "host_observed_monotonic_ns": 850 + i * 100, "left_path": f"sub_{i}.png", "right_path": f"sub_{i}.png"}],
                 "composite_paths": {side: f"mean_{i}.png" for side in cameras}}
        authors.append({"source_frame": i, "visual_frame_id": i, "acquisition": state})
        observations.append({"observation_valid": True, "reference_time": [100 + i, 30],
            "observed_left_subframe_id": i, "observed_right_subframe_id": i,
            "observed_left_visual_id": i, "observed_right_visual_id": i})
        frame = {"frame_index": i, "sdk_timestamp_ns": int(t * 1e9),
                 "camera_timestamp_after_retrieval_ns": int(t * 1e9), "width": 230, "height": 64,
                 "valid_depth_count": 230 * 64, "host_monotonic_ns": 1000 + i * 100,
                 "host_saved_monotonic_ns": 1050 + i * 100,
                 "left_path": f"sdk_{i}.png", "right_path": f"sdk_{i}.png"}
        for kind, value in (("depth", 2), ("confidence", 50)):
            frame[f"{kind}_path"] = f"{kind}_{i}.f32"
            np.full((64, 230), value, dtype="<f4").tofile(capture / frame[f"{kind}_path"])
        frames.append(frame)
    write_rows(source / "source_frames.jsonl", authors)
    write_rows(source / "render_observations.jsonl", observations)
    write_rows(capture / "frames.jsonl", frames)
    return capture, source


def test_controlled_static_baseline_verifies_actual_mean_and_publication(controlled):
    report = verify_composites(*controlled, 2)
    assert report["verified"]
    assert report["maximum_composite_roi_mae"] == 0
    assert report["source_subframes_verified"] == 2


def test_reject_old_mutable_buffer_wrong_id_at_valid_timestamp(controlled):
    capture, source = controlled
    (capture / "sdk_0.png").write_bytes((capture / "sdk_1.png").read_bytes())
    with pytest.raises(ValueError, match="publication ID"):
        verify_composites(capture, source, 2)


def test_reject_composite_not_matching_saved_samples(controlled):
    capture, source = controlled
    array = np.array(Image.open(source / "mean_0.png"))
    array[50, 100, 0] ^= 1
    Image.fromarray(array).save(source / "mean_0.png")
    with pytest.raises(ValueError, match="subframe mean"):
        verify_composites(capture, source, 2)


def test_reject_unobserved_subframe_packet(controlled):
    capture, source = controlled
    path = source / "render_observations.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["observed_right_subframe_id"] = 999
    write_rows(path, values)
    with pytest.raises(ValueError, match="same-packet"):
        verify_composites(capture, source, 2)


def test_reject_false_upload_hash(controlled):
    capture, source = controlled
    path = source / "source_frames.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["acquisition"]["publication"]["source_sha256_left_right"][0] = "bad"
    write_rows(path, values)
    with pytest.raises(ValueError, match="CUDA source digest"):
        verify_composites(capture, source, 2)


def finite_static_fixture(capture, source):
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["acquisition"].update(realized_exposure_s=0.01, exposure_model="encoded_rgba_box_average")
    path.write_text(json.dumps(metadata))
    path = source / "source_frames.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    frames = [json.loads(line) for line in (capture / "frames.jsonl").read_text().splitlines()]
    from fractions import Fraction
    for i, (value, frame) in enumerate(zip(values, frames)):
        start = Fraction(1) + Fraction(i, 30)
        window = {"start_s": start, "mid_s": start + Fraction(1, 200), "end_s": start + Fraction(1, 100)}
        state = value["acquisition"]
        state["window_rational"] = {key: {"numerator": t.numerator, "denominator": t.denominator} for key, t in window.items()}
        for key, t in window.items():
            state[f"t_exposure_{key}"] = float(t)
        target = state["window_rational"]["mid_s"]
        state["subframes"][0]["target_s"] = target
        state["subframes"][0]["state"]["motion_target_s"] = target
        stamp = int(float(window["mid_s"]) * 1e9)
        state["publication"]["timestamp_s"] = float(window["mid_s"])
        state["sdk_composite_timestamp_ns"] = stamp
        frame.update(sdk_timestamp_ns=stamp, camera_timestamp_after_retrieval_ns=stamp)
    write_rows(path, values)
    write_rows(capture / "frames.jsonl", frames)


def test_finite_static_requires_frozen_zero_codec_baseline(controlled):
    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    finite_static_fixture(capture, source)
    with pytest.raises(ValueError, match="frozen static"):
        verify_composites(capture, source, 2)
    report = verify_composites(capture, source, 2, baseline)
    assert report["realized_exposure_s"] == 0.01
    assert report["static_codec_mae_threshold"] == 3


def test_nonzero_window_cannot_be_relabelled_without_subframe_times(controlled):
    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    finite_static_fixture(capture, source)
    path = source / "source_frames.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    values[0]["acquisition"]["subframes"][0]["target_s"] = {"numerator": 1, "denominator": 1}
    write_rows(path, values)
    with pytest.raises(ValueError, match="quadrature"):
        verify_composites(capture, source, 2, baseline)


def test_static_baseline_cannot_normalize_bad_delivered_pixels(controlled):
    capture, source = controlled
    for i in range(2):
        path = capture / f"sdk_{i}.png"
        image = np.array(Image.open(path))
        yy, xx = np.indices(image.shape[:2])
        sphere = (xx - 115) ** 2 + (yy - 44.5) ** 2 <= 7.5 ** 2
        image[(yy >= 30) & ~sphere] = 0
        Image.fromarray(image).save(path)
    with pytest.raises(ValueError, match="fixed 8-level"):
        verify_composites(capture, source, 2)


def test_finite_motion_claim_cannot_pass_instantaneous_images_without_blur_signal(controlled):
    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    finite_static_fixture(capture, source)
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["acquisition"].update(scenario="ball_motion", ball_motion_amplitude_m=0)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Insufficient delivered motion"):
        verify_composites(capture, source, 2, baseline)


@pytest.mark.parametrize("method", ["synchronous_omnigraph_node_request", "synchronous_impulse_execution_graph"])
def test_explicit_synchronous_publication_uses_execution_completion_bound(controlled, method):
    capture, source = controlled
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["acquisition"]["publication_execution"] = method
    path.write_text(json.dumps(metadata))
    path = source / "source_frames.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines()]
    for value in values:
        publication = value["acquisition"]["publication"]
        publication["host_after_execution_monotonic_ns"] = publication.pop("host_after_render_monotonic_ns")
    write_rows(path, values)
    assert verify_composites(capture, source, 2)["verified"]


def test_reject_changed_producer_snapshot(controlled):
    capture, source = controlled
    (source / "producer_snapshot.py").write_text("# changed after capture\n")
    with pytest.raises(ValueError, match="source snapshot digest"):
        verify_composites(capture, source, 2)


def test_offline_replay_must_match_prepared_exposure_plan(controlled):
    capture, source = controlled
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["acquisition"].update(publication_mode="offline_render_then_replay",
                                   prepared_plan_file="prepared.jsonl", prepared_window_count=2)
    path.write_text(json.dumps(metadata))
    path = source / "source_frames.jsonl"
    published = [json.loads(line) for line in path.read_text().splitlines()]
    prepared = []
    for i, value in enumerate(published):
        value["acquisition"]["host_prepared_monotonic_ns"] = 875 + i * 100
        prepared.append({"source_frame": i, "acquisition": {key: item for key, item in value["acquisition"].items() if key != "publication"}})
    write_rows(path, published)
    write_rows(source / "prepared.jsonl", prepared)
    assert verify_composites(capture, source, 2)["prepared_window_count"] == 2
    prepared[0]["acquisition"]["sdk_composite_timestamp_ns"] += 1
    write_rows(source / "prepared.jsonl", prepared)
    with pytest.raises(ValueError, match="differs from prepared"):
        verify_composites(capture, source, 2)


def test_codec_baseline_cannot_cross_transport(controlled):
    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["transport"] = "network"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="transport differs"):
        verify_composites(capture, source, 2, baseline)


def start_reference_fixture(capture, source):
    finite_static_fixture(capture, source)
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["sdk_timestamp_source"] = "immutable_composite_exposure_start"
    metadata["acquisition"].update(sdk_timestamp_reference="exposure_start",
                                   sdk_to_exposure_midpoint_offset_s=0.005)
    path.write_text(json.dumps(metadata))
    path = source / "source_frames.jsonl"
    published = [json.loads(line) for line in path.read_text().splitlines()]
    frames = [json.loads(line) for line in (capture / "frames.jsonl").read_text().splitlines()]
    for author, frame in zip(published, frames):
        state = author["acquisition"]
        stamp = int(state["t_exposure_start_s"] * 1e9)
        state["sdk_composite_timestamp_ns"] = stamp
        state["publication"]["timestamp_s"] = state["t_exposure_start_s"]
        frame.update(sdk_timestamp_ns=stamp, camera_timestamp_after_retrieval_ns=stamp)
    write_rows(path, published)
    write_rows(capture / "frames.jsonl", frames)


def test_declared_start_reference_keeps_optical_midpoint_and_known_offset(controlled):
    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    start_reference_fixture(capture, source)
    report = verify_composites(capture, source, 2, baseline)
    first = report["joined_frames"][0]
    assert first["sdk_timestamp_ns"] == 10 ** 9
    assert first["acquisition"]["t_exposure_mid_s"] == 1.005
    assert first["sdk_to_exposure_midpoint_offset_s"] == 0.005
    assert first["sdk_minus_publication_reference_ns"] == 0
    assert first["earliest_available_authored_s"] == 1.01


def test_false_declared_clock_offset_is_rejected(controlled):
    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    start_reference_fixture(capture, source)
    path = source / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["acquisition"]["sdk_to_exposure_midpoint_offset_s"] = 0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="offset disagrees"):
        verify_composites(capture, source, 2, baseline)


def test_observed_timestamp_offset_is_not_fitted_to_declared_start(controlled):
    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    start_reference_fixture(capture, source)
    path = capture / "frames.jsonl"
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    for frame in frames:
        frame["sdk_timestamp_ns"] += 5_000_000
        frame["camera_timestamp_after_retrieval_ns"] += 5_000_000
    write_rows(path, frames)
    with pytest.raises(ValueError, match="no unique explicit publication"):
        verify_composites(capture, source, 2, baseline)


def test_codec_baseline_cannot_cross_sdk_version(controlled):
    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    path = capture / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["sdk_version"] = "5.1.0"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SDK version differs"):
        verify_composites(capture, source, 2, baseline)


def test_finite_static_must_deliver_mixed_secondary_pixels_not_instant_buffer(controlled):
    from fractions import Fraction

    capture, source = controlled
    baseline = verify_composites(capture, source, 2)
    finite_static_fixture(capture, source)
    meta_path = source / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    metadata["acquisition"]["samples_per_exposure"] = 3
    meta_path.write_text(json.dumps(metadata))
    source_path = source / "source_frames.jsonl"
    authors = [json.loads(line) for line in source_path.read_text().splitlines()]
    observations = []
    for i, author in enumerate(authors):
        state = author["acquisition"]
        old = state["subframes"][0]
        base = np.array(Image.open(source / old["left_path"]))
        samples, pixels = [], []
        for j in range(3):
            sid = 3 * i + j
            sample = json.loads(json.dumps(old))
            t = Fraction(1) + Fraction(i, 30) + Fraction(2 * j + 1, 600)
            target = {"numerator": t.numerator, "denominator": t.denominator}
            image = base.copy()
            marker(image, sid, 16)
            name = f"quadrature_{sid}.png"
            Image.fromarray(image).save(source / name)
            sample.update(subframe_id=sid, target_s=target, left_path=name, right_path=name,
                          reference_time=[100 + sid, 30])
            sample["state"]["motion_target_s"] = target
            samples.append(sample)
            pixels.append(image)
            observations.append({"observation_valid": True, "reference_time": [100 + sid, 30],
                "observed_left_subframe_id": sid, "observed_right_subframe_id": sid,
                "observed_left_visual_id": i, "observed_right_visual_id": i})
        mean = np.rint(np.mean(pixels, axis=0)).astype(np.uint8)
        state["subframes"] = samples
        state["publication"]["source_sha256_left_right"] = [hashlib.sha256(mean.tobytes()).hexdigest()] * 2
        Image.fromarray(mean).save(source / f"mean_{i}.png")
        Image.fromarray(mean[..., :3]).save(capture / f"sdk_{i}.png")
    write_rows(source_path, authors)
    write_rows(source / "render_observations.jsonl", observations)
    assert verify_composites(capture, source, 2, baseline)["verified"]
    # Primary output ID and ball geometry remain correct; only integration is false.
    instant = np.array(Image.open(source / authors[0]["acquisition"]["subframes"][1]["left_path"]))
    Image.fromarray(instant[..., :3]).save(capture / "sdk_0.png")
    with pytest.raises(ValueError, match="secondary-marker pixels"):
        verify_composites(capture, source, 2, baseline)
