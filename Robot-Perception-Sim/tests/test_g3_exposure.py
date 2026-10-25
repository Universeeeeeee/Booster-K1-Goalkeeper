import json

import numpy as np
from PIL import Image
import pytest

from scripts.verify_g3_exposure import calibration_digest, verify_capture


def write_capture(root, motion="ball", exposure_ms=10):
    calibration = {"K": [[100, 0, 64], [0, 100, 32], [0, 0, 1]],
                   "baseline_m": 0.12, "resolution": [128, 64]}
    digest = calibration_digest(calibration)
    metadata = {"source": "mujoco_existing_stereo_capture", "calibration": calibration,
                "calibration_sha256": digest, "physics_dt_s": 0.001,
                "exposure_model": "constant_qvel_encoded_rgb_average"}
    (root / "metadata.json").write_text(json.dumps(metadata))
    velocity = {"static": 0, "ball": 40, "camera": -40}[motion]
    exposure = exposure_ms / 1000
    sample_times = (np.arange(20) + 0.5) * exposure / 20

    def render(t):
        rgb = np.zeros((64, 128, 3), dtype=np.uint8)
        x = int(round(60 + 100 * velocity * t / 2))
        rgb[28:37, x - 4:x + 5, 0] = 240
        return rgb

    images = {"instant_start": render(0), "instant_mid": render(exposure / 2),
              "exposed": np.rint(np.mean([render(t).astype(float) for t in sample_times], axis=0)).astype(np.uint8)}
    paths = {}
    for variant, rgb in images.items():
        paths[variant] = {}
        for side in ("left", "right"):
            name = f"{variant}_{side}.png"
            Image.fromarray(rgb).save(root / name)
            paths[variant][side] = name
    row = {"case_id": motion, "motion": motion, "fps": 60, "exposure_ms": exposure_ms,
           "frame_index": 0, "scheduled_time_s": 0, "sim_state_time_s": 0,
           "sampling_error_s": 0, "exposure_start_s": 0, "exposure_mid_s": exposure / 2,
           "exposure_end_s": exposure, "sample_timestamps_s": sample_times.tolist(),
           "calibration_sha256": digest, "images": paths, "relative_velocity_m_s": velocity,
           "optical_depth_m": 2, "state_file": "states.npz", "state_index": 0}
    np.savez(root / "states.npz", time=[0.0], qpos=[[0.0]], qvel=[[velocity]])
    save_row(root, row)
    return row


def save_row(root, row):
    (root / "frames.jsonl").write_text(json.dumps(row) + "\n")


@pytest.mark.parametrize("motion", ["static", "ball", "camera"])
@pytest.mark.parametrize("exposure_ms", [0, 1, 3, 5, 10])
def test_independent_png_readback(tmp_path, motion, exposure_ms):
    write_capture(tmp_path, motion, exposure_ms)
    assert verify_capture(tmp_path)["verified"]


@pytest.mark.parametrize("field,value,match", [
    ("exposure_end_s", 0.009, "window"),
    ("exposure_mid_s", 0, "window"),
    ("calibration_sha256", "false", "calibration"),
    ("scheduled_time_s", 0.001, "drift"),
    ("sampling_error_s", 0.002, "Sampling"),
    ("sample_timestamps_s", [0, 0.01], "Subframe"),
    ("state_index", 1, "State index"),
])
def test_reject_corrupted_contract(tmp_path, field, value, match):
    row = write_capture(tmp_path)
    row[field] = value
    save_row(tmp_path, row)
    with pytest.raises(ValueError, match=match):
        verify_capture(tmp_path)


def test_reject_missing_png(tmp_path):
    row = write_capture(tmp_path)
    (tmp_path / row["images"]["exposed"]["right"]).unlink()
    with pytest.raises(FileNotFoundError):
        verify_capture(tmp_path)


def test_reject_instantaneous_pixels_with_exposure_metadata(tmp_path):
    row = write_capture(tmp_path)
    for side in ("left", "right"):
        (tmp_path / row["images"]["exposed"][side]).write_bytes(
            (tmp_path / row["images"]["instant_mid"][side]).read_bytes())
    with pytest.raises(ValueError, match="blur extent|meaningful delivered blur"):
        verify_capture(tmp_path)


def test_reject_false_archived_time(tmp_path):
    write_capture(tmp_path)
    np.savez(tmp_path / "states.npz", time=[0.001], qpos=[[0]], qvel=[[40]])
    with pytest.raises(ValueError, match="Archived state timestamp"):
        verify_capture(tmp_path)
