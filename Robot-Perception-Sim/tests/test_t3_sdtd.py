import json
from pathlib import Path

import numpy as np

from robot_perception_sim.t3_sdtd import gray_from_hdr, sd, td


def test_stereo_candidate_geometry_and_optics():
    path = Path(__file__).resolve().parents[1] / "config/cameras/t3_stereo_park_candidate.json"
    stereo = json.loads(path.read_text())["stereo"]
    assert (stereo["width"], stereo["height"], stereo["fps"]) == (1280, 720, 100)
    assert stereo["focal_length_mm"] == 1.79
    np.testing.assert_allclose(stereo["K"], [[518.84057971, 0, 640], [0, 518.84057971, 360], [0, 0, 1]])
    assert stereo["baseline_m"] == 0.07
    np.testing.assert_allclose(stereo["relative_rotation"], np.eye(3))


def test_sd_td_match_reference_definition():
    gray = np.array([[1, 4, 9], [3, 8, 12]], dtype=np.float32)
    up, right = sd(gray)
    np.testing.assert_array_equal(up, [[128, 128, 128], [129, 130, 129]])
    np.testing.assert_array_equal(right, [[127, 126, 128], [126, 126, 128]])
    np.testing.assert_array_equal(td(gray + 2, gray), np.ones_like(gray))


def test_signed_td_for_all_uint8_pairs_and_first_sample():
    levels = np.arange(256, dtype=np.uint8)
    current, previous = np.meshgrid(levels, levels)
    delta = current.astype(np.int16) - previous.astype(np.int16)
    expected = (np.sign(delta) * (np.abs(delta) // 2)).astype(np.int8)
    actual = td(current, previous)
    assert actual.dtype == np.int8
    np.testing.assert_array_equal(actual, expected)
    first = td(current, None)
    assert first.dtype == np.int8
    np.testing.assert_array_equal(first, np.zeros_like(expected))


def test_hdr_grayscale_uses_linear_luminance():
    hdr = np.zeros((1, 1, 4), dtype=np.float32)
    hdr[0, 0, :3] = [1, 2, 3]
    np.testing.assert_allclose(gray_from_hdr(hdr), [[1.8596]], rtol=0, atol=1e-6)
