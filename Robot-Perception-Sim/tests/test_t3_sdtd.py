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
    np.testing.assert_array_equal(up, [[0, 0, 0], [2, 4, 3]])
    np.testing.assert_array_equal(right, [[-3, -5, 0], [-5, -4, 0]])
    np.testing.assert_array_equal(td(gray + 2, gray), np.full_like(gray, 2))


def test_hdr_grayscale_uses_linear_luminance():
    hdr = np.zeros((1, 1, 4), dtype=np.float32)
    hdr[0, 0, :3] = [1, 2, 3]
    np.testing.assert_allclose(gray_from_hdr(hdr), [[1.8596]], rtol=0, atol=1e-6)
