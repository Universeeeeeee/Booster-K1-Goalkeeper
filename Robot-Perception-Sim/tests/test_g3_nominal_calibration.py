import hashlib
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts.verify_g3_sdk_composite import verify_nominal_calibration


@pytest.fixture
def nominal(tmp_path):
    canonical = Path(__file__).parents[1] / "config/cameras/SN45626933.conf"
    (tmp_path / "settings").mkdir()
    copied = tmp_path / "settings/SN45626933.conf"
    shutil.copyfile(canonical, copied)
    K = {"fx": 370.8, "fy": 370.8, "cx": 480, "cy": 300}
    sdk = {"calibration_seed_relative_path": "settings/SN45626933.conf",
           "calibration_declared_sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
           "calibration_hardware_verified": False, "serial_number": 45626933,
           "width": 960, "height": 600, "baseline_m": 0.12,
           "calibration": {side: dict(K) for side in ("left", "right")}}
    cameras = {}
    for side, x in (("left", 0), ("right", 0.12)):
        pose = np.eye(4)
        pose[3, 0] = x
        cameras[side] = {**{key + "_px": value for key, value in K.items()},
                         "horizontal_aperture_offset_usd": 0, "vertical_aperture_offset_usd": 0,
                         "focal_length_usd": 1.1124, "horizontal_aperture_usd": 2.88, "vertical_aperture_usd": 1.8,
                         "world_from_usd_camera_row_matrix": pose.tolist()}
    return tmp_path, sdk, {"resolution": [960, 600], "cameras": cameras}


def test_checks_copied_nominal_bytes_full_K_baseline_and_USD_geometry(nominal):
    report = verify_nominal_calibration(*nominal)
    assert report["source_and_sdk_K_verified"]
    assert report["baseline_m"] == 0.12


def test_rejects_changed_copied_seed(nominal):
    root, sdk, source = nominal
    (root / sdk["calibration_seed_relative_path"]).write_text("# changed\n")
    with pytest.raises(ValueError, match="copied bytes"):
        verify_nominal_calibration(root, sdk, source)


@pytest.mark.parametrize("key", ["fx", "fy", "cx", "cy"])
def test_rejects_wrong_runtime_K_component(nominal, key):
    root, sdk, source = nominal
    sdk["calibration"]["right"][key] += 1
    with pytest.raises(ValueError, match="runtime right"):
        verify_nominal_calibration(root, sdk, source)


def test_rejects_millimetres_reported_as_metres(nominal):
    root, sdk, source = nominal
    sdk["baseline_m"] = 120
    with pytest.raises(ValueError, match="millimetre seed"):
        verify_nominal_calibration(root, sdk, source)


def test_rejects_source_USD_wrong_stereo_separation(nominal):
    root, sdk, source = nominal
    source["cameras"]["right"]["world_from_usd_camera_row_matrix"][3][0] = 120
    with pytest.raises(ValueError, match="Source USD stereo separation"):
        verify_nominal_calibration(root, sdk, source)


def test_rejects_source_K_matching_seed_with_wrong_USD_film_aperture(nominal):
    root, sdk, source = nominal
    source["cameras"]["left"]["vertical_aperture_usd"] *= 2
    with pytest.raises(ValueError, match="focal/aperture"):
        verify_nominal_calibration(root, sdk, source)
