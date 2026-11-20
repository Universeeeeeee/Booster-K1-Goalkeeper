"""Check the production SD kernels against the user's independent reference."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np


def test_opencv_sd_matches_supplied_numpy():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('reference',root/'T3定制相机需求说明书/TD:SD/simulate_sentinel_sdtd.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    gray=np.random.default_rng(42).integers(0,256,(37,53),dtype=np.uint8)
    up=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,-1,0],[0,1,0],[0,0,0]],np.float32));up[0]=0
    right=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,0,0],[0,1,-1],[0,0,0]],np.float32));right[:,-1]=0
    expected=ref.compute_tdsd_numpy(gray)
    np.testing.assert_array_equal(np.trunc(up / 2), expected[0])
    np.testing.assert_array_equal(np.trunc(right / 2), expected[1])


def test_golden_sd_extremes_and_negative_odd_differences():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        'reference', root / 'T3定制相机需求说明书/TD:SD/simulate_sentinel_sdtd.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    gray = np.array([[0, 255, 252, 253], [255, 0, 253, 252]], dtype=np.uint8)
    up, right, _ = ref.compute_tdsd_numpy(gray)
    np.testing.assert_array_equal(up, [[0, 0, 0, 0], [127, -127, 0, 0]])
    np.testing.assert_array_equal(right, [[-127, 1, 0, 0], [127, -126, 0, 0]])
    _, legacy_right, _ = ref.compute_tdsd_numpy(gray, legacy_sd=True)
    np.testing.assert_array_equal(legacy_right[0], [-255, 3, -1, 0])
