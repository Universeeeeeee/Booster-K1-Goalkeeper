"""Check the production SD kernels against the user's independent reference."""
import importlib.util
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from robot_perception_sim.t3_sdtd import TD_ENCODING, sd
from scripts.verify_t3_abc import verify


def test_opencv_sd_matches_supplied_numpy():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('reference',root/'T3定制相机需求说明书/TD:SD/simulate_sentinel_sdtd.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    gray=np.random.default_rng(42).integers(0,256,(37,53),dtype=np.uint8)
    up=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,-1,0],[0,1,0],[0,0,0]],np.float32));up[0]=0
    right=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,0,0],[0,1,-1],[0,0,0]],np.float32));right[:,-1]=0
    expected=ref.compute_tdsd_numpy(gray)
    np.testing.assert_array_equal(np.trunc(up / 2) + 128, expected[0])
    np.testing.assert_array_equal(np.trunc(right / 2) + 128, expected[1])


def test_golden_sd_extremes_and_negative_odd_differences():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        'reference', root / 'T3定制相机需求说明书/TD:SD/simulate_sentinel_sdtd.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    gray = np.array([[0, 255, 252, 253], [255, 0, 253, 252]], dtype=np.uint8)
    up, right, _ = ref.compute_tdsd_numpy(gray)
    np.testing.assert_array_equal(up, [[128, 128, 128, 128], [255, 1, 128, 128]])
    np.testing.assert_array_equal(right, [[1, 129, 128, 128], [255, 2, 128, 128]])
    _, legacy_right, _ = ref.compute_tdsd_numpy(gray, legacy_sd=True)
    np.testing.assert_array_equal(legacy_right[0], [-255, 3, -1, 0])


def test_golden_td_preserves_polarity_and_matches_torch_for_byte_inputs():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        'reference', root / 'T3定制相机需求说明书/TD:SD/simulate_sentinel_sdtd.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    gray = np.array([[0, 255, 2, 5, 4, 5, 0]], dtype=np.uint8)
    previous = np.array([[255, 0, 5, 2, 5, 4, 0]], dtype=np.uint8)
    expected = np.array([[-127, 127, -1, 1, 0, 0, 0]], dtype=np.int8)
    actual = ref.compute_tdsd_numpy(gray, previous)[2]
    assert actual.dtype == np.int8
    np.testing.assert_array_equal(actual, expected)
    if ref.torch is not None:
        actual = ref.compute_tdsd(ref.torch.from_numpy(gray), ref.torch.from_numpy(previous))[2]
        assert actual.dtype == ref.torch.int8
        np.testing.assert_array_equal(actual.cpu().numpy()[0, 0], expected)


@pytest.mark.parametrize(('case', 'error'), [
    ('valid', None),
    ('stale_gray', 'Left TD source gray ID 1, expected 0'),
    ('wrong_mode', 'TD encoding requires stereo difference capture'),
    ('missing_td', 'TD hashes do not match'),
])
def test_td_capture_requires_matching_gray_and_manifest(tmp_path, case, error):
    centres = [(3 + 6*i, 3) for i in range(23)]
    gray = np.zeros((7, 138), dtype=np.uint8)
    for cell in (0, 2, 22):
        x, y = centres[cell]
        gray[y-2:y+3, x-2:x+3] = 255
    rgb = np.repeat(gray[..., None], 3, axis=2)
    hashes, optics = {}, {}
    for eye in ('Left', 'Right'):
        pose = np.eye(4)
        if eye == 'Right':
            pose[0, 3] = 0.07
        optics[eye] = {'resolution': [138, 7], 'local_matrix': pose.tolist()}
        source = gray.copy()
        if eye == 'Left' and case == 'stale_gray':
            for cell in (19, 20):  # frame 1, including parity
                x, y = centres[cell]
                source[y-2:y+3, x-2:x+3] = 255
        for suffix, data in [('rgb', rgb), ('gray', source), ('sd', np.stack(sd(source))),
                             ('td', np.zeros_like(gray, dtype=np.int8))]:
            payload = data.tobytes()
            hashes[eye if suffix == 'rgb' else eye + '_' + suffix] = hashlib.sha256(payload).hexdigest()
            if case != 'missing_td' or suffix != 'td':
                (tmp_path / f'{eye}.{suffix}').write_bytes(payload)
    meta = {'mode': 'A' if case == 'wrong_mode' else 'B', 'frames': 1,
            'elapsed_s': 1, 'optics': optics, 'centres': dict.fromkeys(optics, centres),
            'rows': [{'index': 0, 'simulation_ns': 0, 'hashes': hashes}],
            'sd_encoding': 'half_trunc_plus_128'}
    if case != 'missing_td':
        meta['td_encoding'] = TD_ENCODING
    (tmp_path / 'profile.json').write_text(json.dumps(meta))
    reference = Path(__file__).resolve().parents[2] / 'T3定制相机需求说明书/TD:SD/simulate_sentinel_sdtd.py'
    if error is None:
        assert verify(tmp_path, reference)['td_frames_verified_per_eye'] == 1
    else:
        with pytest.raises(ValueError, match=error):
            verify(tmp_path, reference)
