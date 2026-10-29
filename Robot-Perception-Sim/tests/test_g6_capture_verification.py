"""Acquisition integrity rejects altered pixels and mislabeled exposure clocks."""
from fractions import Fraction
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('g6verify', SCRIPTS/'verify_g6_capture.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def ratio(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


@pytest.fixture
def exposure(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'decode_frame_id', lambda a, centres: int(a[0, 0, centres]))
    start, duration = Fraction(1), Fraction(1, 100)
    mid = start+duration/2
    window = {'start_s': start, 'end_s': start+duration, 'mid_s': mid}
    state = {'window_rational': {k: ratio(v) for k,v in window.items()},
             **{'t_exposure_'+k: float(v) for k,v in window.items()},
             'sdk_composite_timestamp_ns': int(float(mid)*1e9), 'subframes': [],
             'composite_paths': {}, 'publication': {'timestamp_s':float(mid),
                 'host_armed_monotonic_ns':30, 'bytes_per_eye':16, 'source_sha256_left_right':[]}}
    metadata = {'configured_fps':60, 'frame_id_marker':{'pixel_centres':{'left':0,'right':0}},
                'acquisition':{'realized_exposure_s':.01, 'samples_per_exposure':3,
                    'subframe_marker_pixel_centres':{'left':1,'right':1}}}
    pixels = []
    for index in range(3):
        t = start+Fraction(2*index+1, 6)*duration
        image = np.full((2,2,4),255,dtype=np.uint8)
        image[...,0], image[...,1] = 7, index+1
        pixels.append(image)
        sub = {'target_s':ratio(t), 'subframe_id':index+1, 'host_authored_monotonic_ns':10,
               'host_observed_monotonic_ns':20,
               'state':{'motion_target_s':ratio(t), 'native_world_time_s':1.,
                        'native_world_time_after_render_s':1.}}
        for side in ('left','right'):
            name = f'{side}_{index}.png'
            Image.fromarray(image).save(tmp_path/name)
            sub[side+'_path'] = name
        state['subframes'].append(sub)
    composite = np.rint(np.mean(pixels,axis=0)).astype(np.uint8)
    for side in ('left','right'):
        name = side+'_composite.png'
        Image.fromarray(composite).save(tmp_path/name)
        state['composite_paths'][side] = name
        state['publication']['source_sha256_left_right'].append(hashlib.sha256(composite.tobytes()).hexdigest())
    return tmp_path,state,metadata


def test_paired_png_subsamples_recompute_exact_uploaded_mean(exposure):
    root,state,metadata=exposure
    assert verify.verify_exposure(root,state,7,metadata)==[1,2,3]


def test_composite_pixel_tamper_is_rejected(exposure):
    root,state,metadata=exposure
    path=root/state['composite_paths']['left']
    pixels=np.asarray(Image.open(path)).copy(); pixels[1,1,2]-=1
    Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError,match='not archived subframe average'):
        verify.verify_exposure(root,state,7,metadata)


def test_gpu_publication_hash_tamper_is_rejected(exposure):
    root,state,metadata=exposure
    state['publication']['source_sha256_left_right'][0]='0'*64
    with pytest.raises(ValueError,match='CUDA bytes'):
        verify.verify_exposure(root,state,7,metadata)


def test_false_midpoint_publication_is_rejected(exposure):
    root,state,metadata=exposure
    state['publication']['timestamp_s']+=.005
    with pytest.raises(ValueError,match='timestamp differs'):
        verify.verify_exposure(root,state,7,metadata)


def test_subframe_clock_and_causality_are_checked(exposure):
    root,state,metadata=exposure
    state['subframes'][0]['host_observed_monotonic_ns']=31
    with pytest.raises(ValueError,match='Noncausal'):
        verify.verify_exposure(root,state,7,metadata)


def test_state_cannot_advance_physics_during_render(exposure):
    root,state,metadata=exposure
    state['subframes'][0]['state']['native_world_time_after_render_s']=1.001
    with pytest.raises(ValueError,match='Physics advanced'):
        verify.verify_exposure(root,state,7,metadata)
