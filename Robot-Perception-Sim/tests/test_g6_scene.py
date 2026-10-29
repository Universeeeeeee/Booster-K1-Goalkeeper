"""CPU contract checks; actual USD two-pose audit is validate_g6a_mount.py."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acquisition.g6_scene import HEAD_POSES, nearest_sample, sphere_fov


def test_replay_uses_nearest_exact_record_without_interpolation():
    samples=[{'time':0.,'position':[9,0,1]}, {'time':.001,'position':[9.008,0,.999]},
             {'time':.002,'position':[9.016,0,.998]}]
    sample,index=nearest_sample(samples,.0014)
    assert index==1 and sample is samples[1]
    assert nearest_sample(samples,-1)[0] is samples[0]
    assert nearest_sample(samples,2)[0] is samples[-1]
    with pytest.raises(ValueError):nearest_sample(samples,float('nan'))


def test_fov_includes_sphere_limb_and_rejects_behind_camera():
    assert sphere_fov([0,0,1],.11)['fully_in_fov']
    assert not sphere_fov([0,0,-1],.11)['in_fov']
    assert not sphere_fov([3,0,1],.11)['in_fov']
    assert sphere_fov([1.35,0,1],.11)['in_fov']
    assert not sphere_fov([1.35,0,1],.11)['fully_in_fov']


def test_frozen_inputs_and_head_pose_contract():
    config=json.loads((ROOT/'config/scenarios/g6a_sources.json').read_text())
    assert set(config['trajectories'])=={'direct_goal','one_bounce'}
    assert config['trajectories']['direct_goal']['source_shot']['name']=='direct_middle_center'
    assert config['trajectories']['one_bounce']['source_shot']['name']=='bounce_one'
    assert HEAD_POSES['neutral']==(0.,0.)
    assert HEAD_POSES['yaw_pitch']==(.2,.1)
    assert config['head_poses_rad']['yaw_pitch']==[.2,.1]


def test_saved_actual_usd_mount_audit_two_poses():
    audit=json.loads((ROOT/'docs/g6a_mount_acceptance.json').read_text())
    assert audit['checks_passed']
    for name,pose in audit['poses'].items():
        c=pose['camera'];stereo=np.asarray(c['T_left_from_right'])
        assert np.allclose(stereo,np.array([[1,0,0,.12],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),atol=2e-6)
        assert np.allclose(c['K'],[[370.8,0,480],[0,370.8,300],[0,0,1]],atol=1e-4)
        forward=np.asarray(c['left_optical_axis_world'])
        if name=='neutral':assert np.allclose(forward,[-1,0,0],atol=1e-6)
        else:assert forward[2]<-.09
        for samples in pose['trajectory_samples'].values():
            assert samples[0]['visibility_reason']=='visible'
            assert samples[-1]['visibility_reason']=='out_of_fov'


def test_alpha_zero_visual_layer_has_independent_renderer_evidence():
    import hashlib
    report=json.loads((ROOT/'docs/g6a_visual_fidelity_acceptance.json').read_text())
    assert report['runtime_helper_sha256']==hashlib.sha256((ROOT/'acquisition/g6_scene.py').read_bytes()).hexdigest()
    overrides=report['scene_provenance']['zero_opacity_visibility_overrides']
    assert len(overrides)==10
    assert all(item['source_display_opacity']==[0.] and item['g6_visibility']=='invisible' for item in overrides)
    assert all('/b28_ball/' not in item['path'] for item in overrides)
    assert np.allclose(report['turf_display_color'][0],[.12,.36,.18])
    assert report['variants']['A_original']['roi_green_fraction']==0
    assert report['variants']['B_ground_invisible']['roi_green_fraction']>.5
    assert report['variants']['D_all_source_alpha_zero_invisible']['roi_green_fraction']>.5
    assert report['scene_provenance']['ball_appearance']=='frozen_g1_plain_white_no_checker_texture'
