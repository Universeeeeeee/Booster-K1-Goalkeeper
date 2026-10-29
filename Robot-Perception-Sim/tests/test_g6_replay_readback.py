"""Check the independently executed USD readback artifact matches frozen code."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_actual_readback_covers_all_conditions_and_frozen_versions():
    report=json.loads((ROOT/'docs/g6a_replay_readback_acceptance.json').read_text())
    assert report['checks_passed']
    assert not report['renderer_used'] and not report['physics_stepped']
    assert report['source_helper_sha256']==hashlib.sha256((ROOT/'acquisition/g6_scene.py').read_bytes()).hexdigest()
    assert report['validator_sha256']==hashlib.sha256((ROOT/'scripts/validate_g6a_replay.py').read_bytes()).hexdigest()
    assert report['condition_count']==8 and report['checked_requests']==512
    assert len(report['static_robot_body_paths'])==21
    conditions={(c['trajectory'],c['head_pose'],c['exposure_ms'])for c in report['conditions']}
    assert conditions=={(case,pose,exposure) for case in ('direct_goal','one_bounce')
                       for pose in ('neutral','yaw_pitch') for exposure in (0,10)}
    for condition in report['conditions']:
        targets=17 if condition['trajectory']=='direct_goal' else 15
        assert condition['midpoints']==targets
        assert condition['subsamples']==targets*(1 if condition['exposure_ms']==0 else 5)
        assert condition['maximum_quantization_error_s']<.0005
    errors=report['max_errors']
    assert errors['translation_error_m']==0
    assert errors['other_body_matrix_change']==0
    assert errors['camera_state_matrix_error']==0
    assert errors['serialization_matrix_error']==0
    assert errors['stereo_matrix_error']<2e-6
    assert errors['quaternion_sign_invariant_l2_error']<1e-6
    assert errors['rotation_matrix_error']<2e-6
