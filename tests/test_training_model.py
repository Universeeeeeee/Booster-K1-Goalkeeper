"""Opt-in model regression: asset bounds, physical contact, and I1 semantics."""

import mujoco
import numpy as np
import pytest

from k1_goalkeeper.scene import build_scene
from k1_goalkeeper.simulation import Shot
from k1_goalkeeper.training_env import K1GoalkeeperEnv
from k1_goalkeeper.training_model import build_training_scene
from scripts.validate_training_model import coverage_probes, hull_proofs, model_audit, static_audit


@pytest.fixture(scope='module')
def training_models(tmp_path_factory):
    path = tmp_path_factory.mktemp('i2-models')
    paths = {'native': build_scene(path/'native.xml'),
             'training': build_training_scene(path/'training.xml')}
    return paths, {name: mujoco.MjModel.from_xml_path(str(p)) for name, p in paths.items()}


def test_preserves_original_xml_inertia_controls_and_default_builder(training_models):
    paths, models = training_models
    assert model_audit(paths['native'], paths['training'])['passed']
    assert models['training'].ngeom == models['native'].ngeom+2
    assert not any(models['native'].geom(i).name.startswith('i2_') for i in range(models['native'].ngeom))


def test_capsules_fit_official_mesh_convex_envelopes():
    assert all(proof['passed'] for proof in hull_proofs())


def test_actual_gap_contact_agrees_with_independent_segment_distance(training_models):
    _, models = training_models
    assert all(proof['passed'] for proof in coverage_probes(models))


def test_initial_and_small_motion_poses_have_no_self_penetration(training_models):
    _, models = training_models
    assert static_audit(models['training'])['passed']


@pytest.mark.parametrize('side,y', [('left', -.096), ('right', .096)])
def test_free_flight_delivers_force_to_added_shank_without_initial_overlap(training_models, side, y):
    _, models = training_models
    with K1GoalkeeperEnv(models['training'], control_dt=.001) as env:
        env.reset(options={'task': 'single_shot', 'launch_time': 0., 'time_limit': .1,
                           'shot': Shot(position=(10.12862374556, y, .1754145), velocity=(4., 0., 0.))})
        assert not any(env.ball_geom in (c.geom1, c.geom2) and c.dist < 0 for c in env.data.contact)
        force_samples = []
        while True:
            _, _, terminated, truncated, info = env.step(env.default_action)
            for index, contact in enumerate(env.data.contact):
                names = [env.model.geom(g).name for g in (contact.geom1, contact.geom2)]
                if 'ball_collision' in names and f'i2_{side}_shank' in names:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, index, force)
                    force_samples.append(force[0])
            if terminated or truncated:
                break
        assert info['transition_valid'] and info['robot_contact_seen']
        assert max(force_samples) > 1.
        assert env.data.qvel[env.mapping.ball_dof_address] < 3.


def test_training_model_reset_is_deterministic_and_actor_contract_preserved(training_models):
    _, models = training_models
    with K1GoalkeeperEnv(models['training']) as env:
        runs = []
        for _ in range(2):
            obs, _ = env.reset(seed=7, options={'time_limit': .1})
            assert obs.shape == (87,)
            rows = [obs]
            for _ in range(5):
                obs, _, terminated, truncated, info = env.step(env.default_action)
                rows.append(obs)
            assert truncated and not terminated and info['transition_valid']
            runs.append(rows)
        np.testing.assert_array_equal(*runs)


def test_training_model_retains_whole_ball_goal_termination(training_models):
    _, models = training_models
    with K1GoalkeeperEnv(models['training']) as env:
        env.reset(options={'task': 'single_shot', 'launch_time': 0., 'time_limit': 1.,
                           'shot': Shot(position=(10.9, 0., .5), velocity=(8., 0., 0.))})
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(env.default_action)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info['exit_event']['outcome'] == 'Goal'
        assert info['exit_event']['position'][0] == pytest.approx(11.11)
