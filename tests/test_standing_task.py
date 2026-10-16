"""Standing reward, action semantics, disturbances and invalid-transition gates."""

import numpy as np
import pytest

from k1_goalkeeper.standing_task import ACTION_SCALE, StandingTask, standing_reward


@pytest.fixture
def task():
    with StandingTask() as env:
        yield env


def test_reward_prefers_upright_quiet_posture_and_penalizes_fall():
    state = {'gravity': np.array([0., 0., -1.]), 'position_error': np.zeros(3),
             'linear_velocity': np.zeros(3), 'angular_velocity': np.zeros(3),
             'joint_offset': np.zeros(22), 'joint_velocity': np.zeros(22), 'torque_ratio': np.zeros(22)}
    zero = np.zeros(22)
    ideal = standing_reward(state, zero, zero, .02, False)
    moving = {**state, 'gravity': np.array([.3, 0., -.954]), 'linear_velocity': np.ones(3),
              'joint_velocity': np.ones(22)*3.}
    assert sum(ideal.values()) > sum(standing_reward(moving, zero, zero, .02, False).values())
    assert sum(standing_reward(state, zero, zero, .02, True).values()) < -4.
    assert sum(standing_reward(state, zero, zero, .01, False).values()) == pytest.approx(sum(ideal.values())/2)
    assert standing_reward(state, np.ones(22), zero, .02, False)['action_rate'] < 0


@pytest.mark.parametrize('action', [np.zeros(21), np.full(22, np.nan), np.full(22, 1.01)])
def test_invalid_action_does_not_advance_physics(task, action):
    task.reset()
    with pytest.raises(ValueError):
        task.step(action)
    assert task.env.data.time == 0.


@pytest.mark.parametrize('force', [[1.], [1., np.inf], [1., 2., 3.]])
def test_invalid_push_is_rejected(task, force):
    with pytest.raises(ValueError):
        task.reset(options={'push_force': force})


def test_action_scale_and_timeout_contract(task):
    task.reset(options={'push_force': [0., 0.]})
    action = np.linspace(-1., 1., 22).astype(np.float32)
    obs, reward, terminated, truncated, info = task.step(action)
    np.testing.assert_allclose(task.env.backend._targets, task.reference+ACTION_SCALE*action)
    np.testing.assert_array_equal(obs[-22:], action)
    assert task.observation_space.contains(obs) and np.isfinite(reward)
    assert not terminated and not truncated and info['bootstrap_allowed']


def test_future_push_is_not_in_actor_and_delivered_pulse_has_correct_impulse(task):
    prefixes, impulses = [], []
    for force in ([0., 0.], [10., -10.]):
        obs, _ = task.reset(seed=7, options={'push_force': force})
        prefix = [obs]
        impulse = np.zeros(2)
        for step in range(60):
            obs, _, terminated, truncated, info = task.step(np.zeros(22, dtype=np.float32))
            if step < 50:
                prefix.append(obs)
            impulse += np.asarray(info['applied_push_force'])*info['elapsed_dt']
            assert not terminated and not truncated
        prefixes.append(prefix)
        impulses.append(impulse)
    np.testing.assert_array_equal(*prefixes)
    np.testing.assert_allclose(impulses[1], [1., -1.], atol=1e-12)
    np.testing.assert_array_equal(impulses[0], 0.)


def test_invalid_transition_aborts_before_ppo_can_store_it(task, monkeypatch):
    task.reset()
    native = task.env.step
    def invalid(action):
        obs, reward, terminated, truncated, info = native(action)
        return obs, reward, terminated, truncated, {**info, 'transition_valid': False, 'numerical_error': 'injected'}
    monkeypatch.setattr(task.env, 'step', invalid)
    with pytest.raises(FloatingPointError):
        task.step(np.zeros(22, dtype=np.float32))


def test_ppo_vector_wrapper_preserves_timeout_bootstrap_signal():
    pytest.importorskip('stable_baselines3')
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: StandingTask(episode_duration=.02)])
    try:
        env.reset()
        _, _, done, infos = env.step(np.zeros((1, 22), dtype=np.float32))
        assert done[0] and infos[0]['TimeLimit.truncated']
        assert infos[0]['terminal_observation'].shape == (78,)
    finally:
        env.close()
