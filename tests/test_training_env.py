"""I1 contracts: current observations, delivered physics, and episode isolation."""

import builtins
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from k1_goalkeeper.training_actions import JointTargetBackend
from k1_goalkeeper.training_env import K1GoalkeeperEnv, baseline_reward
from k1_goalkeeper.simulation import Shot


@pytest.fixture
def env(free_model):
    result = K1GoalkeeperEnv(free_model)
    yield result
    result.close()


@pytest.fixture
def velocity_env(free_model):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    if not (Path(__file__).resolve().parents[1] / "assets/booster_deploy/manifest.json").exists():
        pytest.skip("optional walking policy assets are unavailable")
    result = K1GoalkeeperEnv(free_model, action_mode="body_velocity")
    yield result
    result.close()


def feature(env, observation, name):
    return observation[slice(*env.observation_layout[name])]


def assert_tree_equal(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_tree_equal(left, right)
    else:
        assert first == second


def test_target_backend_does_not_import_torch_or_walk_policy(free_model, monkeypatch):
    native_import = builtins.__import__

    def deny_policy(name, *args, **kwargs):
        if name == "torch" or name.endswith("walk_controller"):
            raise AssertionError("direct joint targets loaded the optional walking stack")
        return native_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_policy)
    backend = JointTargetBackend(free_model)
    backend.set_action(backend.default_action)
    backend.step_physics()
    assert backend.data.time == pytest.approx(.001)


def test_target_backend_delivers_official_pd_per_physics_tick(free_model, monkeypatch):
    backend = JointTargetBackend(free_model)
    mapping = backend.mapping
    kp = np.array([4, 4] + [20] * 8 + [100, 100, 100, 100, 65, 65] * 2,
                  dtype=np.float32)
    kd = np.array([1, 1] + [2] * 8 + [2, 2, 2, 2, 1, 1] * 2, dtype=np.float32)
    effort = np.array([6, 6] + [14] * 8 + [30, 20, 15, 35, 24, 15] * 2,
                      dtype=np.float32)
    np.testing.assert_array_equal(mapping.kp, kp)
    np.testing.assert_array_equal(mapping.kd, kd)
    np.testing.assert_array_equal(mapping.effort, effort)
    action = backend.default_action.copy()
    action[0] += .1
    backend.set_action(action)
    native_step, controls = mujoco.mj_step, []

    def inspect_step(model, data):
        q = np.array([int(model.joint(name).qposadr[0]) for name in mapping.joint_names])
        v = np.array([int(model.joint(name).dofadr[0]) for name in mapping.joint_names])
        motors = np.array([model.actuator(name).id for name in mapping.joint_names])
        expected = np.clip(kp * (action - data.qpos.astype(np.float32)[q])
                           - kd * data.qvel.astype(np.float32)[v], -effort, effort)
        np.testing.assert_array_equal(data.ctrl[motors], expected)
        np.testing.assert_array_equal(data.qfrc_applied, 0)
        np.testing.assert_array_equal(data.xfrc_applied, 0)
        controls.append(data.ctrl[motors].copy())
        native_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", inspect_step)
    for _ in range(20):
        backend.step_physics()
    assert len(controls) == backend.physics_steps == 20
    assert backend.data.time == pytest.approx(.02)
    assert not np.array_equal(controls[0], controls[-1])


def test_target_backend_named_mapping_with_reordered_scene(scene_paths):
    root = ET.parse(scene_paths["free"]).getroot()
    body = ET.Element("body", name="i1_extra", pos="5 5 1")
    ET.SubElement(body, "joint", name="i1_extra_joint", type="hinge")
    ET.SubElement(body, "geom", type="sphere", size=".1", mass="1")
    root.find("worldbody").insert(0, body)
    actuators = root.find("actuator")
    ET.SubElement(actuators, "motor", name="i1_extra_motor", joint="i1_extra_joint")
    actuators[:] = list(reversed(list(actuators)))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    backend = JointTargetBackend(model)
    assert backend.mapping.base_qpos_address == 1
    assert not np.array_equal(backend.mapping.actuator_ids, np.arange(22))
    extra_motor = model.actuator("i1_extra_motor").id
    backend.data.ctrl[extra_motor] = .123
    ball_q = int(model.joint("ball_free").qposadr[0])
    ball_v = int(model.joint("ball_free").dofadr[0])
    ball_before = backend.data.qpos[ball_q:ball_q + 7].copy()
    velocity_before = backend.data.qvel[ball_v:ball_v + 6].copy()
    backend.set_action(backend.default_action)
    np.testing.assert_array_equal(backend.data.qpos[ball_q:ball_q + 7], ball_before)
    np.testing.assert_array_equal(backend.data.qvel[ball_v:ball_v + 6], velocity_before)
    backend.step_physics()
    assert backend.data.ctrl[extra_motor] == .123
    np.testing.assert_array_equal(backend.data.qfrc_actuator[ball_v:ball_v + 6], 0)


@pytest.mark.parametrize("fixture,dimension", [("env", 22), ("velocity_env", 3)])
def test_gymnasium_contract_and_initial_observation(fixture, dimension, request):
    env = request.getfixturevalue(fixture)
    observation, info = env.reset(seed=7)
    assert env.action_space.shape == (dimension,)
    assert env.action_space.contains(env.default_action)
    assert env.observation_space.shape == (65 + dimension,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["simulation_time"] == env.data.time == 0
    assert info["physics_steps"] == info["decision_steps"] == 0
    assert info["termination_reason"] is None
    np.testing.assert_array_equal(feature(env, observation, "previous_action"), 0)
    np.testing.assert_array_equal(feature(env, observation, "joint_position_offset"), 0)
    np.testing.assert_array_equal(feature(env, observation, "joint_velocity"), 0)
    np.testing.assert_array_equal(feature(env, observation, "projected_gravity"), [0, 0, -1])
    assert env.privileged_observation_space.contains(env.get_privileged_state())
    next_observation, reward, terminated, truncated, info = env.step(env.default_action)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float) and reward == 0
    assert terminated is False and truncated is False
    assert env.data.time == pytest.approx(.02)
    assert info["physics_steps"] == 20 and info["decision_steps"] == 1
    assert info["elapsed_dt"] == pytest.approx(.02)
    assert info["bootstrap_allowed"] and not info["discard_transition"]
    np.testing.assert_array_equal(feature(env, next_observation, "previous_action"), env.default_action)
    assert sum(info["reward_terms"].values()) == reward
    check_env(env, skip_render_check=True)


def test_layout_uses_current_body_frame_and_named_joint_state(env):
    env.reset()
    m, d = env.mapping, env.data
    q, v, bq, bv = m.base_qpos_address, m.base_dof_address, m.ball_qpos_address, m.ball_dof_address
    d.qpos[q + 3:q + 7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    d.qvel[v:v + 6] = [1, 2, 3, .1, .2, .3]
    d.qpos[bq:bq + 3] = [9, 1, .8]
    d.qvel[bv:bv + 3] = [4, 5, 6]
    mujoco.mj_forward(env.model, d)
    # Stop at the current state to inspect the public step result without a rollout.
    d.qpos[q + 2] = .29
    observation, _, terminated, _, _ = env.step(env.default_action)
    assert terminated
    np.testing.assert_allclose(feature(env, observation, "body_linear_velocity"), [2, -1, 3], atol=1e-6)
    np.testing.assert_allclose(feature(env, observation, "body_angular_velocity"), [.1, .2, .3], atol=1e-6)
    np.testing.assert_allclose(feature(env, observation, "ball_relative_position"),
                               [1, 1.4, .51], atol=1e-6)
    np.testing.assert_allclose(feature(env, observation, "ball_relative_linear_velocity"),
                               [3, -3, 3], atol=1e-6)


def test_pending_shot_launch_time_and_seed_do_not_leak(env):
    first_shot = Shot("hidden_a", (9, -2, .5), (4, 1, 0), (1, 2, 3), seed=10)
    second_shot = Shot("hidden_b", (6, 2, 1), (10, -2, 5), (5, 4, 3), seed=90)
    observations, critics, infos, evaluations = [], [], [], []
    for shot, launch, seed in [(first_shot, 1., 3), (second_shot, 3., 99)]:
        initial, initial_info = env.reset(seed=seed, options={"task": "single_shot", "shot": shot,
                                                            "launch_time": launch})
        step_result = env.step(env.default_action)
        observations.append([initial, step_result[0]])
        critics.append(env.get_privileged_state())
        infos.append([initial_info, step_result[4]])
        evaluations.append(env.get_evaluation_state())
    assert_tree_equal(observations[0], observations[1])
    assert_tree_equal(critics[0], critics[1])
    assert_tree_equal(infos[0], infos[1])
    assert evaluations[0]["scheduled_shot"] != evaluations[1]["scheduled_shot"]
    assert evaluations[0]["scheduled_launch_time"] == 1
    assert evaluations[1]["scheduled_launch_time"] == 3
    forbidden = {"scheduled_shot", "scheduled_launch_time", "delivered_launch", "critic", "privileged_state"}
    assert all(forbidden.isdisjoint(info) for info in infos[0])
    assert forbidden.isdisjoint(critics[0])


def test_returned_state_arrays_and_offline_launch_record_are_copies(env):
    observation, info = env.reset(options={"task": "single_shot", "launch_time": 0,
                                          "shot": Shot(position=(7, 0, .5))})
    qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
    observation[:] = 123
    privileged = env.get_privileged_state()
    for values in privileged.values():
        values[:] = 456
    evaluation = env.get_evaluation_state()
    evaluation["qpos"][:] = 789
    evaluation["qvel"][:] = 789
    evaluation["ctrl"][:] = 789
    evaluation["delivered_launch"]["position"][0] = 789
    info["reward_terms"]["fake"] = 100
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.qvel, qvel)
    assert env.get_evaluation_state()["delivered_launch"]["position"][0] == 7
    assert np.isfinite(env.get_privileged_state()["qpos"]).all()


@pytest.mark.parametrize("fixture", ["env", "velocity_env"])
@pytest.mark.parametrize("invalid", ["short", "long", "matrix", "nan", "inf", "low", "high"])
def test_invalid_actions_are_rejected_before_any_episode_mutation(fixture, invalid, request):
    env = request.getfixturevalue(fixture)
    env.reset()
    env.step(env.default_action)
    action = env.default_action.copy()
    if invalid == "short":
        action = action[:-1]
    elif invalid == "long":
        action = np.append(action, 0)
    elif invalid == "matrix":
        action = action[None, :]
    elif invalid in ("nan", "inf"):
        action[0] = np.nan if invalid == "nan" else np.inf
    else:
        action[0] = env.action_space.low[0] - .01 if invalid == "low" else env.action_space.high[0] + .01
    before = env.get_evaluation_state()
    previous = env._previous_action.copy()
    decisions = env._decision_steps
    history = None
    if fixture == "velocity_env":
        history = env.backend.simulation.controller.policy.obs_history.clone()
    with pytest.raises(ValueError):
        env.step(action)
    assert_tree_equal(env.get_evaluation_state(), before)
    np.testing.assert_array_equal(env._previous_action, previous)
    assert env._decision_steps == decisions
    if history is not None:
        np.testing.assert_array_equal(env.backend.simulation.controller.policy.obs_history.numpy(), history.numpy())


@pytest.mark.parametrize("mode", ["joint_targets", "body_velocity"])
def test_action_bounds_have_documented_units_and_valid_boundaries(mode, free_model):
    if mode == "body_velocity":
        pytest.importorskip("torch").set_num_threads(1)
    env = K1GoalkeeperEnv(free_model, action_mode=mode, episode_duration=.001)
    if mode == "body_velocity":
        np.testing.assert_array_equal(env.action_space.low, np.array([-.3, -.3, -1.8], dtype=np.float32))
        np.testing.assert_array_equal(env.action_space.high, np.array([1.6, .3, 1.8], dtype=np.float32))
    else:
        ranges = np.array([free_model.joint(name).range for name in env.mapping.joint_names])
        np.testing.assert_array_equal(env.action_space.low, ranges[:, 0].astype(np.float32))
        np.testing.assert_array_equal(env.action_space.high, ranges[:, 1].astype(np.float32))
    for action in (env.action_space.low, env.action_space.high):
        env.reset()
        assert env.step(action)[3]
    env.close()


def test_launch_is_on_exact_physics_tick_and_teleport_refreshes_detector(env, monkeypatch):
    env.reset(options={"task": "single_shot", "shot": Shot(position=(10, 6, .5), velocity=(0, 0, 0)),
                       "launch_time": .003, "time_limit": .007})
    native_step, samples = mujoco.mj_step, []
    ball_q = env.mapping.ball_qpos_address

    def inspect_step(model, data):
        samples.append((float(data.time), data.qpos[ball_q:ball_q + 3].copy()))
        native_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", inspect_step)
    _, reward, terminated, truncated, info = env.step(env.default_action)
    assert len(samples) == 7
    assert samples[2][0] == pytest.approx(.002)
    assert samples[2][1][0] == -7
    assert samples[3][0] == pytest.approx(.003)
    np.testing.assert_array_equal(samples[3][1], [10, 6, .5])
    assert reward == 0 and not terminated and truncated
    assert info["exit_event"] is None
    assert env.get_evaluation_state()["delivered_launch"]["time"] == pytest.approx(.003)


@pytest.mark.parametrize("limit", [.001, .007, .023])
def test_time_limit_stops_on_exact_substep_and_requires_reset(env, limit):
    env.reset(options={"time_limit": limit})
    done = False
    while not done:
        observation, reward, terminated, truncated, info = env.step(env.default_action)
        done = terminated or truncated
    assert not terminated and truncated
    assert info["termination_reason"] == "time_limit" and info["bootstrap_allowed"]
    assert info["physics_steps"] == round(limit / .001)
    assert env.data.time == pytest.approx(limit)
    assert reward == 0 and np.isfinite(observation).all()
    before = env.get_evaluation_state()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.default_action)
    assert_tree_equal(env.get_evaluation_state(), before)
    env.reset()
    assert not env.step(env.default_action)[3]


@pytest.mark.parametrize("position,velocity,outcome,goal,reward", [
    ((11.095, 0, .5), (10, 0, 0), "Goal", "right", -1.),
    ((-11.095, 0, .5), (-10, 0, 0), "Goal", "left", 0.),
    ((11.095, 2, .5), (10, 0, 0), "Miss", "right", 0.),
    ((0, 7.095, .5), (0, 10, 0), "Miss", None, 0.),
])
def test_whole_ball_exit_is_substep_interpolated_and_terminal(env, position, velocity, outcome, goal, reward):
    env.reset(options={"task": "single_shot", "launch_time": 0,
                       "shot": Shot(position=position, velocity=velocity)})
    _, actual_reward, terminated, truncated, info = env.step(env.default_action)
    assert terminated and not truncated and actual_reward == reward
    event = info["exit_event"]
    assert event["outcome"] == outcome and event["goal"] == goal
    assert event["time"] == pytest.approx(.0015, abs=1e-5)
    assert info["physics_steps"] == 2
    assert .001 < event["time"] < info["simulation_time"]
    axis = 1 if goal is None else 0
    assert abs(event["position"][axis]) == pytest.approx(7.11 if axis else 11.11)


def test_ball_center_crossing_does_not_end_episode(env):
    env.reset(options={"task": "single_shot", "launch_time": 0,
                       "shot": Shot(position=(10.999, 0, .5), velocity=(1, 0, 0))})
    _, reward, terminated, truncated, info = env.step(env.default_action)
    assert env.data.qpos[env.mapping.ball_qpos_address] > 11
    assert not terminated and not truncated and reward == 0
    assert info["exit_event"] is None


def test_normal_ball_contact_does_not_imply_save_or_termination(env):
    env.reset(options={"task": "single_shot", "launch_time": 0,
                       "shot": Shot(position=(7, 0, .11), velocity=(0, 0, 0))})
    _, reward, terminated, truncated, info = env.step(env.default_action)
    assert info["active_contact_count"] > 0
    assert not terminated and not truncated and reward == 0
    assert info["exit_event"] is None


def test_actual_ball_robot_contact_does_not_imply_save_or_termination(scene_paths):
    model = mujoco.MjModel.from_xml_path(str(scene_paths["free"]))
    env = K1GoalkeeperEnv(model)
    env.reset(options={"task": "single_shot", "launch_time": 0,
                       "shot": Shot(position=(10.15, 0, .5), velocity=(2, 0, 0))})
    mujoco.mj_forward(model, env.data)
    assert all(contact.dist >= 0 for contact in env.data.contact
               if env.ball_geom in (contact.geom1, contact.geom2))
    for _ in range(10):
        _, reward, terminated, truncated, info = env.step(env.default_action)
        assert not terminated and not truncated and reward == 0
        if info["robot_contact_seen"]:
            break
    assert info["robot_contact_seen"]
    assert info["exit_event"] is None and info["termination_reason"] is None
    env.close()


@pytest.mark.parametrize("failure", ["height", "tilt"])
def test_fall_terminates_before_further_physics(env, failure):
    env.reset()
    q = env.mapping.base_qpos_address
    if failure == "height":
        env.data.qpos[q + 2] = .29
    else:
        angle = np.deg2rad(61)
        env.data.qpos[q + 3:q + 7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
    mujoco.mj_forward(env.model, env.data)
    observation, reward, terminated, truncated, info = env.step(env.default_action)
    assert terminated and not truncated and reward == -1
    assert info["termination_reason"] == "fall" and info["transition_valid"]
    assert env.data.time == 0 and info["physics_steps"] == 0
    assert np.isfinite(observation).all()


@pytest.mark.parametrize("failure", ["nonfinite", "warning", "time_reset"])
def test_numerical_failure_truncates_with_last_valid_observation(env, monkeypatch, failure):
    initial, _ = env.reset()
    native_step, calls = mujoco.mj_step, []

    def inject_failure(model, data):
        native_step(model, data)
        calls.append(1)
        if failure == "nonfinite":
            data.qvel[0] = np.nan
        elif failure == "warning":
            data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number += 1
        else:
            mujoco.mj_resetData(model, data)

    monkeypatch.setattr(mujoco, "mj_step", inject_failure)
    observation, reward, terminated, truncated, info = env.step(env.default_action)
    assert calls == [1] and not terminated and truncated and reward == 0
    np.testing.assert_array_equal(observation, initial)
    assert info["termination_reason"] == "numerical_error"
    assert info["invalid_state"] and not info["transition_valid"] and not info["bootstrap_allowed"]
    assert info["discard_transition"]
    assert info["numerical_error"]["message"]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.default_action)


def test_repeated_reset_reproduces_rollout_and_clears_episode_state(env):
    options = {"task": "single_shot", "shot": Shot(position=(9, 0, .5)), "launch_time": .03}
    first = env.reset(seed=42, options=options)
    first_steps = [env.step(env.default_action) for _ in range(3)]
    assert env.get_evaluation_state()["delivered_launch"]
    second = env.reset(seed=42, options=options)
    assert_tree_equal(first, second)
    assert env.get_evaluation_state()["delivered_launch"] is None
    assert_tree_equal(first_steps, [env.step(env.default_action) for _ in range(3)])
    observation, info = env.reset()
    assert info["exit_event"] is None and not info["robot_contact_seen"]
    assert info["active_contact_count"] == 0
    np.testing.assert_array_equal(feature(env, observation, "previous_action"), 0)


def test_velocity_reset_clears_policy_history_without_warmup(velocity_env):
    env = velocity_env
    env.reset()
    policy = env.backend.simulation.controller.policy
    assert policy.obs_history is None
    first = env.step(np.array([0, .1, 0], dtype=np.float32))
    assert policy.obs_history.shape == (10, 69)
    env.reset()
    assert env.data.time == 0 and policy.obs_history is None
    np.testing.assert_array_equal(policy.last_action.numpy(), 0)
    np.testing.assert_array_equal(policy.filtered_dof_target.numpy(), env.mapping.default_positions)
    assert_tree_equal(first, env.step(np.array([0, .1, 0], dtype=np.float32)))


def test_environments_own_independent_data_and_rng(free_model):
    first, second = K1GoalkeeperEnv(free_model), K1GoalkeeperEnv(free_model)
    assert first.data is not second.data
    first.reset(seed=1)
    second.reset(seed=2)
    before = second.get_evaluation_state()
    first.step(first.default_action)
    assert_tree_equal(second.get_evaluation_state(), before)
    first.reset(seed=3)
    assert_tree_equal(second.get_evaluation_state(), before)
    assert first.np_random is not second.np_random
    first.close()
    second.close()


def test_seed_does_not_reseed_global_numpy_rng(env):
    state = np.random.get_state()
    env.reset(seed=12)
    assert_tree_equal(np.random.get_state(), state)


def test_ball_spin_is_current_critic_information_only(env):
    actors, spins = [], []
    for spin in ((1, 2, 3), (4, 5, 6)):
        observation, _ = env.reset(options={"task": "single_shot", "launch_time": 0,
                                            "shot": Shot(position=(7, 0, .5), angular_velocity=spin)})
        actors.append(observation)
        spins.append(env.get_privileged_state()["ball_angular_velocity_body"])
    np.testing.assert_array_equal(actors[0], actors[1])
    np.testing.assert_array_equal(spins, [[1, 2, 3], [4, 5, 6]])


def test_inactive_and_closed_environments_require_reset_or_reject_access(env):
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.default_action)
    with pytest.raises(RuntimeError, match="reset"):
        env.get_privileged_state()
    with pytest.raises(RuntimeError, match="reset"):
        env.get_evaluation_state()
    env.reset()
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    with pytest.raises(RuntimeError):
        env.step(env.default_action)
    with pytest.raises(RuntimeError):
        env.get_privileged_state()


@pytest.mark.parametrize("kwargs", [
    {"action_mode": "invalid"}, {"control_dt": 0}, {"control_dt": .0005},
    {"control_dt": np.inf}, {"episode_duration": 0}, {"episode_duration": .0005},
    {"episode_duration": np.nan},
])
def test_invalid_constructor_settings_are_rejected(free_model, kwargs):
    with pytest.raises(ValueError):
        K1GoalkeeperEnv(free_model, **kwargs)


def test_constructor_rejects_non_millisecond_physics(scene_paths):
    model = mujoco.MjModel.from_xml_path(str(scene_paths["free"]))
    model.opt.timestep = .002
    with pytest.raises(ValueError, match="0.001"):
        K1GoalkeeperEnv(model)


@pytest.mark.parametrize("options", [
    {"task": "unknown"}, {"unknown": 1}, {"shot": Shot()}, {"launch_time": 1},
    {"time_limit": 0}, {"time_limit": .0005}, {"time_limit": np.nan},
    {"task": "single_shot", "launch_time": .0005},
    {"task": "single_shot", "launch_time": -1},
    {"task": "single_shot", "launch_time": 1, "time_limit": 1},
    {"task": "single_shot", "shot": "invalid"},
])
def test_invalid_reset_options_fail_before_mutating_existing_episode(env, options):
    env.reset()
    env.step(env.default_action)
    before = env.get_evaluation_state()
    with pytest.raises(ValueError):
        env.reset(options=options)
    assert_tree_equal(env.get_evaluation_state(), before)


def test_baseline_reward_is_pure_and_contains_no_contact_bonus():
    transition = {"right_goal": False, "fell": False, "numerical_error": False,
                  "elapsed_dt": .02, "robot_contact": True}
    original = transition.copy()
    assert sum(baseline_reward(transition).values()) == 0
    assert transition == original
    assert sum(baseline_reward({**transition, "right_goal": True}).values()) == -1
    assert sum(baseline_reward({**transition, "fell": True}).values()) == -1


def test_failure_preserves_last_delivered_physics_state_separately_from_returned_actor(env, monkeypatch):
    initial, _ = env.reset()
    native_step, samples = mujoco.mj_step, []

    def fail_third_substep(model, data):
        native_step(model, data)
        samples.append({"time": float(data.time), "qpos": data.qpos.copy(),
                        "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy()})
        if len(samples) == 3:
            data.qvel[0] = np.nan

    monkeypatch.setattr(mujoco, "mj_step", fail_third_substep)
    observation, _, terminated, truncated, info = env.step(env.default_action)
    assert not terminated and truncated and len(samples) == 3
    np.testing.assert_array_equal(observation, initial)
    assert info["returned_observation_time"] == 0
    assert info["last_valid_physics_time"] == pytest.approx(.002)
    evaluation = env.get_evaluation_state()
    assert not evaluation["current_state_valid"]
    assert np.isnan(evaluation["qvel"][0])
    assert_tree_equal(evaluation["last_valid_state"], samples[1])
    for key in ("qpos", "qvel", "ctrl"):
        evaluation["last_valid_state"][key][:] = 456
    assert_tree_equal(env.get_evaluation_state()["last_valid_state"], samples[1])


def test_reset_failure_disables_episode_until_successful_reset(env, monkeypatch):
    env.reset()
    env.step(env.default_action)
    native_reset = env.backend.reset

    def fail_reset():
        native_reset()
        raise RuntimeError("injected reset failure")

    monkeypatch.setattr(env.backend, "reset", fail_reset)
    with pytest.raises(RuntimeError, match="injected reset failure"):
        env.reset()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.default_action)
    with pytest.raises(RuntimeError, match="reset"):
        env.get_privileged_state()
    monkeypatch.setattr(env.backend, "reset", native_reset)
    env.reset()
    assert not env.step(env.default_action)[2]


def test_reward_callback_receives_only_current_copied_channels(free_model):
    transitions = []

    def reward_callback(transition):
        transitions.append(deepcopy(transition))
        transition["actor_observation"][:] = 456
        for array in transition["privileged_state"].values():
            array[:] = 789
        return {"custom_current_reward": .25}

    env = K1GoalkeeperEnv(free_model, reward_fn=reward_callback)
    for shot, launch in [(Shot("future_a", velocity=(3, 0, 0)), 1.),
                         (Shot("future_b", velocity=(9, 2, 0)), 3.)]:
        env.reset(options={"task": "single_shot", "shot": shot, "launch_time": launch})
        observation, reward, _, _, info = env.step(env.default_action)
        assert reward == .25 and info["reward_terms"] == {"custom_current_reward": .25}
        np.testing.assert_array_equal(observation, transitions[-1]["actor_observation"])
        assert_tree_equal(env.get_privileged_state(), transitions[-1]["privileged_state"])
    assert_tree_equal(transitions[0], transitions[1])
    assert transitions[0]["transition_valid"] and transitions[0]["termination_reason"] is None
    forbidden = {"scheduled_shot", "scheduled_launch_time", "delivered_launch", "evaluation_state"}
    assert forbidden.isdisjoint(transitions[0])
    assert forbidden.isdisjoint(transitions[0]["privileged_state"])
    env.close()


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_reward_callback_failure_requires_reset(env, failure):
    def broken_reward(transition):
        if failure == "exception":
            raise RuntimeError("injected reward failure")
        return {"invalid": np.nan}

    env.reward_fn = broken_reward
    env.reset()
    with pytest.raises((RuntimeError, ValueError)):
        env.step(env.default_action)
    before = env.get_evaluation_state()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(env.default_action)
    assert_tree_equal(env.get_evaluation_state(), before)
    env.reward_fn = baseline_reward
    env.reset()
    assert env.step(env.default_action)[1] == 0
