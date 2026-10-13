import builtins
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from k1_goalkeeper.scene import build_scene, load_config
from k1_goalkeeper.scene_simulation import SceneSimulation


@pytest.fixture
def walk_runtime(free_model):
    torch = pytest.importorskip("torch", reason="Install requirements-walk.txt for scene walking")
    if not (Path(__file__).resolve().parents[1] / "assets/booster_deploy/manifest.json").exists():
        pytest.skip("Run scripts/fetch_walk_policy.py for optional policy assets")
    torch.set_num_threads(1)
    return SceneSimulation(free_model, control_mode="k1_walk")


@pytest.mark.parametrize("mode", ["free", "fixed", "absent"])
def test_zero_torque_matches_one_uncontrolled_scene_step(mode, request, monkeypatch):
    model = request.getfixturevalue(f"{mode}_model")
    real_import = builtins.__import__

    def deny_optional_policy(name, *args, **kwargs):
        if name == "torch" or name.endswith("walk_controller"):
            raise AssertionError("zero-torque mode imported the optional walking stack")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny_optional_policy)
    data = mujoco.MjData(model)
    runtime = SceneSimulation(model, data)
    reference = mujoco.MjData(model)
    mujoco.mj_forward(model, reference)
    assert runtime.model is model and runtime.data is data
    assert runtime.controller is None
    data.ctrl[:] = 1
    runtime.step()
    mujoco.mj_step(model, reference)
    np.testing.assert_array_equal(data.qpos, reference.qpos)
    np.testing.assert_array_equal(data.qvel, reference.qvel)
    np.testing.assert_array_equal(data.ctrl, np.zeros(model.nu))
    assert data.time == pytest.approx(.001)
    assert runtime.physics_steps == 1 and runtime.policy_steps == 0


def test_walk_owns_shared_named_scene_and_initial_stance(walk_runtime):
    runtime, controller = walk_runtime, walk_runtime.controller
    assert controller.model is runtime.model and controller.data is runtime.data
    assert runtime.model.nq == 36 and runtime.model.nv == 34
    assert runtime.model.joint("ball_free").type == mujoco.mjtJoint.mjJNT_FREE
    q = controller.base_qpos_address
    np.testing.assert_allclose(runtime.data.qpos[q:q + 7], [10.4, 0, .55369, 0, 0, 0, 1], atol=1e-6)
    np.testing.assert_array_equal(runtime.data.qpos[controller.qpos_addresses],
                                  controller.robot.default_joint_pos.numpy())
    np.testing.assert_array_equal(runtime.targets.numpy(), controller.robot.default_joint_pos.numpy())
    assert runtime.targets.shape == (22,)
    assert controller.policy.obs_history is None
    for side in ("left", "right"):
        body_id = runtime.model.body(f"{side}_ankle_roll_link").id
        feet = np.flatnonzero((runtime.model.geom_bodyid == body_id)
                             & (runtime.model.geom_type == mujoco.mjtGeom.mjGEOM_BOX))
        assert len(feet) == 1
        geom = feet[0]
        rotation = runtime.data.geom_xmat[geom].reshape(3, 3)
        bottom = runtime.data.geom_xpos[geom, 2] - np.abs(rotation[2]) @ runtime.model.geom_size[geom]
        assert -.00001 <= bottom <= .01
    np.testing.assert_array_equal(runtime.data.qfrc_applied, 0)
    np.testing.assert_array_equal(runtime.data.xfrc_applied, 0)


def test_custom_loaded_scene_pose_is_used_before_first_observation(tmp_path, walk_runtime):
    config = load_config()
    config["robot"]["position"] = [9.7, -.3, .55369]
    config["robot"]["yaw_degrees"] = 90
    path = build_scene(tmp_path / "custom_control.xml", config=config)
    model = mujoco.MjModel.from_xml_path(str(path))
    runtime = SceneSimulation(model, control_mode="k1_walk")
    np.testing.assert_allclose(runtime.data.body("trunk").xpos, config["robot"]["position"], atol=1e-6)
    np.testing.assert_allclose(runtime.data.body("trunk").xmat.reshape(3, 3)[:, 0], [0, 1, 0], atol=1e-6)
    assert runtime.controller.policy.obs_history is None
    runtime.step()
    assert runtime.controller.policy.obs_history.shape == (10, 69)


def test_ball_state_is_not_in_walking_observation(walk_runtime):
    runtime, controller = walk_runtime, walk_runtime.controller
    initial = controller.policy.compute_observation().clone()
    joint = runtime.model.joint("ball_free")
    q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
    runtime.data.qpos[q:q + 7] = [3, 4, 5, .5, .5, .5, .5]
    runtime.data.qvel[v:v + 6] = [1, 2, 3, 4, 5, 6]
    mujoco.mj_forward(runtime.model, runtime.data)
    controller.update_state()
    np.testing.assert_array_equal(controller.policy.compute_observation().numpy(), initial.numpy())
    ball_position, ball_velocity = runtime.data.qpos[q:q + 7].copy(), runtime.data.qvel[v:v + 6].copy()
    controller.initialize()
    np.testing.assert_array_equal(runtime.data.qpos[q:q + 7], ball_position)
    np.testing.assert_array_equal(runtime.data.qvel[v:v + 6], ball_velocity)


def test_single_physics_owner_and_twenty_pd_updates_per_policy(walk_runtime, monkeypatch):
    runtime, controller = walk_runtime, walk_runtime.controller
    native_step = mujoco.mj_step
    native_inference = controller.policy.inference
    controls, inference_times = [], []

    def inference():
        inference_times.append(float(runtime.data.time))
        return native_inference()

    def inspect_step(model, data):
        assert model is runtime.model and data is runtime.data
        np.testing.assert_array_equal(data.qfrc_applied, 0)
        np.testing.assert_array_equal(data.xfrc_applied, 0)
        position = data.qpos.astype(np.float32)[controller.qpos_addresses]
        velocity = data.qvel.astype(np.float32)[controller.dof_addresses]
        limits = controller.robot.effort_limit.numpy()
        expected = np.clip(controller.robot.joint_stiffness.numpy() * (runtime.targets.numpy() - position)
                           - controller.robot.joint_damping.numpy() * velocity, -limits, limits)
        np.testing.assert_array_equal(data.ctrl[controller.actuator_ids], expected)
        controls.append(data.ctrl[controller.actuator_ids].copy())
        native_step(model, data)

    monkeypatch.setattr(controller.policy, "inference", inference)
    monkeypatch.setattr(mujoco, "mj_step", inspect_step)
    runtime.step()
    assert runtime.data.time == pytest.approx(.001)
    assert len(controls) == len(inference_times) == 1
    first_targets = runtime.targets.clone()
    for _ in range(19):
        runtime.step()
        np.testing.assert_array_equal(runtime.targets.numpy(), first_targets.numpy())
    assert len(controls) == 20 and inference_times == [0]
    assert not np.array_equal(controls[0], controls[-1])
    assert runtime.data.time == pytest.approx(.02)
    assert runtime.physics_steps == 20 and runtime.policy_steps == 1
    runtime.advance_policy_step()
    assert len(controls) == 40 and inference_times == pytest.approx([0, .02])
    assert runtime.data.time == pytest.approx(.04)
    assert runtime.physics_steps == 40 and runtime.policy_steps == 2


def test_command_switch_retains_history_and_reset_starts_new_episode(walk_runtime, monkeypatch):
    runtime, controller = walk_runtime, walk_runtime.controller
    policy = controller.policy
    runtime.set_velocity((0, .2, 0))
    runtime.advance_policy_step()
    history, last_action = policy.obs_history.clone(), policy.last_action.clone()
    filtered = policy.filtered_dof_target.clone()
    native_reset = policy.reset
    reset_calls = []

    def count_reset():
        reset_calls.append(1)
        native_reset()

    monkeypatch.setattr(policy, "reset", count_reset)
    runtime.set_velocity((0, 0, 0))
    np.testing.assert_array_equal(policy.obs_history.numpy(), history.numpy())
    np.testing.assert_array_equal(policy.last_action.numpy(), last_action.numpy())
    np.testing.assert_array_equal(policy.filtered_dof_target.numpy(), filtered.numpy())
    assert not reset_calls
    runtime.advance_policy_step()
    runtime.set_velocity((.1, .1, .1))
    ball_q = int(runtime.model.joint("ball_free").qposadr[0])
    runtime.data.qpos[ball_q:ball_q + 3] = [3, 4, 5]
    runtime.reset()
    assert reset_calls == [1]
    assert runtime.physics_steps == runtime.policy_steps == 0
    assert runtime.data.time == 0
    assert controller._step_count == 0 and controller._elapsed_s == 0
    assert policy.obs_history is None
    np.testing.assert_array_equal(policy.last_action.numpy(), 0)
    np.testing.assert_array_equal(policy.filtered_dof_target.numpy(), controller.robot.default_joint_pos.numpy())
    np.testing.assert_array_equal(runtime.targets.numpy(), controller.robot.default_joint_pos.numpy())
    np.testing.assert_array_equal(runtime.data.qvel, 0)
    np.testing.assert_array_equal(runtime.data.qpos[ball_q:ball_q + 3], load_config()["ball"]["position"])
    assert (controller.vel_command.lin_vel_x, controller.vel_command.lin_vel_y,
            controller.vel_command.ang_vel_yaw) == (0, 0, 0)
    assert controller.is_running
    expected_observation = policy.compute_observation().numpy().copy()
    runtime.step()
    np.testing.assert_array_equal(policy.obs_history.numpy(), np.tile(expected_observation, (10, 1)))
    assert reset_calls == [1]


def test_scene_policy_period_matches_existing_controller(walk_runtime):
    runtime = walk_runtime
    reference = type(runtime.controller)(runtime.model, mujoco.MjData(runtime.model))
    q = reference.base_qpos_address
    reference.cfg.mujoco.init_pos = runtime.model.qpos0[q:q + 3].tolist()
    reference.cfg.mujoco.init_quat = runtime.model.qpos0[q + 3:q + 7].tolist()
    reference.initialize()
    for command in [(0, .2, 0)] * 3 + [(0, 0, 0)] * 2:
        runtime.set_velocity(command)
        reference.vel_command.lin_vel_x, reference.vel_command.lin_vel_y, reference.vel_command.ang_vel_yaw = command
        runtime.advance_policy_step()
        expected_targets = reference.step()
        np.testing.assert_array_equal(runtime.targets.numpy(), expected_targets)
        for name in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(getattr(runtime.data, name), getattr(reference.data, name))
        np.testing.assert_array_equal(runtime.controller.policy.obs_history.numpy(), reference.policy.obs_history.numpy())
    assert runtime.data.time == reference.data.time == pytest.approx(.1)


def test_scene_recorder_logs_physics_frames_counters_and_passive_ball(walk_runtime, tmp_path):
    from k1_goalkeeper.velocity_validation import Segment, run_case

    summary, trajectory = run_case("scene_frames", [Segment("move", .04, (0, .2, 0))],
                                   physics_dt=.001, simulation=walk_runtime, output=tmp_path)
    assert summary["completed"] and summary["error"] is None
    scene = summary["scene"]
    assert scene["control_mode"] == "k1_walk"
    assert scene["physics_steps"] == walk_runtime.physics_steps == 40
    assert scene["policy_steps"] == walk_runtime.policy_steps == 2
    assert scene["unexpected_contact_events"] == []
    assert scene["ball_actuator_force_final_maximum"] == 0
    assert scene["ball_maximum_horizontal_displacement_m"] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(trajectory["time"], np.arange(41) * .001, atol=1e-12)
    np.testing.assert_array_equal(trajectory["unexpected_contacts"], np.zeros((41, 3), dtype=bool))
    np.testing.assert_allclose(trajectory["ball_position"][:, :2], np.tile([7, 0], (41, 1)), atol=1e-12)
    np.testing.assert_allclose(scene["goal_origin_world"], [11, 0, 0])
    initial_rotation = np.diag([-1., -1., 1.])
    goal_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(scene["initial_body_rotation_world"], initial_rotation, atol=1e-12)
    np.testing.assert_allclose(scene["goal_rotation_world"], goal_rotation, atol=1e-12)
    rotations = np.empty((41, 3, 3))
    for rotation, quaternion in zip(rotations, trajectory["base_quaternion"]):
        mujoco.mju_quat2Mat(rotation.reshape(9), quaternion)
    np.testing.assert_allclose(trajectory["world_linear_velocity"],
                               np.einsum("nij,nj->ni", rotations, trajectory["body_linear_velocity"]), atol=1e-12)
    np.testing.assert_allclose(trajectory["world_angular_velocity"],
                               np.einsum("nij,nj->ni", rotations, trajectory["body_angular_velocity"]), atol=1e-12)
    np.testing.assert_allclose(trajectory["initial_body_position"],
                               (trajectory["base_position"] - trajectory["base_position"][0]) @ initial_rotation,
                               atol=1e-12)
    np.testing.assert_allclose(trajectory["initial_body_rotation"],
                               np.einsum("ij,njk->nik", initial_rotation.T, rotations), atol=1e-12)
    np.testing.assert_allclose(trajectory["goal_position"],
                               (trajectory["base_position"] - [11, 0, 0]) @ goal_rotation, atol=1e-12)
    np.testing.assert_allclose(trajectory["goal_rotation"],
                               np.einsum("ij,njk->nik", goal_rotation.T, rotations), atol=1e-12)
    assert json.loads((tmp_path / "summary.json").read_text()) == json.loads(json.dumps(summary))
    with np.load(tmp_path / "trajectory.npz") as saved:
        for name, values in trajectory.items():
            np.testing.assert_array_equal(saved[name], values)


def test_recorder_rejects_synthetic_ball_robot_contact_before_continuation(walk_runtime, monkeypatch):
    from k1_goalkeeper.velocity_validation import Segment, run_case

    native_step = mujoco.mj_step
    calls = []
    ball_id = walk_runtime.model.geom("ball_collision").id
    trunk_id = walk_runtime.model.body("trunk").id
    robot_id = int(np.flatnonzero((walk_runtime.model.geom_bodyid == trunk_id)
                                 & (walk_runtime.model.geom_contype != 0))[0])

    def add_synthetic_contact(model, data):
        native_step(model, data)
        calls.append(float(data.time))
        contact = mujoco.MjContact()
        contact.geom1, contact.geom2, contact.dist = ball_id, robot_id, -.001
        assert mujoco.mj_addContact(model, data, contact) == 0

    monkeypatch.setattr(mujoco, "mj_step", add_synthetic_contact)
    summary, trajectory = run_case("synthetic_contact", [Segment("zero", .04, (0, 0, 0))],
                                   physics_dt=.001, simulation=walk_runtime)
    assert calls == [.001]
    assert not summary["completed"] and not summary["mobility_usable"]
    assert "unexpected scene contact" in summary["error"]
    assert summary["simulated_duration_s"] == pytest.approx(.001)
    assert summary["scene"]["physics_steps"] == walk_runtime.physics_steps == 1
    assert summary["scene"]["policy_steps"] == walk_runtime.policy_steps == 1
    assert walk_runtime.data.time == pytest.approx(.001)
    np.testing.assert_array_equal(trajectory["unexpected_contacts"], [[False, False, False], [True, False, False]])
    events = summary["scene"]["unexpected_contact_events"]
    assert len(events) == 1 and events[0]["pair"] == ["ball", "robot"]
    assert events[0]["time"] == pytest.approx(.001)
    assert events[0]["geoms"] == ["ball_collision", walk_runtime.model.geom(robot_id).name]
    assert summary["scene"]["ball_actuator_force_final_maximum"] == 0
    np.testing.assert_array_equal(trajectory["ball_position"][:, :2], [[7, 0], [7, 0]])
    assert not walk_runtime.controller.is_running


def test_stopped_controller_cannot_advance_mid_policy(walk_runtime, monkeypatch):
    walk_runtime.step()
    walk_runtime.controller.stop()
    before = {name: getattr(walk_runtime.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
    before_time = walk_runtime.data.time

    def unexpected_step(model, data):
        raise AssertionError("stopped controller advanced physics")

    monkeypatch.setattr(mujoco, "mj_step", unexpected_step)
    with pytest.raises(RuntimeError, match="stopped"):
        walk_runtime.step()
    assert walk_runtime.data.time == before_time
    assert walk_runtime.physics_steps == walk_runtime.policy_steps == 1
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(walk_runtime.data, name), values)


def test_invalid_control_mode_rejected(free_model):
    with pytest.raises(ValueError, match="control_mode"):
        SceneSimulation(free_model, control_mode="walking")


@pytest.mark.parametrize("mode", ["fixed", "absent"])
def test_walking_requires_free_k1(mode, request):
    with pytest.raises(ValueError, match="free|K1"):
        SceneSimulation(request.getfixturevalue(f"{mode}_model"), control_mode="k1_walk")


@pytest.mark.parametrize("dt", [.002, .003, 0])
def test_scene_walking_requires_one_millisecond_timestep(scene_paths, dt):
    model = mujoco.MjModel.from_xml_path(str(scene_paths["free"]))
    model.opt.timestep = dt
    with pytest.raises(ValueError, match="timestep|0.001|1 ms"):
        SceneSimulation(model, control_mode="k1_walk")


def test_commands_require_finite_bounded_body_velocity(walk_runtime):
    controller = walk_runtime.controller
    bounds = controller.vel_command
    invalid = [(0, 0), (0, 0, 0, 0), (np.nan, 0, 0), (0, np.inf, 0),
               (bounds.vx_forward_max + .01, 0, 0), (-bounds.vx_backward_max - .01, 0, 0),
               (0, bounds.vy_max + .01, 0), (0, -bounds.vy_max - .01, 0),
               (0, 0, bounds.vyaw_max + .01), (0, 0, -bounds.vyaw_max - .01)]
    walk_runtime.set_velocity((.1, -.1, .1))
    for command in invalid:
        with pytest.raises(ValueError):
            walk_runtime.set_velocity(command)
        assert (bounds.lin_vel_x, bounds.lin_vel_y, bounds.ang_vel_yaw) == (.1, -.1, .1)


def test_positive_body_y_moves_toward_negative_world_y_and_recovers(walk_runtime):
    runtime, controller = walk_runtime, walk_runtime.controller
    q, v = controller.base_qpos_address, controller.base_dof_address
    origin = runtime.data.qpos[q:q + 3].copy()
    runtime.set_velocity((0, .2, 0))
    for _ in range(100):
        runtime.advance_policy_step()
        assert runtime.data.qpos[q + 2] > .45
    displacement = runtime.data.qpos[q:q + 3] - origin
    assert displacement[1] < -.15
    runtime.set_velocity((0, 0, 0))
    final_velocities = []
    for step in range(150):
        runtime.advance_policy_step()
        assert runtime.data.qpos[q + 2] > .45
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, runtime.data.qpos[q + 3:q + 7])
        assert rotation.reshape(3, 3)[2, 2] > np.cos(np.deg2rad(10))
        if step >= 100:
            final_velocities.append(rotation.reshape(3, 3).T @ runtime.data.qvel[v:v + 3])
    assert np.max(np.abs(np.mean(final_velocities, axis=0)[:2])) < .05
    assert runtime.data.time == pytest.approx(5)
    assert runtime.physics_steps == 5000 and runtime.policy_steps == 250
