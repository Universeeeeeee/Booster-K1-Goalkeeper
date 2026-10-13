from pathlib import Path
import sys
from types import ModuleType
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install requirements-walk.txt for optional policy validation")
if not (Path(__file__).resolve().parents[1] / "assets/booster_deploy/manifest.json").exists():
    pytest.skip("Run scripts/fetch_walk_policy.py for optional policy assets", allow_module_level=True)

from k1_goalkeeper.scene import K1_XML
from k1_goalkeeper.walk_controller import K1WalkController


@pytest.fixture
def controller(official_model):
    result = K1WalkController(official_model, mujoco.MjData(official_model))
    result.initialize()
    return result


def modified_model(edit):
    root = ET.parse(K1_XML).getroot()
    root.find("compiler").set("meshdir", str(K1_XML.parent / "meshes"))
    edit(root)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def add_unrelated_actors(root):
    body = ET.Element("body", name="extra", pos="5 5 1")
    ET.SubElement(body, "joint", name="extra_joint", type="hinge")
    ET.SubElement(body, "geom", type="sphere", size=".1", mass="1")
    root.find("worldbody").insert(0, body)
    motor = ET.Element("motor", name="extra_motor", joint="extra_joint")
    root.find("actuator").insert(0, motor)
    ball = ET.SubElement(root.find("worldbody"), "body", name="ball", pos="6 6 1")
    ET.SubElement(ball, "freejoint", name="ball_free")
    ET.SubElement(ball, "geom", type="sphere", size=".1", mass="1")


def test_official_task_values_and_torchscript_inference(controller):
    cfg = controller.cfg
    assert cfg.policy_dt == .02
    assert cfg.policy.enable_safety_fallback
    assert cfg.policy.device == "cpu"
    assert Path(cfg.policy.checkpoint_path).is_absolute()
    assert cfg.policy.actor_obs_history_length == 10
    assert cfg.policy.history_init == "repeat"
    assert cfg.policy.use_actor_module
    assert cfg.policy.action_scale == .25
    assert cfg.policy.action_filter == .8
    assert cfg.policy.obs_dof_vel_scale == .1
    assert cfg.policy.clip_action == cfg.policy.clip_observation == 100
    assert cfg.policy.arm_action_fix_joint_name == "right_elbow_pitch_joint"
    assert cfg.policy.arm_action_fix_offset == .2
    assert cfg.robot.default_joint_pos == [
        0, 0, .2, -1.25, 0, -.5, .2, 1.25, 0, .5,
        -.15, 0, 0, .3, -.15, 0, -.15, 0, 0, .3, -.15, 0,
    ]
    assert cfg.robot.joint_stiffness == [
        4, 4, 20, 20, 20, 20, 20, 20, 20, 20,
        100, 100, 100, 100, 65, 65, 100, 100, 100, 100, 65, 65,
    ]
    assert cfg.robot.joint_damping == [
        1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1,
    ]
    assert cfg.robot.effort_limit == [
        6, 6, 14, 14, 14, 14, 14, 14, 14, 14,
        30, 20, 15, 35, 24, 15, 30, 20, 15, 35, 24, 15,
    ]
    assert cfg.policy.policy_joint_names == [
        "aaleft_shoulder_pitch_joint", "aaright_shoulder_pitch_joint",
        "left_hip_pitch_joint", "right_hip_pitch_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_hip_roll_joint", "right_hip_roll_joint",
        "left_elbow_pitch_joint", "right_elbow_pitch_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint",
        "left_elbow_yaw_joint", "right_elbow_yaw_joint",
        "left_knee_pitch_joint", "right_knee_pitch_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint",
    ]
    policy = controller.policy
    obs = policy.compute_observation()
    assert obs.shape == (69,)
    assert obs.dtype == torch.float32
    assert torch.equal(obs[:9], torch.tensor([0, 0, 0, 0, 0, -1, 0, 0, 0]))
    raw_action = policy._model(obs.repeat(10, 1).flatten()).reshape(-1)
    assert raw_action.shape == (20,)
    assert torch.isfinite(raw_action).all()
    expected = controller.robot.default_joint_pos.clone()
    for i, name in enumerate(cfg.policy.policy_joint_names):
        action = raw_action[i].clamp(-100, 100)
        if name == "right_elbow_pitch_joint":
            action = action - .2
        expected[cfg.robot.joint_names.index(name)] += action * .25 * .8
    targets = policy.inference()
    torch.testing.assert_close(targets, expected)
    torch.testing.assert_close(policy.obs_history, obs.repeat(10, 1))
    assert policy._model.model.original_name == "Sequential"


def test_named_state_addresses_ignore_ball_and_other_joint():
    model = modified_model(add_unrelated_actors)
    data = mujoco.MjData(model)
    extra = model.joint("extra_joint")
    ball = model.joint("ball_free")
    data.qpos[int(extra.qposadr[0])] = .75
    data.qpos[int(ball.qposadr[0]):int(ball.qposadr[0]) + 3] = [7, 8, 9]
    data.ctrl[model.actuator("extra_motor").id] = 2.3
    controller = K1WalkController(model, data)
    controller.initialize()
    assert controller.base_qpos_address == 1
    assert data.qpos[int(extra.qposadr[0])] == .75
    np.testing.assert_array_equal(
        data.qpos[int(ball.qposadr[0]):int(ball.qposadr[0]) + 3], [7, 8, 9]
    )
    q, v = controller.base_qpos_address, controller.base_dof_address
    np.testing.assert_allclose(data.qpos[q:q + 7], [0, 0, .6, 1, 0, 0, 0])
    data.qpos[controller.qpos_addresses] = np.linspace(-.2, .2, 22)
    data.qvel[controller.dof_addresses] = np.arange(22) * .1
    data.qfrc_actuator[controller.dof_addresses] = np.arange(22) + 10
    data.qvel[v:v + 6] = [1, 2, 3, 4, 5, 6]
    controller.update_state()
    np.testing.assert_allclose(controller.robot.data.joint_pos.numpy(), np.linspace(-.2, .2, 22))
    np.testing.assert_allclose(controller.robot.data.joint_vel.numpy(), np.arange(22) * .1)
    np.testing.assert_allclose(controller.robot.data.feedback_torque.numpy(), np.arange(22) + 10)
    np.testing.assert_allclose(controller.robot.data.root_ang_vel_b.numpy(), [4, 5, 6])
    np.testing.assert_allclose(controller.robot.data.root_lin_vel_b.numpy(), [1, 2, 3])
    controller.initialize()
    controller.step()
    assert data.ctrl[model.actuator("extra_motor").id] == 2.3
    assert model.jnt_type[model.joint("world_joint").id] == mujoco.mjtJoint.mjJNT_FREE


def test_reordered_motors_only_write_k1_controls():
    def edit(root):
        add_unrelated_actors(root)
        actuators = root.find("actuator")
        actuators[:] = list(reversed(list(actuators)))

    model = modified_model(edit)
    data = mujoco.MjData(model)
    controller = K1WalkController(model, data)
    extra_id = model.actuator("extra_motor").id
    data.ctrl[extra_id] = .123
    forces = data.qfrc_applied.copy()
    external = data.xfrc_applied.copy()
    controller.initialize()
    controller.step()
    assert data.ctrl[extra_id] == .123
    assert not np.array_equal(controller.actuator_ids, np.arange(22))
    np.testing.assert_array_equal(data.qfrc_applied, forces)
    np.testing.assert_array_equal(data.xfrc_applied, external)


def test_body_angular_velocity_matches_official_imu(controller):
    q, v = controller.base_qpos_address, controller.base_dof_address
    controller.data.qpos[q + 3:q + 7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    controller.data.qvel[v:v + 3] = [1, 0, 0]
    controller.data.qvel[v + 3:v + 6] = [.1, .2, .3]
    mujoco.mj_forward(controller.model, controller.data)
    controller.update_state()
    np.testing.assert_allclose(controller.robot.data.root_ang_vel_b.numpy(),
                               controller.data.sensor("angular-velocity").data, atol=1e-7)
    np.testing.assert_allclose(controller.robot.data.root_lin_vel_b.numpy(), [0, -1, 0], atol=1e-6)


def test_matches_unmodified_official_mujoco_controller(monkeypatch):
    # Only supply the upstream package's asset-directory constant, without installing its helper package.
    locator = ModuleType("booster_assets")
    locator.BOOSTER_ASSETS_DIR = str(K1_XML.parents[2])
    monkeypatch.setitem(sys.modules, "booster_assets", locator)
    from booster_deploy.controllers.mujoco_controller import MujocoController
    from tasks.locomotion.robots.k1 import K1WalkTaskCfg

    cfg = K1WalkTaskCfg()
    cfg.policy.checkpoint_path = str(Path(__file__).resolve().parents[1]
                                   / "assets/booster_deploy/tasks/locomotion/robots/k1/models/k1_walk.pt")
    reference = MujocoController(cfg)
    # There is no human velocity input during the zero-command comparison.
    monkeypatch.setattr(reference, "update_vel_command", lambda: None)
    reference.update_state()
    reference.start()
    model = mujoco.MjModel.from_xml_path(str(K1_XML))
    model.opt.timestep = cfg.mujoco.physics_dt
    named = K1WalkController(model, mujoco.MjData(model))
    named.initialize()
    for _ in range(20):
        reference.update_state()
        target = reference.policy_step()
        reference.ctrl_step(target)
        np.testing.assert_array_equal(named.step(), target.numpy())
        for attribute in ["qpos", "qvel", "ctrl"]:
            np.testing.assert_array_equal(getattr(named.data, attribute), getattr(reference.mj_data, attribute))
        torch.testing.assert_close(named.policy.obs_history, reference.policy.obs_history, rtol=0, atol=0)


@pytest.mark.parametrize("timestep,decimation", [(.001, 20), (.002, 10)])
def test_step_duration_and_per_substep_pd(timestep, decimation, monkeypatch):
    model = mujoco.MjModel.from_xml_path(str(K1_XML))
    model.opt.timestep = timestep
    controller = K1WalkController(model, mujoco.MjData(model))
    controller.initialize()
    step = mujoco.mj_step
    applied = []

    def inspect_step(model, data):
        targets = controller.policy.filtered_dof_target.numpy()
        position = data.qpos.astype(np.float32)[controller.qpos_addresses]
        velocity = data.qvel.astype(np.float32)[controller.dof_addresses]
        limits = controller.robot.effort_limit.numpy()
        expected = np.clip(
            controller.robot.joint_stiffness.numpy() * (targets - position)
            - controller.robot.joint_damping.numpy() * velocity, -limits, limits,
        )
        np.testing.assert_array_equal(data.ctrl[controller.actuator_ids], expected)
        applied.append(data.ctrl[controller.actuator_ids].copy())
        step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", inspect_step)
    assert controller.step().shape == (22,)
    assert controller.decimation == len(applied) == decimation
    assert controller.data.time == pytest.approx(.02)
    assert not np.array_equal(applied[0], applied[-1])


def test_falling_stops_before_motor_output(controller):
    q = controller.base_qpos_address
    controller.data.qpos[q + 3:q + 7] = [0, 1, 0, 0]
    mujoco.mj_forward(controller.model, controller.data)
    before = controller.data.ctrl.copy()
    with pytest.raises(RuntimeError, match="safety fallback"):
        controller.step()
    assert not controller.is_running
    np.testing.assert_array_equal(controller.data.ctrl, before)
    assert controller.data.time == 0
    with pytest.raises(RuntimeError, match="stopped"):
        controller.step()


def test_nonfinite_policy_targets_stop_before_motor_output(controller):
    controller.policy._model = lambda obs: torch.full((20,), float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite joint targets"):
        controller.step()
    assert not controller.is_running
    assert controller.data.time == 0


@pytest.mark.parametrize("failure", ["state", "warning", "reset"])
def test_simulation_failure_stops(controller, failure):
    controller.step()
    if failure == "state":
        controller.data.qvel[0] = np.nan
    elif failure == "warning":
        controller.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number += 1
    else:
        mujoco.mj_resetData(controller.model, controller.data)
    with pytest.raises(RuntimeError, match="MuJoCo"):
        controller.step()
    assert not controller.is_running


@pytest.mark.parametrize("failure,match", [
    ("missing", "missing K1 joint"),
    ("non_hinge", "must be a hinge"),
    ("wrong_transmission", "matching named motor"),
    ("duplicate_motor", "matching named motor"),
    ("gear", "unit-gear torque motor"),
    ("missing_base", "free base joint"),
])
def test_invalid_robot_mapping_fails(failure, match):
    def edit(root):
        joint = root.find(".//joint[@name='aahead_yaw_joint']")
        motor = root.find("actuator/motor[@name='aahead_yaw_joint']")
        if failure == "missing":
            joint.set("name", "renamed_head")
            motor.set("joint", "renamed_head")
        elif failure == "non_hinge":
            joint.set("type", "slide")
        elif failure == "wrong_transmission":
            motor.set("joint", "aahead_pitch_joint")
        elif failure == "duplicate_motor":
            ET.SubElement(root.find("actuator"), "motor", name="duplicate", joint="aahead_yaw_joint")
        elif failure == "gear":
            motor.set("gear", "2")
        else:
            root.find(".//joint[@name='world_joint']").set("name", "renamed_base")

    model = modified_model(edit)
    with pytest.raises(ValueError, match=match):
        K1WalkController(model, mujoco.MjData(model))


def test_nonintegral_physics_timestep_fails(official_model):
    model = mujoco.MjModel.from_xml_path(str(K1_XML))
    model.opt.timestep = .003
    with pytest.raises(ValueError, match="must divide"):
        K1WalkController(model, mujoco.MjData(model))
