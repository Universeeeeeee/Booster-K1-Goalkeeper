"""H1 integration checks independent of the report's target/response getters."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install requirements-walk.txt for H1")
if not (Path(__file__).resolve().parents[1] / "assets/booster_deploy/manifest.json").exists():
    pytest.skip("Run scripts/fetch_walk_policy.py for H1 assets", allow_module_level=True)

from k1_goalkeeper.defense_controller import DefensePulse, PostFilteredArmController
from k1_goalkeeper.defense_validation import contact_state
from k1_goalkeeper.scene_simulation import SceneSimulation


def runtime(model):
    torch.set_num_threads(1)
    return SceneSimulation(model, control_mode="k1_walk")


@pytest.mark.parametrize("offsets", [(), (("left_shoulder_roll_joint", 0.),)])
def test_zero_residual_matches_delivered_native_state_and_policy_history(free_model, offsets):
    reference, intervention = runtime(free_model), runtime(free_model)
    wrapper = PostFilteredArmController(intervention.controller, DefensePulse(offsets, start_s=0.))
    intervention.controller.compute_targets = wrapper.compute_targets
    for _ in range(120):
        reference.step()
        intervention.step()
        for name in ("qpos", "qvel", "qacc", "ctrl", "qfrc_actuator"):
            np.testing.assert_array_equal(getattr(reference.data, name), getattr(intervention.data, name))
        assert reference.data.time == intervention.data.time
        for name in ("obs_history", "last_action", "filtered_dof_target"):
            assert torch.equal(getattr(reference.controller.policy, name),
                               getattr(intervention.controller.policy, name))
    assert reference.physics_steps == intervention.physics_steps == 120
    assert reference.policy_steps == intervention.policy_steps == 6
    np.testing.assert_array_equal(intervention.data.qfrc_applied, 0)
    np.testing.assert_array_equal(intervention.data.xfrc_applied, 0)


def test_post_filter_offset_does_not_mutate_policy_state_or_leg_targets(free_model):
    reference, intervention = runtime(free_model), runtime(free_model)
    for _ in range(20):
        reference.step()
        intervention.step()
    pulse = DefensePulse((("left_shoulder_roll_joint", .15),), start_s=0., rise_s=.01)
    wrapper = PostFilteredArmController(intervention.controller, pulse)
    before_qpos = intervention.data.qpos.copy()
    before_time = intervention.data.time
    delivered = wrapper.compute_targets()
    nominal = reference.controller.compute_targets()
    index = intervention.controller.cfg.robot.joint_names.index("left_shoulder_roll_joint")
    others = [i for i in range(22) if i != index]
    torch.testing.assert_close(delivered[others], nominal[others], rtol=0, atol=0)
    assert float(delivered[index] - nominal[index]) == pytest.approx(.15, abs=1e-7)
    for name in ("obs_history", "last_action", "filtered_dof_target"):
        assert torch.equal(getattr(reference.controller.policy, name),
                           getattr(intervention.controller.policy, name))
    delivered[index] += 1
    assert torch.equal(reference.controller.policy.filtered_dof_target,
                       intervention.controller.policy.filtered_dof_target)
    np.testing.assert_array_equal(intervention.data.qpos, before_qpos)
    assert intervention.data.time == before_time


def test_nonzero_residual_reaches_real_motor_and_joint_without_external_forces(free_model):
    reference, intervention = runtime(free_model), runtime(free_model)
    pulse = DefensePulse((("left_shoulder_roll_joint", .05),), start_s=3., rise_s=2.)
    wrapper = PostFilteredArmController(intervention.controller, pulse)
    intervention.controller.compute_targets = wrapper.compute_targets
    name = "left_shoulder_roll_joint"
    q = int(free_model.joint(name).qposadr[0])
    motor = free_model.actuator(name).id
    differences = []
    for step in range(5200):
        reference.step()
        intervention.step()
        differences.append(intervention.data.ctrl[motor] - reference.data.ctrl[motor])
        if step < 3000:
            np.testing.assert_array_equal(reference.data.qpos, intervention.data.qpos)
            np.testing.assert_array_equal(reference.data.ctrl, intervention.data.ctrl)
    assert max(differences) > .001
    assert intervention.data.qpos[q] - reference.data.qpos[q] > .001
    np.testing.assert_array_equal(intervention.data.qfrc_applied, 0)
    np.testing.assert_array_equal(intervention.data.xfrc_applied, 0)
    assert intervention.data.time == reference.data.time == pytest.approx(5.2)


@pytest.mark.parametrize("offsets", [
    (("left_hip_pitch_joint", .15),),
    (("missing_arm_joint", .15),),
    (("left_shoulder_roll_joint", .15), ("left_shoulder_roll_joint", .15)),
    (("left_shoulder_roll_joint", .31),),
    (("left_shoulder_roll_joint", float("nan")),),
])
def test_invalid_residual_cannot_enter_original_controller(offsets):
    with pytest.raises(ValueError):
        DefensePulse(offsets)


def test_wrapper_reset_preserves_history_and_scene_reset_starts_new_episode(free_model, monkeypatch):
    sim = runtime(free_model)
    wrapper = PostFilteredArmController(sim.controller,
                                        DefensePulse((("left_shoulder_roll_joint", .15),)))
    sim.controller.compute_targets = wrapper.compute_targets
    sim.advance_policy_step()
    history = sim.controller.policy.obs_history.clone()
    action = sim.controller.policy.last_action.clone()
    targets = sim.controller.policy.filtered_dof_target.clone()
    before = sim.data.qpos.copy()
    reset_calls = []
    native_reset = sim.controller.policy.reset
    def reset():
        reset_calls.append(1)
        native_reset()
    monkeypatch.setattr(sim.controller.policy, "reset", reset)
    wrapper.reset()
    assert not reset_calls
    assert torch.equal(sim.controller.policy.obs_history, history)
    assert torch.equal(sim.controller.policy.last_action, action)
    assert torch.equal(sim.controller.policy.filtered_dof_target, targets)
    np.testing.assert_array_equal(sim.data.qpos, before)
    sim.reset()
    wrapper.reset()
    assert reset_calls == [1]
    assert sim.data.time == 0 and sim.physics_steps == sim.policy_steps == 0
    assert sim.controller.policy.obs_history is None
    sim.step()
    assert sim.data.time == pytest.approx(.001)
    assert reset_calls == [1]


def test_unexpected_native_reset_is_detected_before_new_controls(free_model):
    sim = runtime(free_model)
    wrapper = PostFilteredArmController(sim.controller, DefensePulse(()))
    sim.controller.compute_targets = wrapper.compute_targets
    sim.advance_policy_step()
    mujoco.mj_resetData(free_model, sim.data)
    before = sim.data.ctrl.copy()
    with pytest.raises(RuntimeError, match="MuJoCo"):
        sim.step()
    np.testing.assert_array_equal(sim.data.ctrl, before)
    assert sim.data.time == 0


def test_trial_rejects_unsafe_selected_target_before_pd_without_physics(free_model, monkeypatch):
    from k1_goalkeeper import defense_validation
    sim = runtime(free_model)
    # Synthetic policy tick only: no unsafe controls or native steps are run.
    sim.data.time = 5.
    nominal = sim.controller.robot.default_joint_pos.clone()
    index = sim.controller.cfg.robot.joint_names.index("left_shoulder_roll_joint")
    nominal[index] = float(free_model.joint("left_shoulder_roll_joint").range[1]) - .01
    native_wrapper = PostFilteredArmController
    def wrapper(controller, pulse):
        result = native_wrapper(controller, pulse)
        result.original_compute_targets = lambda: nominal.clone()
        return result
    monkeypatch.setattr(defense_validation, "PostFilteredArmController", wrapper)
    calls = []
    def pd(targets):
        calls.append(targets)
        pytest.fail("unsafe selected target reached original PD")
    monkeypatch.setattr(sim.controller, "apply_control", pd)
    def recorder(model, case, **kwargs):
        kwargs["command_provider"](sim)
        targets = sim.controller.compute_targets()
        sim.controller.apply_control(targets)
    monkeypatch.setattr(defense_validation, "run_impact", recorder)
    before = sim.data.ctrl.copy()
    with pytest.raises(RuntimeError, match="before PD"):
        defense_validation.run_defense_trial(
            free_model, DefensePulse((("left_shoulder_roll_joint", .05),)), "invalid_target")
    assert not calls and not sim.controller.is_running
    np.testing.assert_array_equal(sim.data.ctrl, before)
    assert sim.data.time == 5. and sim.physics_steps == sim.policy_steps == 0


def test_nonfinite_observer_fault_is_numerical_not_deliberate_guard(free_model, monkeypatch):
    from k1_goalkeeper import defense_validation
    sim = runtime(free_model)
    diagnostic = {"message": "H1 guard: nonfinite state or control", "last_valid_time": 0.}
    def recorder(model, case, **kwargs):
        kwargs["command_provider"](sim)
        sim.targets = sim.controller.compute_targets()
        # Fault-unit only: artificial time/torque values, never a physics step.
        sim.data.time = 3.001
        sim.data.qfrc_actuator[sim.controller.dof_addresses[2]] = np.nan
        with pytest.raises(RuntimeError, match="nonfinite state or control"):
            kwargs["step_observer"](sim)
        state = {
            "time": np.array([0.]),
            "body_linear_velocity": np.zeros((1, 3)),
            "body_angular_velocity": np.zeros((1, 3)),
            "base_position": np.array([[10.4, 0., .55369]]),
            "rpy": np.zeros((1, 3)),
        }
        summary = {"completed": False, "fall_events": [], "numerical_error": diagnostic,
                   "policy_stop": None, "joint_names": sim.controller.cfg.robot.joint_names}
        return summary, state
    monkeypatch.setattr(defense_validation, "run_impact", recorder)
    summary, _ = defense_validation.run_defense_trial(free_model, None, "fault_unit")
    assert summary["failure_class"] == "numerical"
    assert summary["numerical_error"] is diagnostic
    assert not summary["censored_by_deliberate_guard"]
    assert summary["recorder_guard_diagnostic"] is None
    assert not summary["passed"] and not summary["stable"]
    assert sim.physics_steps == sim.policy_steps == 0


def test_contact_guard_observes_real_self_contact_and_nonfoot_ground(free_model):
    sim = runtime(free_model)
    baseline = sim.data.qpos.copy()
    # A static, in-limit crossed-leg pose exercises native robot-robot contacts.
    for side, angle in (("left", -.2), ("right", .2)):
        sim.data.qpos[free_model.joint(f"{side}_hip_roll_joint").qposadr[0]] = angle
    mujoco.mj_forward(free_model, sim.data)
    state = contact_state(free_model, sim.data)
    assert state["self_contact_pairs"]
    assert any({free_model.body(int(free_model.geom_bodyid[g])).name for g in pair}
               == {"left_knee_pitch_link", "right_knee_pitch_link"}
               for pair in state["self_contact_pairs"])
    sim.data.qpos[:] = baseline
    sim.data.qpos[sim.controller.base_qpos_address + 2] = -.2
    mujoco.mj_forward(free_model, sim.data)
    state = contact_state(free_model, sim.data)
    assert state["nonfoot_ground_pairs"]
    assert any("left_knee_pitch_link" in {free_model.body(int(free_model.geom_bodyid[g])).name
                                          for g in pair} for pair in state["nonfoot_ground_pairs"])
    assert all(not any(free_model.body(int(free_model.geom_bodyid[g])).name
                      in ("left_ankle_roll_link", "right_ankle_roll_link") for g in pair)
               for pair in state["nonfoot_ground_pairs"])
    # Independent lower-level evidence: retained contacts carry solver forces.
    for pair in state["nonfoot_ground_pairs"]:
        forces = []
        for index, contact in enumerate(sim.data.contact):
            if tuple(sorted((int(contact.geom1), int(contact.geom2)))) == pair:
                force = np.zeros(6)
                mujoco.mj_contactForce(free_model, sim.data, index, force)
                forces.append(force[0])
        assert max(forces) > 1
