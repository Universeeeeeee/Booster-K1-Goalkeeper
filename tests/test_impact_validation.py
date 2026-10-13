import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from k1_goalkeeper import impact_validation as validation


@pytest.fixture
def walking_assets():
    torch = pytest.importorskip("torch", reason="Install requirements-walk.txt for scene impact validation")
    if not (Path(__file__).resolve().parents[1] / "assets/booster_deploy/manifest.json").exists():
        pytest.skip("Run scripts/fetch_walk_policy.py for optional policy assets")
    torch.set_num_threads(1)


def test_impact_case_is_immutable():
    case = validation.ImpactCase("immutable")
    with pytest.raises(FrozenInstanceError):
        case.speed = 4


def test_contact_force_uses_transposed_frame_and_ball_geom_order():
    # MuJoCo contact axes are rows; a cyclic frame distinguishes R from R.T.
    frame = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
    contact = SimpleNamespace(frame=frame.ravel(), geom1=4, geom2=9)
    force = np.array([2., 3, 5, 100, 200, 300])
    np.testing.assert_array_equal(validation.ball_contact_force_world(contact, force, 9), [5, 2, 3])
    np.testing.assert_array_equal(validation.ball_contact_force_world(contact, force, 4), [-5, -2, -3])
    with pytest.raises(ValueError, match="ball"):
        validation.ball_contact_force_world(contact, force, 12)


@pytest.mark.parametrize("changes", [
    {"robot_mode": "unknown"}, {"speed": 0}, {"speed": -1}, {"speed": np.nan},
    {"offset_y": np.inf}, {"moving_duration": -.001}, {"duration": 0},
    {"duration": np.inf}, {"command": (0, 0)}, {"command": (0, np.nan, 0)},
    {"robot_mode": "fixed", "command": (0, .2, 0)},
    {"robot_mode": "absent", "moving_duration": .1}, {"target_geom_index": -1},
])
def test_invalid_impact_protocol_is_rejected(changes):
    with pytest.raises(ValueError):
        validation.ImpactCase("invalid", **changes)


def test_fixed_contact_impulses_match_delivered_ball_momentum_and_absent_control(
        fixed_model, absent_model, walking_assets):
    case = validation.ImpactCase("fixed_trunk", robot_mode="fixed", duration=.5)
    summary, trajectory = validation.run_impact(fixed_model, case)
    absent, absent_trajectory = validation.run_impact(
        absent_model, replace(case, name="absent_trunk", robot_mode="absent"), launch=summary["launch"])

    assert summary["completed"] and absent["completed"]
    assert summary["effective_contact"] and not absent["effective_contact"]
    assert summary["numerical_error"] is absent["numerical_error"] is None
    assert summary["balance_maintained"] is summary["balance_recovered"] is None
    assert summary["policy_steps"] == absent["policy_steps"] == 0
    assert summary["launch"]["initial_overlap"] is False
    assert summary["launch"]["target_body"] == "trunk"
    assert summary["launch"]["time"] == pytest.approx(2)
    assert summary["simulated_duration_s"] == pytest.approx(2.5)
    np.testing.assert_array_equal(summary["launch"]["position"], absent["launch"]["position"])
    np.testing.assert_array_equal(summary["launch"]["velocity"], absent["launch"]["velocity"])
    launch_row = np.searchsorted(trajectory["time"], summary["launch"]["time"])
    np.testing.assert_array_equal(trajectory["ball_position"][launch_row], summary["launch"]["position"])
    np.testing.assert_array_equal(trajectory["ball_velocity"][launch_row], summary["launch"]["velocity"])
    assert trajectory["ball_velocity"][-1, 0] < 0 < absent_trajectory["ball_velocity"][-1, 0]

    contacts = trajectory["contacts"]
    for episode in summary["contacts"]:
        selected = contacts[(contacts[:, 2] == episode["geom_id"])
                            & (contacts[:, 0] >= episode["start_time"] - 1e-12)
                            & (contacts[:, 1] <= episode["end_time"] + 1e-12)]
        assert len(selected) > 0
        dt = selected[:, 1] - selected[:, 0]
        assert episode["duration_s"] == pytest.approx(dt.sum())
        assert episode["normal_impulse_ns"] == pytest.approx(selected[:, 10] @ dt)
        np.testing.assert_allclose(episode["impulse_on_ball_world_ns"],
                                   (selected[:, 7:10] * dt[:, None]).sum(axis=0), atol=1e-12)
        assert episode["peak_force_n"] == pytest.approx(np.linalg.norm(selected[:, 7:10], axis=1).max())
    robot_episodes = [episode for episode in summary["contacts"] if episode["kind"] == "robot"]
    assert robot_episodes[0]["body"] == "trunk"
    assert robot_episodes[0]["geom_id"] == summary["launch"]["target_geom_id"]
    np.testing.assert_allclose(
        trajectory["robot_contact_force_world"].sum(axis=0) * fixed_model.opt.timestep,
        np.sum([episode["impulse_on_ball_world_ns"] for episode in robot_episodes], axis=0), atol=1e-12)

    ball_body = fixed_model.geom("ball_collision").bodyid[0]
    mass = fixed_model.body_mass[ball_body]
    # Cross-check contact-force direction against delivered dynamics, independently of the recorder helper.
    residual = mass * np.diff(trajectory["ball_velocity"][launch_row:], axis=0) - (
        trajectory["ball_contact_force_world"][launch_row + 1:] + mass * fixed_model.opt.gravity
    ) * fixed_model.opt.timestep
    assert np.abs(residual).max() < 1e-6
    np.testing.assert_allclose(trajectory["momentum_residual_ns"][launch_row + 1:], residual, atol=1e-12)


def test_low_impact_repeat_records_actual_physics_and_serialized_artifacts(free_model, walking_assets, tmp_path):
    case = validation.ImpactCase("repeat", duration=.5, seed=7)
    summary, trajectory = validation.run_impact(free_model, case, output=tmp_path)
    repeated_summary, repeated = validation.run_impact(free_model, case)

    assert summary["completed"] and summary["effective_contact"]
    assert not summary["fall_events"]
    assert summary["numerical_error"] is summary["policy_stop"] is None
    assert summary["balance_maintained"] is True
    assert summary["balance_recovered"] is False
    assert not summary["recovery"]["verified"]
    assert summary["physics_steps"] == 2500
    assert summary["policy_steps"] == 125
    np.testing.assert_allclose(trajectory["time"], np.arange(2501) * .001, atol=1e-12)
    assert np.abs(trajectory["momentum_residual_ns"]).max() < 1e-6
    assert not np.array_equal(trajectory["base_position"][0], trajectory["base_position"][-1])
    limits = np.asarray(summary["effort_limits_nm"])
    np.testing.assert_allclose(trajectory["motor_controls"],
                               np.clip(trajectory["raw_torque"], -limits, limits), atol=1e-6)
    np.testing.assert_array_equal(trajectory["applied_torque"], trajectory["motor_controls"])
    for key, array in trajectory.items():
        np.testing.assert_array_equal(array, repeated[key])
    assert summary["trajectory_sha256"] == repeated_summary["trajectory_sha256"]
    assert summary["contacts"] == repeated_summary["contacts"]
    assert json.loads((tmp_path / "summary.json").read_text()) == json.loads(json.dumps(summary))
    with np.load(tmp_path / "trajectory.npz") as saved:
        assert set(saved.files) == set(trajectory)
        for key, array in trajectory.items():
            np.testing.assert_array_equal(saved[key], array)


def test_miss_can_recover_without_claiming_effective_contact_or_contact_recovery(free_model, walking_assets):
    summary, trajectory = validation.run_impact(
        free_model, validation.ImpactCase("miss", miss=True, duration=3.2))

    assert summary["completed"]
    assert not summary["effective_contact"]
    assert summary["balance_maintained"] is True
    assert summary["recovery"]["verified"]
    assert summary["balance_recovered"] is False
    assert not summary["fall_events"]
    assert summary["numerical_error"] is summary["policy_stop"] is None
    assert all(contact["kind"] != "robot" for contact in summary["contacts"])
    np.testing.assert_array_equal(trajectory["robot_contact_force_world"], 0)


@pytest.mark.parametrize("failure", ["physical_fall", "numerical"])
def test_post_contact_failure_retains_contact_and_distinguishes_physical_fall_from_numerical_error(
        free_model, walking_assets, monkeypatch, failure):
    native_step = mujoco.mj_step
    steps = 0
    base_q = int(free_model.joint("world_joint").qposadr[0])
    ball_v = int(free_model.joint("ball_free").dofadr[0])

    def inject_failure(model, data):
        nonlocal steps
        native_step(model, data)
        steps += 1
        if steps == 2200:
            if failure == "physical_fall":
                data.qpos[base_q + 2] = .29
            else:
                data.qvel[ball_v] = np.nan

    monkeypatch.setattr(mujoco, "mj_step", inject_failure)
    summary, trajectory = validation.run_impact(free_model, validation.ImpactCase(failure, duration=.5))

    assert steps == 2200
    assert summary["effective_contact"]
    assert not summary["completed"]
    assert summary["balance_maintained"] is summary["balance_recovered"] is False
    assert not summary["recovery"]["verified"]
    assert summary["policy_stop"] is None
    assert summary["goal"] is None
    assert summary["exit_event"] is None
    assert all(np.isfinite(array).all() for array in trajectory.values())
    if failure == "physical_fall":
        assert len(summary["fall_events"]) == 1
        assert summary["fall_events"][0]["height_m"] == pytest.approx(.29)
        assert summary["numerical_error"] is None
        assert trajectory["time"][-1] == pytest.approx(2.2)
    else:
        assert summary["fall_events"] == []
        assert "nonfinite" in summary["numerical_error"]["message"]
        assert summary["numerical_error"]["last_valid_time"] == pytest.approx(2.199)
        assert trajectory["time"][-1] == pytest.approx(2.199)


def comparison_summary(name, goal=None, completed=True):
    return {"case": {"name": name},
            "launch": {"time": 2., "position": [10., 0, .6], "velocity": [20., 0, 0],
                       "angular_velocity": [0., 0, 0]},
            "completed": completed, "effective_contact": True,
            "goal": bool(goal) if completed else None,
            "exit_event": {"outcome": "Goal", "goal": goal} if goal else None,
            "maximum_momentum_residual_ns": 0., "numerical_error": None,
            "fall_events": [], "policy_stop": None,
            "balance_maintained": False, "balance_recovered": False}


def comparison_trace(final_x, final_vx):
    return {"time": np.array([2., 3.]),
            "ball_position": np.array([[10., 0, .6], [final_x, 0, .6]]),
            "ball_velocity": np.array([[20., 0, 0], [final_vx, 0, 0]])}


@pytest.mark.parametrize("goal,blocked", [("left", True), ("right", False)])
def test_block_comparison_uses_defended_right_goal_and_preserves_any_goal(walking_assets, goal, blocked):
    from scripts.validate_impacts import assess_experiment, compare

    summary = comparison_summary("fixed_front_20", goal=goal)
    absent = comparison_summary("absent_fixed_front_20", goal="right")
    absent["effective_contact"] = False
    assess_experiment(summary)
    assess_experiment(absent)
    result = compare(summary, comparison_trace(9., -20.), absent, comparison_trace(12., 20.))

    assert summary["goal"] is True
    assert summary["defended_goal"] is (goal == "right")
    assert result["control_goal"] is result["control_defended_goal"] is True
    assert result["launch_identical"] is result["trajectory_changed"] is True
    assert result["successfully_blocked"] is blocked


def test_partial_contact_comparison_does_not_claim_successful_block(walking_assets):
    from scripts.validate_impacts import assess_experiment, compare

    summary = comparison_summary("front_20", completed=False)
    summary["fall_events"] = [{"time": 3., "height_m": .29}]
    absent = comparison_summary("absent_front_20", goal="right")
    assess_experiment(summary)
    assess_experiment(absent)
    result = compare(summary, comparison_trace(9., -20.), absent, comparison_trace(12., 20.))

    assert summary["effective_contact"]
    assert summary["experiment_valid"]
    assert summary["goal"] is summary["defended_goal"] is None
    assert summary["balance_maintained"] is summary["balance_recovered"] is False
    assert result["control_defended_goal"] is True
    assert result["trajectory_changed"] is True
    assert result["successfully_blocked"] is None


@pytest.mark.parametrize("residual,passed", [(1e-6, True), (2e-5, False)])
def test_valid_finite_experiment_keeps_strict_momentum_precision_flag_separate(walking_assets, residual, passed):
    from scripts.validate_impacts import assess_experiment

    summary = comparison_summary("precision")
    summary["maximum_momentum_residual_ns"] = residual
    assess_experiment(summary)

    assert summary["numerical_error"] is None
    assert summary["experiment_valid"] is True
    assert summary["momentum_precision_passed"] is passed


def test_small_momentum_residual_does_not_make_numerically_failed_experiment_valid(walking_assets):
    from scripts.validate_impacts import assess_experiment

    summary = comparison_summary("numerical", completed=False)
    summary["numerical_error"] = {"message": "nonfinite state"}
    assess_experiment(summary)

    assert summary["momentum_precision_passed"] is True
    assert summary["experiment_valid"] is False
    assert summary["balance_maintained"] is summary["balance_recovered"] is False

