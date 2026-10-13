from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install requirements-walk.txt for optional policy validation")
if not (Path(__file__).resolve().parents[1] / "assets/booster_deploy/manifest.json").exists():
    pytest.skip("Run scripts/fetch_walk_policy.py for optional policy assets", allow_module_level=True)

from k1_goalkeeper import velocity_validation as validation


@pytest.mark.parametrize("duration", [0, -.02, np.nan, np.inf, .03])
def test_invalid_segment_duration_is_rejected(duration):
    with pytest.raises(ValueError):
        validation.run_case("invalid", [validation.Segment("invalid", duration, (0, 0, 0))])


@pytest.mark.parametrize("command", [(np.nan, 0, 0), (0, np.inf, 0), (100, 0, 0),
                                     (0, -100, 0), (0, 0, 100), (0, 0)])
def test_invalid_velocity_command_is_rejected(command):
    with pytest.raises(ValueError):
        validation.run_case("invalid", [validation.Segment("invalid", .02, command)])


def test_body_velocity_is_rotated_from_world_and_angular_velocity_is_already_body_frame(official_model):
    data = mujoco.MjData(official_model)
    joint = official_model.joint("world_joint")
    q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
    data.qpos[q:q + 3] = [4, 5, .6]
    data.qpos[q + 3:q + 7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    data.qvel[v:v + 6] = [1, 2, 3, .1, .2, .3]
    controller = SimpleNamespace(data=data, base_qpos_address=q, base_dof_address=v)

    state = validation.body_state(controller)

    np.testing.assert_allclose(state["base_position"], [4, 5, .6])
    np.testing.assert_allclose(state["body_linear_velocity"], [2, -1, 3], atol=1e-12)
    np.testing.assert_allclose(state["body_angular_velocity"], [.1, .2, .3], atol=1e-12)
    np.testing.assert_allclose(state["rpy"], [0, 0, np.pi / 2], atol=1e-12)


@pytest.mark.parametrize("physics_dt", [.001, .002])
def test_actual_dynamics_segments_physics_samples_and_repeatability(tmp_path, physics_dt):
    segments = [validation.Segment("forward", .04, (.2, 0, 0)),
                validation.Segment("left", .06, (0, .2, 0))]
    summary, trajectory = validation.run_case("repeat", segments, physics_dt=physics_dt, initial_yaw=np.pi / 2,
                                              seed=7, output=tmp_path)
    repeated_summary, repeated = validation.run_case("repeat", segments, physics_dt=physics_dt, initial_yaw=np.pi / 2,
                                                    seed=7)

    assert summary["completed"]
    assert summary["error"] is None
    assert summary["simulated_duration_s"] == pytest.approx(.1)
    n0, n1 = round(.04 / physics_dt), round(.06 / physics_dt)
    np.testing.assert_allclose(trajectory["time"], np.arange(n0 + n1 + 1) * physics_dt, atol=1e-12)
    np.testing.assert_array_equal(trajectory["segment_index"], [-1] + [0] * n0 + [1] * n1)
    np.testing.assert_allclose(trajectory["command"][1:n0 + 1], np.tile([.2, 0, 0], (n0, 1)))
    np.testing.assert_allclose(trajectory["command"][n0 + 1:], np.tile([0, .2, 0], (n1, 1)))
    np.testing.assert_allclose(trajectory["base_quaternion"][0],
                               [np.sqrt(.5), 0, 0, np.sqrt(.5)], atol=1e-12)
    assert trajectory["saturated"].shape == (n0 + n1 + 1, 22)
    assert trajectory["saturated"].dtype == np.bool_
    limits = np.asarray(summary["effort_limits_nm"])
    np.testing.assert_allclose(trajectory["motor_controls"],
                               np.clip(trajectory["raw_torque"], -limits, limits), atol=1e-6)
    assert not np.array_equal(trajectory["motor_controls"][1], trajectory["motor_controls"][2])
    assert not np.array_equal(trajectory["base_position"][0], trajectory["base_position"][-1])
    for key, array in trajectory.items():
        np.testing.assert_array_equal(array, repeated[key])
    assert summary["segments"] == repeated_summary["segments"]
    assert (tmp_path / "summary.json").exists()
    with np.load(tmp_path / "trajectory.npz") as saved:
        for key, array in trajectory.items():
            np.testing.assert_array_equal(saved[key], array)


@pytest.mark.parametrize("command,initial_yaw", [((.3, 0, 0), 0), ((-.3, 0, 0), 0),
                                               ((0, .2, 0), 0), ((0, -.2, 0), 0),
                                               ((0, 0, .3), 0), ((0, 0, -.3), 0),
                                               ((.3, 0, 0), np.pi / 2)])
def test_short_actual_dynamics_move_in_commanded_direction(command, initial_yaw):
    summary, trajectory = validation.run_case("direction", [validation.Segment("move", 1, command)],
                                              initial_yaw=initial_yaw)
    assert summary["completed"]
    measured = np.asarray(summary["segments"][0]["mean_body_velocity"])
    active = np.flatnonzero(command)[0]
    assert measured[active] * np.sign(command[active]) > .1 * abs(command[active])
    if active == 2:
        assert summary["segments"][0]["yaw_change_radians"] * np.sign(command[2]) > .05
    else:
        rotation = np.array([[np.cos(initial_yaw), -np.sin(initial_yaw)],
                             [np.sin(initial_yaw), np.cos(initial_yaw)]])
        world_command = rotation @ command[:2]
        displacement = np.asarray(summary["segments"][0]["steady_displacement_world"][:2])
        assert displacement @ world_command > .002
    assert np.isfinite(trajectory["base_position"]).all()


@pytest.mark.parametrize("failure", ["height", "tilt", "nonfinite"])
def test_failure_returns_partial_trajectory_and_does_not_report_completion(monkeypatch, failure):
    native_step = mujoco.mj_step
    calls = 0

    def fail_on_substep(model, data):
        nonlocal calls
        native_step(model, data)
        calls += 1
        if calls == 13:
            joint = model.joint("world_joint")
            q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
            if failure == "height":
                data.qpos[q + 2] = .29
            elif failure == "tilt":
                data.qpos[q + 3:q + 7] = [0, 1, 0, 0]
            else:
                data.qvel[v] = np.nan

    monkeypatch.setattr(mujoco, "mj_step", fail_on_substep)
    summary, trajectory = validation.run_case("failure", [validation.Segment("zero", .08, (0, 0, 0))])

    assert not summary["completed"]
    assert summary["error"]
    assert calls == 13
    last_valid_time = .024 if failure == "nonfinite" else .026
    assert summary["simulated_duration_s"] == pytest.approx(last_valid_time)
    assert trajectory["time"][-1] == pytest.approx(last_valid_time)
    assert len(trajectory["time"]) == (13 if failure == "nonfinite" else 14)
    assert not summary["segments"][0]["completed"]
    if failure == "nonfinite":
        assert "nonfinite" in summary["error"]
    else:
        assert summary["fall_events"]
        assert np.isfinite(trajectory["base_position"]).all()


def synthetic_trajectory(command, measured):
    time = np.arange(21) * .002
    measured = np.asarray(measured)
    position = np.column_stack([time * measured[0], time * measured[1], np.full(21, .6)])
    return {"time": time, "segment_index": np.array([-1] + [0] * 20),
            "command": np.tile(command, (21, 1)), "base_position": position,
            "base_quaternion": np.column_stack([np.cos(time*measured[2]/2), np.zeros(21),
                                                 np.zeros(21), np.sin(time*measured[2]/2)]),
            "rpy": np.column_stack([np.zeros(21), np.zeros(21), time * measured[2]]),
            "body_linear_velocity": np.tile([*measured[:2], 0], (21, 1)),
            "body_angular_velocity": np.tile([0, 0, measured[2]], (21, 1)),
            "raw_torque": np.zeros((21, 22)), "motor_controls": np.zeros((21, 22)),
            "saturated": np.zeros((21, 22), dtype=bool)}


def test_tracking_bias_rmse_and_saturation_measure_physics_steps():
    segment = validation.Segment("move", .04, (.3, .2, -.2))
    trajectory = synthetic_trajectory(segment.command, (.2, -.1, .3))
    # Large startup error is outside the steady window, but its torque sample must still count.
    trajectory["body_linear_velocity"][:11] = 20
    trajectory["body_angular_velocity"][:11] = 20
    trajectory["saturated"][7, 3] = True
    trajectory["raw_torque"][7, 3] = 50
    trajectory["motor_controls"][7, 3] = 35

    summary = validation.summarize_case("metrics", [segment], trajectory, .002)
    result = summary["segments"][0]

    assert result["physics_samples"] == 20
    assert result["steady_physics_samples"] == 10
    np.testing.assert_allclose(result["mean_body_velocity"], [.2, -.1, .3])
    np.testing.assert_allclose(result["mean_tracking_error"], [-.1, -.3, .5])
    np.testing.assert_allclose(result["tracking_rmse"], [.1, .3, .5])
    assert result["torque_saturation_fraction"][3] == pytest.approx(1 / 20)
    assert result["any_saturation_step_fraction"] == pytest.approx(1 / 20)
    assert result["steady_torque_saturation_fraction"][3] == 0
    assert result["peak_raw_torque_nm"][3] == 50
    assert result["peak_motor_controls_nm"][3] == 35
    np.testing.assert_allclose(result["displacement_world"], [.008, -.004, 0])
    np.testing.assert_allclose(result["steady_displacement_world"], [.004, -.002, 0])
    assert not result["response_present"]
    assert not result["tracks_command"]


@pytest.mark.parametrize("command", [(.3, 0, 0), (-.3, 0, 0), (0, .2, 0),
                                     (0, -.2, 0), (0, 0, .3), (0, 0, -.3)])
def test_direction_response_requires_correct_signed_body_velocity(command):
    segment = validation.Segment("direction", .04, command)
    matching = validation.summarize_case("direction", [segment],
                                        synthetic_trajectory(command, command), .002)
    opposite = validation.summarize_case("direction", [segment],
                                        synthetic_trajectory(command, -np.asarray(command)), .002)
    assert matching["segments"][0]["response_present"]
    assert matching["segments"][0]["tracks_command"]
    assert not opposite["segments"][0]["response_present"]
    assert not opposite["segments"][0]["tracks_command"]


def test_partial_metrics_do_not_claim_unrun_segment_or_steady_tracking():
    segments = [validation.Segment("move", .04, (.3, 0, 0)),
                validation.Segment("stop", .02, (0, 0, 0))]
    trajectory = {key: value[:9] for key, value in synthetic_trajectory((.3, 0, 0), (.3, 0, 0)).items()}

    summary = validation.summarize_case("partial", segments, trajectory, .002, error="stopped")

    assert summary["requested_duration_s"] == pytest.approx(.06)
    assert summary["simulated_duration_s"] == pytest.approx(.016)
    assert not summary["completed"]
    assert not summary["mobility_usable"]
    assert not summary["zero_recovery"]["verified"]
    first, second = summary["segments"]
    assert first["physics_samples"] == 8
    assert first["steady_physics_samples"] == 0
    assert not first["completed"]
    assert not first["response_present"]
    assert not first["tracks_command"]
    assert second["physics_samples"] == 0
    assert not second["completed"]
    assert not second["tracks_command"]


def test_recovery_requires_sustained_stop_not_cancelling_average():
    time = np.arange(1601) * .002
    oscillation = np.where((np.floor((time-1e-10)/.2).astype(int) % 2) == 0, .2, -.2)
    velocity = np.column_stack([oscillation, np.zeros(len(time)), np.zeros(len(time))])
    trajectory = {"time": time, "base_position": np.column_stack([
        np.cumsum(oscillation)*.002, np.zeros(len(time)), np.full(len(time), .6)]),
        "rpy": np.zeros((len(time), 3))}
    result = validation.recovery_metrics(trajectory, velocity, 0, 3.2, True)
    np.testing.assert_allclose(result["mean_body_velocity"], [0, 0, 0], atol=1e-12)
    assert result["stop_delay_s"] is None
    assert not result["verified"]


def test_intermediate_zero_command_must_also_recover():
    segments = [validation.Segment("move", .04, (.3, 0, 0)),
                validation.Segment("bad_stop", 3.2, (0, 0, 0)),
                validation.Segment("move_again", .04, (.3, 0, 0)),
                validation.Segment("recover", 3.2, (0, 0, 0))]
    time = np.arange(3241) * .002
    idx = np.searchsorted([.04, 3.24, 3.28, 6.48], time-1e-10)
    idx[0] = -1
    velocity = np.zeros((len(time), 3))
    velocity[np.isin(idx, [0, 2]), 0] = .3
    oscillation = np.where(np.floor((time-.04-1e-10)/.2).astype(int) % 2 == 0, .2, -.2)
    velocity[idx == 1, 0] = oscillation[idx == 1]
    trajectory = {"time": time, "segment_index": idx,
                  "command": np.array([segments[i].command if i >= 0 else (0, 0, 0) for i in idx]),
                  "base_position": np.column_stack([np.cumsum(velocity[:, 0])*.002,
                                                     np.zeros(len(time)), np.full(len(time), .6)]),
                  "base_quaternion": np.tile([1., 0, 0, 0], (len(time), 1)),
                  "body_linear_velocity": velocity, "body_angular_velocity": np.zeros_like(velocity),
                  "rpy": np.zeros_like(velocity), "raw_torque": np.zeros((len(time), 22)),
                  "motor_controls": np.zeros((len(time), 22)), "saturated": np.zeros((len(time), 22), bool)}
    result = validation.summarize_case("intermediate", segments, trajectory, .002)
    assert result["completed"]
    assert result["zero_recovery"]["verified"]
    assert not result["zero_recoveries"][0]["verified"]
    assert not result["mobility_usable"]
