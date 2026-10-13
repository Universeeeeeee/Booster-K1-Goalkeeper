"""Stage F checks of decisions and delivered scene trajectories."""
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from k1_goalkeeper.impact_validation import ImpactCase, run_impact
from k1_goalkeeper.positioning import (
    LateralPositionController,
    ResponseLimits,
    predict_ball,
    reachable_target,
    world_to_goal,
)
from k1_goalkeeper.positioning_validation import TruthGoalkeeper, prediction_errors, run_trial
from k1_goalkeeper import positioning_validation


@pytest.fixture
def walking_assets():
    torch = pytest.importorskip("torch", reason="Install requirements-walk.txt for positioning dynamics")
    if not (Path(__file__).resolve().parents[1] / "assets/booster_deploy/manifest.json").exists():
        pytest.skip("Run scripts/fetch_walk_policy.py for optional policy assets")
    torch.set_num_threads(1)


PARKED_BALL = {"position": [0., 3., .11], "velocity": [0., 0., 0.],
               "angular_velocity": [0., 0., 0.]}


def test_world_goal_coordinates_preserve_height_and_measure_distance_in_front_of_goal():
    np.testing.assert_allclose(world_to_goal([10.4, -.3, .62]), [-.3, .6, .62])
    np.testing.assert_allclose(world_to_goal([11., .3, .11]), [.3, 0., .11])


@pytest.mark.parametrize("yaw", [np.pi, np.pi-.03, -np.pi+.03])
@pytest.mark.parametrize("target", [-.3, .3])
def test_lateral_command_has_correct_world_sign_when_facing_field(yaw, target):
    controller = LateralPositionController()
    command, tracking = controller.command(2., [10.4, 0., .6], yaw, [0., 0., 0.], target)
    assert command[0] == command[2] == 0.
    assert command[1] * np.cos(yaw) * target > 0.
    assert tracking["safe"]
    assert tracking["target_m"] == target


@pytest.mark.parametrize("target", [-.10, -.01, 0., .01, .10])
def test_position_deadzone_does_not_start_walking(target):
    controller = LateralPositionController()
    command, tracking = controller.command(2., [10.4, 0., .6], np.pi, [0., 0., 0.], target)
    assert command == [0., 0., 0.]
    assert tracking["direction"] == 0


def test_predictive_stop_then_hysteresis_prevents_immediate_restart():
    controller = LateralPositionController()
    command, _ = controller.command(0., [10.4, 0., .6], np.pi, [0., 0., 0.], .2)
    assert command[1] < 0
    command, tracking = controller.command(.5, [10.4, .12, .6], np.pi, [0., .3, 0.], .2)
    assert tracking["predicted_stop_m"] > .2
    assert command == [0., 0., 0.]
    # A disturbance outside the start deadzone must still respect the stop dwell.
    command, _ = controller.command(.6, [10.4, .08, .6], np.pi, [0., 0., 0.], .2)
    assert command == [0., 0., 0.]
    command, _ = controller.command(1.8, [10.4, .08, .6], np.pi, [0., 0., 0.], .2)
    assert command[1] < 0


@pytest.mark.parametrize("target", [-2., 2.])
def test_requested_position_is_clipped_to_safe_post_clearance(target):
    controller = LateralPositionController()
    command, tracking = controller.command(2., [10.4, 0., .6], np.pi, [0., 0., 0.], target)
    assert tracking["requested_target_m"] == target
    assert tracking["target_m"] == np.sign(target)*.65
    assert command[1]*target < 0


@pytest.mark.parametrize("position", [[10., 0., .6], [10.8, 0., .6], [10.4, .7, .6]])
def test_outside_safe_goalkeeper_strip_stops_lateral_motion(position):
    command, tracking = LateralPositionController().command(2., position, np.pi, [0., 0., 0.], .4)
    assert command == [0., 0., 0.]
    assert not tracking["safe"]


def test_heading_alignment_has_wraparound_and_hysteresis():
    controller = LateralPositionController()
    command, _ = controller.command(0., [10.4, 0., .6], np.pi-.2, [0., 0., 0.], .3)
    assert command == [0., 0., .2]
    command, _ = controller.command(.02, [10.4, 0., .6], np.pi-.08, [0., 0., 0.], .3)
    assert command == [0., 0., .2]
    command, _ = controller.command(.04, [10.4, 0., .6], -np.pi+.02, [0., 0., 0.], .3)
    assert command == [0., 0., 0.]
    assert not controller.turning


def test_rewinding_time_requires_reset_and_reset_repeats_decisions():
    controller = LateralPositionController()
    inputs = [(2., [10.4, 0., .6], np.pi, [0., 0., 0.], .2),
              (2.5, [10.4, .12, .6], np.pi, [0., .3, 0.], .2)]
    first = [controller.command(*args) for args in inputs]
    with pytest.raises(ValueError, match="reset"):
        controller.command(*inputs[0])
    controller.reset()
    assert [controller.command(*args) for args in inputs] == first


@pytest.mark.parametrize("field", ["time", "position", "yaw", "velocity", "target"])
def test_nonfinite_control_state_is_rejected(field):
    values = {"time": 2., "position": [10.4, 0., .6], "yaw": np.pi,
              "velocity": [0., 0., 0.], "target": .2}
    values[field] = [np.nan, 0., 0.] if field in ("position", "velocity") else np.nan
    with pytest.raises(ValueError, match="finite"):
        LateralPositionController().command(*values.values())


def test_ballistic_prediction_uses_only_current_state_and_gravity():
    prediction = predict_ball([9., -.2, 1.], [10., 1., 1.], [0., 3., 4.])
    assert prediction["valid"]
    assert prediction["time_to_plane_s"] == pytest.approx(.2)
    assert prediction["lateral_m"] == pytest.approx(0.)
    assert prediction["height_m"] == pytest.approx(1.+.2-.5*9.81*.2**2)
    assert prediction["model"] == "gravity_ballistic"
    assert not prediction["future_floor_contact_unmodeled"]
    assert prediction["inside_aperture"] is True
    assert prediction["spin_norm_rad_s"] == 5.
    # Spin is observed and disclosed; it must not invent a Magnus trajectory.
    unspun = predict_ball([9., -.2, 1.], [10., 1., 1.])
    for key in ("time_to_plane_s", "lateral_m", "height_m"):
        assert prediction[key] == unspun[key]


def test_ground_prediction_reports_constant_speed_assumption_and_whole_ball_plane():
    state = ([9., .1, .11], [2., .2, 0.])
    center = predict_ball(*state)
    whole_ball = predict_ball(*state, plane_x=11.11)
    assert center["model"] == "constant_speed_ground"
    assert center["time_to_plane_s"] == 1.
    assert center["lateral_m"] == pytest.approx(.3)
    assert center["height_m"] == .11
    assert center["inside_aperture"] is True
    assert whole_ball["time_to_plane_s"] > center["time_to_plane_s"]
    assert whole_ball["lateral_m"] > center["lateral_m"]
    assert "deceleration" in center["limitations"]


def test_unmodeled_future_bounce_abstains_from_aperture_claim():
    prediction = predict_ball([8., 0., .5], [1., 0., 0.])
    assert prediction["valid"]
    assert prediction["future_floor_contact_unmodeled"]
    assert prediction["inside_aperture"] is None
    assert "bounce" in prediction["limitations"]


@pytest.mark.parametrize("position,velocity", [([9., 0., .5], [-2., 0., 0.]),
                                               ([9., 0., .5], [0., 0., 0.]),
                                               ([11.1, 0., .5], [2., 0., 0.])])
def test_not_approaching_ball_has_no_interception_prediction(position, velocity):
    prediction = predict_ball(position, velocity)
    assert prediction["valid"] is False
    assert "time_to_plane_s" not in prediction


@pytest.mark.parametrize("kwargs", [{"position": [1., 2.]}, {"velocity": [np.nan, 0., 0.]},
                                    {"angular_velocity": [0., np.inf, 0.]}, {"plane_x": np.nan}])
def test_invalid_ball_state_is_rejected(kwargs):
    state = {"position": [9., 0., .5], "velocity": [2., 0., 0.]}
    state.update(kwargs)
    with pytest.raises(ValueError, match="finite"):
        predict_ball(**state)


def test_reachability_reserves_startup_and_braking_and_clips_to_safe_range():
    limits = ResponseLimits()
    ample = reachable_target(.2, 0., 0., 6., limits)
    assert ample["reachable"]
    assert ample["decision_lateral_m"] == .2
    assert ample["travel_budget_m"] < limits.speed_m_s*6.
    urgent = reachable_target(.2, 0., 0., 1., limits)
    assert not urgent["reachable"]
    assert urgent["decision_lateral_m"] == 0.
    near_post = reachable_target(1., .6, 0., 20., limits)
    assert not near_post["reachable"]
    assert near_post["decision_lateral_m"] == .65
    assert all(-.65 <= value <= .65 for value in near_post["interval_m"])


def test_reversal_does_not_credit_instantaneous_velocity_as_extra_reach():
    still = reachable_target(-.25, 0., 0., 5.)
    reversing = reachable_target(-.25, 0., .2, 5.)
    assert still["reachable"]
    assert not reversing["reachable"]
    assert reversing["travel_budget_m"] < still["travel_budget_m"]
    assert reversing["decision_lateral_m"] > -.25
    outside = reachable_target(0., .8, 0., 1.)
    assert not outside["reachable"]
    assert outside["reason"] == "outside safe range"
    assert outside["decision_lateral_m"] == .65


@pytest.mark.parametrize("arguments", [(np.nan, 0., 0., 1.), (0., np.inf, 0., 1.),
                                        (0., 0., np.nan, 1.), (0., 0., 0., -1.),
                                        (0., 0., 0., np.inf)])
def test_invalid_reachability_input_is_rejected(arguments):
    with pytest.raises(ValueError):
        reachable_target(*arguments)


@pytest.mark.parametrize("time_left,velocity,can_stop", [(.5, .2, False), (1., .2, True),
                                                       (.5, .04, True)])
def test_already_at_target_is_not_reachable_without_time_to_brake(time_left, velocity, can_stop):
    decision = reachable_target(0., 0., velocity, time_left)
    assert decision["decision_lateral_m"] == 0.
    assert decision["braking_time_available"] is can_stop
    assert decision["reachable"] is can_stop


def test_negligible_position_error_is_reachable_without_new_lateral_travel():
    decision = reachable_target(0., .000864, 0., .7)
    assert decision["travel_budget_m"] == 0.
    assert decision["reachable"]
    assert abs(decision["decision_lateral_m"]) < .10


@pytest.mark.parametrize("sign", [-1., 1.])
def test_position_tolerance_cannot_make_target_outside_safe_range_reachable(sign):
    decision = reachable_target(sign*.66, sign*.65, 0., .7)
    assert not decision["reachable"]
    assert decision["decision_lateral_m"] == sign*.65


def observed_sim(model, position=(8., .3, .11), velocity=(.2, 0., 0.)):
    data = mujoco.MjData(model)
    base, ball = model.joint("world_joint"), model.joint("ball_free")
    q, v = int(base.qposadr[0]), int(base.dofadr[0])
    bq, bv = int(ball.qposadr[0]), int(ball.dofadr[0])
    data.qpos[q:q+7] = [10.4, 0., .6, 0., 0., 0., 1.]
    data.qpos[bq:bq+3], data.qvel[bv:bv+3] = position, velocity
    data.time = 2.
    controller = SimpleNamespace(data=data, base_qpos_address=q, base_dof_address=v)
    return SimpleNamespace(model=model, data=data, controller=controller, physics_steps=2000)


@pytest.mark.parametrize("base_x,ball_x", [(10.2, 8.), (10.4, 8.), (10.4, 9.7)])
def test_interception_deadline_precedes_all_articulation_collision_bound(free_model, base_x, ball_x):
    sim = observed_sim(free_model, position=(ball_x, 0., .11), velocity=(2., 0., 0.))
    sim.data.qpos[sim.controller.base_qpos_address] = base_x
    adapter = TruthGoalkeeper("active")
    command = adapter.command(sim)
    prediction = adapter.records[-1]["interception_prediction"]
    assert prediction["plane_x"] == pytest.approx(base_x-.81)
    assert prediction["plane_x"] <= base_x-(.693084+.11)
    if ball_x < prediction["plane_x"]:
        assert prediction["time_to_plane_s"] == pytest.approx((base_x-.81-ball_x)/2.)
        assert prediction["time_to_plane_s"] < adapter.records[-1]["prediction"]["time_to_plane_s"]
    else:
        assert not prediction["valid"]
        assert adapter.records[-1]["reachability"] is None
        assert command == [0., 0., 0.]


@pytest.mark.parametrize("position,velocity", [((8., .3, .5), (.2, 0., 0.)),
                                               ((8., 2., .11), (.2, 0., 0.)),
                                               ((8., .3, .11), (-.2, 0., 0.))])
def test_active_control_stops_when_forecast_becomes_unusable(free_model, position, velocity):
    adapter = TruthGoalkeeper("active")
    sim = observed_sim(free_model)
    assert adapter.command(sim)[1] < 0.
    ball = free_model.joint("ball_free")
    q, v = int(ball.qposadr[0]), int(ball.dofadr[0])
    sim.data.qpos[q:q+3], sim.data.qvel[v:v+3] = position, velocity
    sim.data.time, sim.physics_steps = 2.02, 2020
    assert adapter.command(sim) == [0., 0., 0.]
    row = adapter.records[-1]
    assert row["reachability"] is None
    assert row["tracking"] is None
    assert row["planning_reason"].startswith("abstain")


@pytest.mark.parametrize("ball_first", [True, False])
def test_contact_latch_requires_delivered_positive_normal_force(free_model, monkeypatch, ball_first):
    adapter = TruthGoalkeeper("active")
    sim = observed_sim(free_model)
    ball = free_model.geom("ball_collision").id
    robot = next(g for g in range(free_model.ngeom)
                 if free_model.geom_bodyid[g] == free_model.body("trunk").id)
    pair = (ball, robot) if ball_first else (robot, ball)
    contact = SimpleNamespace(geom1=pair[0], geom2=pair[1], efc_address=0, dist=-.001)
    # An overlapping, active constraint can have no force; overlap alone is not a save.
    sim.data = SimpleNamespace(qpos=sim.data.qpos, time=2.01, ncon=1, contact=[contact])
    normal_force = 0.

    def contact_force(model, data, index, force):
        force[0] = normal_force

    monkeypatch.setattr(mujoco, "mj_contactForce", contact_force)
    adapter.observe(sim)
    assert not adapter.contact_seen
    normal_force = 2.
    adapter.observe(sim)
    assert adapter.contact_seen
    normal_force = 0.
    adapter.observe(sim)
    assert adapter.contact_seen


@pytest.mark.parametrize("violation", [None, "height", "tilt", "lateral", "front", "back"])
def test_positioning_gate_rejects_intermediate_safety_violation_despite_final_recovery(monkeypatch, violation):
    time = np.arange(31)*.2
    position = np.tile([10.4, 0., .6], (len(time), 1))
    rpy = np.zeros((len(time), 3))
    if violation == "height":
        position[11, 2] = .44
    elif violation == "tilt":
        rpy[11, 0] = np.deg2rad(11.)
    elif violation == "lateral":
        position[11, 1] = .7
    elif violation in ("front", "back"):
        position[11, 0] = 10. if violation == "front" else 10.8
    trace = {"time": time, "command": np.zeros((len(time), 3)), "base_position": position,
             "body_linear_velocity": np.zeros((len(time), 3)),
             "body_angular_velocity": np.zeros((len(time), 3)), "rpy": rpy}
    summary = {"completed": True, "fall_events": [], "numerical_error": None, "exit_event": None,
               "balance_maintained": violation not in ("height", "tilt")}
    monkeypatch.setattr(positioning_validation, "run_impact", lambda *args, **kwargs: (summary, trace))
    result, _ = run_trial(None, "gate", "position", PARKED_BALL, duration=4.)
    assert result["final_stop_recovery"]["verified"]
    assert result["positioning"]["final_error_m"] == 0.
    assert result["positioning"]["passed"] is (violation is None)


def test_prediction_error_uses_observed_reference_crossing_and_excludes_future_samples():
    records = [{"time": 2.1, "prediction": predict_ball([10., .1, 1.], [2., 0., 0.])},
               {"time": 2.7, "prediction": predict_ball([10., .1, 1.], [2., 0., 0.])}]
    result = prediction_errors(records, {"11.0": {"time": 2.6, "position": [11., .2, .5]}})
    assert len(result["11.0"]["samples"]) == 1
    assert result["11.0"]["first"]["time_error_s"] == pytest.approx(0.)
    assert result["11.0"]["first"]["lateral_error_m"] == pytest.approx(-.1)
    assert result["11.11"]["reference"] is None
    assert result["11.11"]["maximum_absolute_errors"] is None


def test_noop_hooks_preserve_default_impact_physics_and_observe_delivered_steps(free_model, walking_assets):
    case = ImpactCase("hooks", miss=True, duration=.04)
    baseline, original = run_impact(free_model, case)
    observations = []

    def observe(sim):
        observations.append((sim.physics_steps, float(sim.data.time)))

    hooked, delivered = run_impact(free_model, case, command_provider=lambda sim: [0., 0., 0.],
                                   step_observer=observe)
    assert baseline["completed"] and hooked["completed"]
    assert baseline["trajectory_sha256"] == hooked["trajectory_sha256"]
    for key in original:
        np.testing.assert_array_equal(delivered[key], original[key])
    assert len(observations) == hooked["physics_steps"]
    assert observations[0] == (1, pytest.approx(.001))
    assert observations[-1] == (hooked["physics_steps"], pytest.approx(2.04))


def test_high_level_commands_are_delivered_recorded_and_repeat_after_fresh_reset(free_model, walking_assets):
    first, trace = run_trial(free_model, "position_short", "position", PARKED_BALL, duration=.08, target=.2)
    second, repeated = run_trial(free_model, "position_short", "position", PARKED_BALL, duration=.08, target=.2)
    assert first["completed"] and second["completed"]
    records = first["high_level_records"]
    assert len(records) > 100
    np.testing.assert_allclose(np.diff([row["time"] for row in records]), .02, atol=1e-12)
    assert all(row["command"] == [0., 0., 0.] for row in records if row["time"] < 2.-1e-9)
    assert any(row["command"][1] < 0 for row in records if row["time"] >= 2.-1e-9)
    for row in records:
        index = np.searchsorted(trace["time"], row["time"]+1e-9)
        np.testing.assert_array_equal(trace["command"][index], row["command"])
        np.testing.assert_allclose(row["goal_position"], world_to_goal(row["world_position"]))
    decoded = json.loads(trace["high_level_json_utf8"].tobytes())
    assert decoded == records
    assert first["trajectory_sha256"] == second["trajectory_sha256"]
    assert records == second["high_level_records"]
    for key in trace:
        np.testing.assert_array_equal(trace[key], repeated[key])
    assert trace["base_position"][-1, 1] != trace["base_position"][0, 1]


def test_free_robot_reaches_small_lateral_target_without_ball_contact(free_model, walking_assets):
    summary, trace = run_trial(free_model, "position_small", "position", PARKED_BALL, duration=16., target=.2)
    assert summary["completed"]
    assert not summary["fall_events"]
    assert summary["numerical_error"] is None
    assert not summary["effective_contact"]
    assert not summary["unexpected_contacts"]
    assert summary["positioning"]["passed"]
    assert abs(trace["goal_position"][-1, 0]-.2) <= .10
    assert summary["maximum_lateral_center_m"] <= .65
