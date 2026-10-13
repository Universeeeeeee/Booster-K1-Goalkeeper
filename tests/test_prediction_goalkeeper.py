"""Stage G current-state physics, modeled collision coverage and decisions."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from k1_goalkeeper.ball_prediction import BallPredictor, build_ball_floor_model
from k1_goalkeeper.predictive_goalkeeper import (
    CollisionCoverage,
    PredictiveGoalkeeper,
    SettlingPositionController,
)


@pytest.fixture(scope="module")
def predictor_model(scene_paths):
    return build_ball_floor_model(scene_paths["free"])


@pytest.fixture
def coverage_model():
    # Separate visible-only, incompatible-mask and receive-only geometry so a
    # positive result cannot be explained by the robot's rendered silhouette.
    return mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".001"/><worldbody>
      <body name="trunk" pos="10.4 0 .6"><freejoint name="world_joint"/>
        <geom name="body_collision" type="box" size=".2 .04 .1"/>
        <geom name="visual_only" type="sphere" size=".1" pos="0 .5 0"
              contype="0" conaffinity="0" mass="0"/>
        <geom name="incompatible_mask" type="sphere" size=".1" pos="0 1 0"
              contype="2" conaffinity="2"/>
        <geom name="receive_only" type="sphere" size=".1" pos="0 -1.5 0"
              contype="0" conaffinity="1"/>
      </body>
      <body name="ball" pos="8 0 .6"><freejoint name="ball_free"/>
        <geom name="ball_collision" type="sphere" size=".11"/>
      </body>
    </worldbody></mujoco>
    """)


def test_forecast_is_repeatable_current_state_only_and_does_not_mutate_inputs(predictor_model):
    predictor = BallPredictor(predictor_model, horizon=1.)
    qpos = np.array([8., .1, .5, 1., 0., 0., 0.])
    qvel = np.array([3., .2, -.5, 0., 10., 2.])
    before = (qpos.copy(), qvel.copy(), predictor_model.qpos0.copy())
    qpos.flags.writeable = qvel.flags.writeable = False
    first = predictor.forecast(qpos, qvel, planes=(9.,))
    assert first["valid"]
    expected_path = first["path"].copy()
    expected_crossing = dict(first["crossings"]["9.0"])
    # An unrelated forecast and caller edits to a returned path must not become
    # hidden history or affect the same observed current state on the next call.
    first["path"][:] = -100.
    first["crossings"]["9.0"]["lateral_m"] = -100.
    predictor.forecast([7., -.3, 1., 1., 0., 0., 0.], [4., 0., 1., 0., 0., 0.], (8.,))
    repeated = predictor.forecast(qpos, qvel, (9.,))
    np.testing.assert_array_equal(repeated["path"], expected_path)
    assert repeated["crossings"]["9.0"] == expected_crossing
    for original, actual in zip(before, (qpos, qvel, predictor_model.qpos0)):
        np.testing.assert_array_equal(actual, original)


def test_reduced_model_preserves_current_scene_ball_floor_and_solver_physics(predictor_model, free_model):
    assert predictor_model.nq == 7 and predictor_model.nv == 6
    assert predictor_model.ngeom == 2
    assert predictor_model.nu == 0
    for field in ("timestep", "gravity", "integrator", "solver", "iterations", "tolerance"):
        np.testing.assert_array_equal(getattr(predictor_model.opt, field), getattr(free_model.opt, field))
    for name in ("ground", "ball_collision"):
        reduced, full = predictor_model.geom(name).id, free_model.geom(name).id
        for field in ("geom_type", "geom_size", "geom_friction", "geom_solref", "geom_solimp",
                      "geom_condim", "geom_priority", "geom_contype", "geom_conaffinity"):
            np.testing.assert_array_equal(getattr(predictor_model, field)[reduced], getattr(free_model, field)[full])
    np.testing.assert_array_equal(predictor_model.body("ball").mass, free_model.body("ball").mass)
    np.testing.assert_array_equal(predictor_model.body("ball").inertia, free_model.body("ball").inertia)


def test_bounce_forecast_matches_independent_full_scene_steps(predictor_model, absent_model):
    qpos = np.array([8., .15, .5, 1., 0., 0., 0.])
    qvel = np.array([3., .1, -1., 0., 0., 0.])
    result = BallPredictor(predictor_model, horizon=1.).forecast(qpos, qvel, (9.,))
    assert result["valid"]
    assert result["floor_contact_events"]
    assert result["floor_contact_events"][0]["velocity_before"][2] < 0
    assert result["floor_contact_events"][0]["velocity_after"][2] > 0
    prediction = result["crossings"]["9.0"]
    assert prediction["model"] == "ball_floor_mujoco"
    assert prediction["future_floor_contact_unmodeled"] is False
    data = mujoco.MjData(absent_model)
    ball = absent_model.joint("ball_free")
    q, v = int(ball.qposadr[0]), int(ball.dofadr[0])
    data.qpos[q:q+7], data.qvel[v:v+6] = qpos, qvel
    mujoco.mj_forward(absent_model, data)
    for _ in range(1000):
        before, time = data.qpos[q:q+3].copy(), float(data.time)
        mujoco.mj_step(absent_model, data)
        after = data.qpos[q:q+3]
        if before[0] <= 9. < after[0]:
            fraction = (9.-before[0])/(after[0]-before[0])
            crossing = before+fraction*(after-before)
            assert prediction["time_to_plane_s"] == pytest.approx(time+fraction*.001, abs=1e-10)
            np.testing.assert_allclose([prediction["lateral_m"], prediction["height_m"]], crossing[1:], atol=1e-10)
            break
    else:
        pytest.fail("independent scene did not cross the requested plane")


@pytest.mark.parametrize("qpos,qvel,planes", [
    ([8., 0., .5], [2., 0., 0., 0., 0., 0.], (9.,)),
    ([8., 0., .5, 1., 0., 0., 0.], [2., 0., 0.], (9.,)),
    ([8., 0., np.nan, 1., 0., 0., 0.], [2., 0., 0., 0., 0., 0.], (9.,)),
    ([8., 0., .5, 1., 0., 0., 0.], [2., 0., 0., np.inf, 0., 0.], (9.,)),
    ([8., 0., .5, 2., 0., 0., 0.], [2., 0., 0., 0., 0., 0.], (9.,)),
    ([8., 0., .5, 1., 0., 0., 0.], [-2., 0., 0., 0., 0., 0.], (9.,)),
    ([8., 0., .5, 1., 0., 0., 0.], [2., 0., 0., 0., 0., 0.], (7.,)),
    ([8., 0., .5, 1., 0., 0., 0.], [2., 0., 0., 0., 0., 0.], (np.inf,)),
    ([8., 0., .5, 1., 0., 0., 0.], [2., 0., 0., 0., 0., 0.], ()),
])
def test_invalid_state_or_planes_abstain_without_crossing_claim(predictor_model, qpos, qvel, planes):
    result = BallPredictor(predictor_model, horizon=.01).forecast(qpos, qvel, planes)
    assert result["valid"] is False
    assert result["reason"]
    assert result["path"].shape == (0, 4)
    assert all(not crossing["valid"] and "time_to_plane_s" not in crossing
               for crossing in result["crossings"].values())


def test_unavailable_plane_and_short_horizon_do_not_invent_arrival(predictor_model):
    result = BallPredictor(predictor_model, horizon=.01).forecast(
        [8., 0., .5, 1., 0., 0., 0.], [2., 0., 0., 0., 0., 0.], (7., 9., 9.))
    assert result["valid"] is False
    assert len(result["crossings"]) == 2
    assert result["crossings"]["7.0"]["reason"] == "not approaching plane"
    assert result["crossings"]["9.0"]["reason"] == "no crossing within horizon"
    assert result["path"][-1, 0] == pytest.approx(.01)


@pytest.mark.parametrize("horizon", [0., -1., np.inf, np.nan])
def test_invalid_forecast_horizon_is_rejected(predictor_model, horizon):
    with pytest.raises(ValueError, match="horizon"):
        BallPredictor(predictor_model, horizon=horizon)


def test_coverage_excludes_visuals_and_incompatible_masks_but_accepts_receive_only(coverage_model):
    coverage = CollisionCoverage(coverage_model)
    names = {mujoco.mj_id2name(coverage_model, mujoco.mjtObj.mjOBJ_GEOM, geom) for geom in coverage.geoms}
    assert names == {"body_collision", "receive_only"}
    data = mujoco.MjData(coverage_model)
    for y, expected in ((0., True), (.5, False), (1., False), (-1.5, True)):
        result = coverage.evaluate(data, np.array([[0., 10.4, y, .6]]))
        assert result["covered"] is expected
        if expected:
            # Cross-check the delivered native contact set, independently of
            # CollisionCoverage's geometry list and signed-distance getter.
            actual = mujoco.MjData(coverage_model)
            q = int(coverage_model.joint("ball_free").qposadr[0])
            actual.qpos[q:q+3] = [10.4, y, .6]
            mujoco.mj_forward(coverage_model, actual)
            assert any(coverage.ball in (contact.geom1, contact.geom2) and contact.dist <= -.015
                       for contact in actual.contact[:actual.ncon])


def test_real_scene_disabled_visual_meshes_never_count_as_collision_coverage(free_model):
    coverage = CollisionCoverage(free_model)
    disabled_meshes = {geom for geom in range(free_model.ngeom)
                       if free_model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
                       and free_model.geom_contype[geom] == free_model.geom_conaffinity[geom] == 0}
    assert disabled_meshes
    assert disabled_meshes.isdisjoint(coverage.geoms)
    assert free_model.geom("ground").id not in coverage.geoms
    for geom in coverage.geoms:
        name = mujoco.mj_id2name(free_model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        assert not name.startswith("goal_")
        assert ((free_model.geom_contype[geom] & free_model.geom_conaffinity[coverage.ball])
                or (free_model.geom_contype[coverage.ball] & free_model.geom_conaffinity[geom]))


def test_coverage_requires_overlap_margin_and_does_not_mutate_real_pose(coverage_model):
    data = mujoco.MjData(coverage_model)
    data.qvel[:] = np.arange(coverage_model.nv)*.01
    data.time = 2.
    mujoco.mj_forward(coverage_model, data)
    before = {key: np.array(getattr(data, key), copy=True) for key in ("qpos", "qvel", "qacc", "geom_xpos")}
    coverage = CollisionCoverage(coverage_model)
    grazing = coverage.evaluate(data, np.array([[.1, 10.4-.2-.11+.01, 0., .6]]))
    assert not grazing["covered"]
    assert grazing["signed_distance_m"] == pytest.approx(-.01)
    overlapping = coverage.evaluate(data, np.array([[.1, 10.4-.2-.11+.02, 0., .6]]))
    assert overlapping["covered"]
    assert overlapping["signed_distance_m"] == pytest.approx(-.02)
    path = np.array([[.1, 10.4, .4, .6]])
    untouched_path = path.copy()
    assert not coverage.evaluate(data, path)["covered"]
    assert coverage.evaluate(data, path, target_lateral=.4)["covered"]
    assert not coverage.evaluate(data, path)["covered"]
    for key, original in before.items():
        np.testing.assert_array_equal(getattr(data, key), original)
    np.testing.assert_array_equal(path, untouched_path)
    assert data.time == 2.


def observed_sim(model):
    data = mujoco.MjData(model)
    base, ball = model.joint("world_joint"), model.joint("ball_free")
    q, v = int(base.qposadr[0]), int(base.dofadr[0])
    bq, bv = int(ball.qposadr[0]), int(ball.dofadr[0])
    data.qpos[q:q+7] = [10.4, 0., .6, 0., 0., 0., 1.]
    data.qpos[bq:bq+3], data.qvel[bv:bv+3] = [8., .25, .6], [2., 0., 0.]
    data.time = 2.
    controller = SimpleNamespace(data=data, base_qpos_address=q, base_dof_address=v)
    return SimpleNamespace(model=model, data=data, controller=controller, physics_steps=2000)


@pytest.mark.parametrize("time_left,height,inside,station,body,intercept,path_lateral,decision_body", [
    (7., .6, True, True, True, True, .25, True),
    (1., .6, True, False, True, False, .25, False),
    (7., 1.5, True, True, False, False, .25, False),
    (7., .6, False, True, True, False, .25, True),
    # F's 10cm station tolerance can accept a target beyond the travel interval;
    # a covered ideal station alone must not certify the commanded partial one.
    (4.5, .6, True, True, True, False, .36, False),
])
def test_station_reachability_body_coverage_and_intercept_are_separate(
        coverage_model, predictor_model, monkeypatch, time_left, height, inside, station, body, intercept,
        path_lateral, decision_body):
    adapter = PredictiveGoalkeeper(coverage_model, predictor_model)
    sim = observed_sim(coverage_model)
    def forecast(qpos, qvel, planes):
        return {"valid": True, "crossings": {
            str(float(plane)): {"valid": True, "plane_x": plane, "time_to_plane_s": time_left+offset,
                               "lateral_m": .25, "height_m": height, "inside_aperture": inside}
            for plane, offset in zip(planes, (1., 1.2, 0., .2))},
            "path": np.array([[time_left, 10.4, path_lateral, height]]), "floor_contact_events": []}
    monkeypatch.setattr(adapter.predictor, "forecast", forecast)
    command = adapter.command(sim)
    row = adapter.records[-1]
    assert row["station_reachable"] is station
    assert row["current_body_coverage"]["covered"] is False
    assert row["target_body_coverage"]["covered"] is body
    assert row["decision_body_coverage"]["covered"] is decision_body
    assert row["intercept_feasible"] is intercept
    assert row["interception_prediction"]["time_to_plane_s"] < row["prediction"]["time_to_plane_s"]
    assert row["interception_prediction"]["time_to_plane_s"] < adapter.forecast["crossings"]["10.4"]["time_to_plane_s"]
    if not body or not inside:
        assert command == [0., 0., 0.]
        assert row["tracking"] is None
    if station and body and inside:
        assert command[1] < 0.


def test_passive_goalkeeper_never_starts_prediction_or_motion(coverage_model, predictor_model, monkeypatch):
    adapter = PredictiveGoalkeeper(coverage_model, predictor_model, mode="passive")
    def unexpected_forecast(*args):
        pytest.fail("passive control must not run active forecasts")
    monkeypatch.setattr(adapter.predictor, "forecast", unexpected_forecast)
    assert adapter.command(observed_sim(coverage_model)) == [0., 0., 0.]
    assert adapter.refresh_count == 0


def test_invalid_forecasts_refresh_at_ten_hertz_and_stop_between_observations(
        coverage_model, predictor_model, monkeypatch):
    adapter = PredictiveGoalkeeper(coverage_model, predictor_model)
    sim = observed_sim(coverage_model)
    monkeypatch.setattr(adapter.predictor, "forecast", lambda qpos, qvel, planes:
                        BallPredictor._invalid(planes, "no crossing within horizon"))
    assert adapter.command(sim) == [0., 0., 0.]
    assert adapter.refresh_count == 1
    sim.data.time, sim.physics_steps = 2.02, 2020
    assert adapter.command(sim) == [0., 0., 0.]
    assert adapter.refresh_count == 1
    assert not adapter.records[-1]["forecast_refreshed"]
    sim.data.time, sim.physics_steps = 2.1, 2100
    assert adapter.command(sim) == [0., 0., 0.]
    assert adapter.refresh_count == 2
    assert adapter.records[-1]["forecast_refreshed"]


def test_cached_covered_station_must_remain_inside_current_travel_interval(
        coverage_model, predictor_model, monkeypatch):
    adapter = PredictiveGoalkeeper(coverage_model, predictor_model)
    sim = observed_sim(coverage_model)
    def forecast(qpos, qvel, planes):
        return {"valid": True, "crossings": {
            str(float(plane)): {"valid": True, "plane_x": plane, "time_to_plane_s": 7.+offset,
                               "lateral_m": .25, "height_m": .6, "inside_aperture": True}
            for plane, offset in zip(planes, (1., 1.2, 0., .2))},
            "path": np.array([[7., 10.4, .25, .6]]), "floor_contact_events": []}
    monkeypatch.setattr(adapter.predictor, "forecast", forecast)
    adapter.command(sim)
    assert adapter.records[-1]["intercept_feasible"]
    sim.data.qpos[sim.controller.base_qpos_address+1] = .66
    sim.data.time, sim.physics_steps = 2.02, 2020
    adapter.command(sim)
    row = adapter.records[-1]
    assert row["station_reachable"]
    assert row["decision_body_coverage"]["covered"]
    assert row["coverage_decision_lateral_m"] < row["reachability"]["interval_m"][0]
    assert row["intercept_feasible"] is False
    assert adapter.refresh_count == 1
    assert row["coverage_observed_time"] == 2.


@pytest.mark.parametrize("target", [-.099, .099])
def test_settling_controller_corrects_residual_inside_f_tolerance(target):
    controller = SettlingPositionController()
    command, tracking = controller.command(2., [10.4, 0., .6], np.pi, [0., 0., 0.], target)
    assert tracking["settling_correction_started"]
    assert command[1]*target < 0
    assert tracking["direction"] == np.sign(target)


@pytest.mark.parametrize("target", [-.08, 0., .08])
def test_settling_controller_keeps_two_centimeter_residual_margin(target):
    command, tracking = SettlingPositionController().command(2., [10.4, 0., .6], np.pi, [0., 0., 0.], target)
    assert command == [0., 0., 0.]
    assert not tracking["settling_correction_started"]


def test_settling_correction_respects_dwell_and_heading_safety():
    controller = SettlingPositionController()
    controller.last_switch = 1.
    command, tracking = controller.command(2., [10.4, 0., .6], np.pi, [0., 0., 0.], .09)
    assert command == [0., 0., 0.]
    assert not tracking["settling_correction_started"]
    command, tracking = controller.command(2.2, [10.4, 0., .6], np.pi, [0., 0., 0.], .09)
    assert command[1] < 0.
    assert tracking["settling_correction_started"]
    command, tracking = controller.command(2.22, [10.4, 0., .6], np.pi-.2, [0., 0., 0.], .09)
    assert command[1] == 0.
    assert not tracking["settling_correction_started"]
