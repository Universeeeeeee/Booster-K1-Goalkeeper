import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from conftest import ROOT, ball_state, contacts_with


@pytest.mark.parametrize("side,sign", [("left", -1), ("right", 1)])
def test_converted_goal_retains_source_geometry_and_clear_opening(absent_model, side, sign):
    model = absent_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    source = ET.parse(ROOT / "models/futsal_goal_3x2_reference.urdf").getroot()
    body = model.body(f"goal_{side}")
    assert body.jntnum[0] == 0
    np.testing.assert_allclose(data.xpos[body.id], (sign * 11, 0, 0), atol=1e-12)
    rotation = data.xmat[body.id].reshape(3, 3)
    np.testing.assert_allclose(rotation[:, 1], (-sign, 0, 0), atol=1e-12)
    for collision in source.findall("link/collision"):
        geom = model.geom(f"goal_{side}_{collision.attrib['name']}")
        cylinder = collision.find("geometry/cylinder")
        expected = (float(cylinder.attrib["radius"]), float(cylinder.attrib["length"]) / 2)
        np.testing.assert_allclose(geom.size[:2], expected, atol=1e-12)
        assert geom.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
        local_pos = np.fromstring(collision.find("origin").attrib["xyz"], sep=" ")
        np.testing.assert_allclose(data.geom_xpos[geom.id], data.xpos[body.id] + rotation @ local_pos, atol=1e-12)
        assert geom.contype[0] and geom.conaffinity[0]
    posts = [model.geom(f"goal_{side}_{post}_post_collision") for post in ("left", "right")]
    y_positions = sorted(data.geom_xpos[g.id, 1] for g in posts)
    assert y_positions[1] - y_positions[0] - sum(g.size[0] for g in posts) == pytest.approx(3)
    crossbar = model.geom(f"goal_{side}_crossbar_collision")
    assert data.geom_xpos[crossbar.id, 2] - crossbar.size[0] == pytest.approx(2)
    axis = data.geom_xmat[crossbar.id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(np.abs(axis), (0, 1, 0), atol=1e-12)
    goal_geoms = np.flatnonzero(model.geom_bodyid == body.id)
    assert len(goal_geoms) == len(source.findall("link/visual")) + 3
    collidable = [g for g in goal_geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
    assert len(collidable) == 3
    for visual in source.findall("link/visual"):
        geom = model.geom(f"goal_{side}_{visual.attrib['name']}")
        cylinder = visual.find("geometry/cylinder")
        # mj_saveLastXML writes six significant digits during URDF conversion.
        np.testing.assert_allclose(geom.size[:2], (float(cylinder.attrib["radius"]), float(cylinder.attrib["length"]) / 2), atol=3e-6, rtol=0)
        origin = visual.find("origin")
        local_pos = np.fromstring(origin.attrib["xyz"], sep=" ")
        np.testing.assert_allclose(data.geom_xpos[geom.id], data.xpos[body.id] + rotation @ local_pos, atol=1e-6, rtol=0)
        roll, pitch, yaw = np.fromstring(origin.attrib["rpy"], sep=" ")
        sr, sp, sy = np.sin((roll, pitch, yaw))
        cr, cp, cy = np.cos((roll, pitch, yaw))
        source_rotation = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ])
        np.testing.assert_allclose(data.geom_xmat[geom.id].reshape(3, 3), rotation @ source_rotation, atol=3e-6, rtol=0)
        assert geom.contype[0] == geom.conaffinity[0] == 0


@pytest.mark.parametrize("side,sign", [("left", -1), ("right", 1)])
@pytest.mark.parametrize("part", ["left_post_collision", "right_post_collision", "crossbar_collision"])
def test_ball_hits_each_post_and_crossbar_and_goal_stays_static(absent_model, side, sign, part):
    model = absent_model
    initial = mujoco.MjData(model)
    mujoco.mj_forward(model, initial)
    target = model.geom(f"goal_{side}_{part}").id
    center = initial.geom_xpos[target].copy()
    if part != "crossbar_collision":
        center[2] = 0.7
    speed = 12
    approach_time = (1 - 0.11 - 0.04) / speed
    start = center + (-sign, 0, 0.5 * 9.81 * approach_time ** 2)
    data, qpos, qvel = ball_state(model, start, (sign * speed, 0, 0))
    ball = model.geom("ball_collision").id
    goal_body = model.body(f"goal_{side}").id
    body_position = data.xpos[goal_body].copy()
    body_orientation = data.xquat[goal_body].copy()
    contact_seen = False
    reflected = False
    farthest_forward = -np.inf
    for _ in range(250):
        mujoco.mj_step(model, data)
        contact_seen |= target in contacts_with(model, data, ball)
        reflected |= contact_seen and sign * data.qvel[qvel] < -0.05
        farthest_forward = max(farthest_forward, sign * (data.qpos[qpos] - center[0]))
        np.testing.assert_array_equal(data.xpos[goal_body], body_position)
        np.testing.assert_array_equal(data.xquat[goal_body], body_orientation)
    assert contact_seen
    assert reflected
    assert farthest_forward < 0


@pytest.mark.parametrize("part", ["torso", "foot"])
@pytest.mark.parametrize("model_fixture", ["fixed_model", "free_model"])
def test_ball_contacts_official_robot_collision_geometries(request, model_fixture, part):
    model = request.getfixturevalue(model_fixture)
    pose = mujoco.MjData(model)
    mujoco.mj_forward(model, pose)
    body_name = "trunk" if part == "torso" else "left_ankle_roll_link"
    body = model.body(body_name).id
    geoms = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0) & (model.geom_type == mujoco.mjtGeom.mjGEOM_BOX))
    assert len(geoms) == 1
    target = geoms[0]
    center = pose.geom_xpos[target].copy()
    if part == "torso":
        center[2] += 0.5 * 9.81 * (0.43 / 5) ** 2
    else:
        center[2] = 0.11
    center[0] -= 0.6
    data, qpos, qvel = ball_state(model, center, (5, 0, 0), (0, 5 / 0.11, 0) if part == "foot" else (0, 0, 0))
    ball = model.geom("ball_collision").id
    touched = False
    changed_velocity = False
    for _ in range(250):
        mujoco.mj_step(model, data)
        touched |= target in contacts_with(model, data, ball)
        changed_velocity |= touched and abs(data.qvel[qvel] - 5) > 0.5
    assert touched
    assert changed_velocity
    assert np.isfinite(data.qpos).all()
