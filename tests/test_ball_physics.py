import mujoco
import numpy as np
import pytest

from conftest import ball_state, contacts_with


def test_ball_has_regulation_size_mass_and_documented_shell_inertia(absent_model):
    geom = absent_model.geom("ball_collision")
    body = absent_model.body("ball")
    assert geom.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert geom.size[0] == pytest.approx(0.11)
    assert 0.68 < 2 * np.pi * geom.size[0] < 0.70
    assert body.mass[0] == pytest.approx(0.43)
    np.testing.assert_allclose(body.inertia, np.full(3, 2 / 3 * 0.43 * 0.11 ** 2), rtol=1e-10)
    assert absent_model.joint("ball_free").type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_ball_free_fall_before_ground_contact(absent_model):
    model = absent_model
    data, qpos, qvel = ball_state(model, (0, 0, 1.5))
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert data.time == pytest.approx(0.1)
    assert data.qvel[qvel + 2] == pytest.approx(-9.81 * data.time, abs=0.005)
    assert data.qpos[qpos + 2] == pytest.approx(1.5 - 0.5 * 9.81 * data.time ** 2, abs=0.002)
    assert not contacts_with(model, data, model.geom("ball_collision").id)


def test_ball_bounces_and_does_not_tunnel_through_ground(absent_model):
    model = absent_model
    data, qpos, qvel = ball_state(model, (0, 0, 1.2))
    ground = model.geom("ground").id
    ball = model.geom("ball_collision").id
    contact_seen = False
    positive_vz_after_contact = False
    minimum_height = np.inf
    for _ in range(2000):
        mujoco.mj_step(model, data)
        minimum_height = min(minimum_height, data.qpos[qpos + 2])
        contact_seen |= ground in contacts_with(model, data, ball)
        positive_vz_after_contact |= contact_seen and data.qvel[qvel + 2] > 0.5
    assert contact_seen
    assert positive_vz_after_contact
    assert minimum_height > 0.095
    assert np.isfinite(data.qpos).all()
    assert data.qpos[qpos + 2] >= 0.105


def test_high_speed_downward_ball_stays_above_plane(absent_model):
    data, qpos, _ = ball_state(absent_model, (0, 0, 0.5), (0, 0, -15))
    minimum_height = 0.5
    contact_seen = False
    for _ in range(300):
        mujoco.mj_step(absent_model, data)
        minimum_height = min(minimum_height, data.qpos[qpos + 2])
        contact_seen |= absent_model.geom("ground").id in contacts_with(absent_model, data, absent_model.geom("ball_collision").id)
    assert contact_seen
    assert minimum_height > 0.07


def test_ball_rolls_rotates_and_crosses_white_markings_without_contact(absent_model):
    model = absent_model
    data, qpos, qvel = ball_state(model, (-2, 0, 0.11), (3, 0, 0), (0, 3 / 0.11, 0))
    initial_quat = data.qpos[qpos + 3:qpos + 7].copy()
    ball = model.geom("ball_collision").id
    ground_seen = False
    for _ in range(1000):
        mujoco.mj_step(model, data)
        for other in contacts_with(model, data, ball):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            assert not name.startswith("line_")
            ground_seen |= name == "ground"
    assert ground_seen
    assert data.qpos[qpos] > 0.2
    assert abs(data.qvel[qvel + 4]) > 1
    assert abs(np.dot(initial_quat, data.qpos[qpos + 3:qpos + 7])) < 0.99
    assert data.qpos[qpos + 2] >= 0.105


def test_ball_spin_in_free_flight(absent_model):
    data, qpos, qvel = ball_state(absent_model, (0, 0, 3), (1, 0, 0), (0, 0, 10))
    for _ in range(100):
        mujoco.mj_step(absent_model, data)
    assert abs(data.qpos[qpos + 6]) > 0.4
    assert data.qvel[qvel + 5] == pytest.approx(10, abs=1e-8)


def test_identical_ball_initial_state_is_deterministic(absent_model):
    states = []
    for _ in range(2):
        data, _, _ = ball_state(absent_model, (-2, 0, 1), (5, 0.3, 0), (0, 8, 3))
        trajectory = []
        for step in range(1200):
            mujoco.mj_step(absent_model, data)
            if step % 10 == 0:
                trajectory.append(np.concatenate((data.qpos.copy(), data.qvel.copy())))
        states.append(trajectory)
    np.testing.assert_array_equal(states[0], states[1])
