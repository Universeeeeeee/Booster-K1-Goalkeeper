import mujoco
import numpy as np
import pytest

from conftest import mesh_world_vertices


def test_turf_ground_and_observation_cameras(absent_model):
    model = absent_model
    ground = model.geom("ground")
    assert ground.type[0] == mujoco.mjtGeom.mjGEOM_PLANE
    assert ground.contype[0] != 0 and ground.conaffinity[0] != 0
    turf = model.geom("turf")
    np.testing.assert_allclose(turf.size, (13, 9, 0.01), atol=1e-12)
    np.testing.assert_allclose(turf.pos, (0, 0, -0.011), atol=1e-12)
    assert turf.contype[0] == turf.conaffinity[0] == 0
    assert model.ncam >= 3


@pytest.mark.parametrize("name,position,size", [
    ("boundary_left", (-11, 0), (0.04, 7)),
    ("boundary_right", (11, 0), (0.04, 7)),
    ("boundary_top", (0, 7), (11, 0.04)),
    ("boundary_bottom", (0, -7), (11, 0.04)),
    ("halfway", (0, 0), (0.04, 7)),
    ("penalty_left_front", (-6, 0), (0.04, 4)),
    ("penalty_right_front", (6, 0), (0.04, 4)),
    ("penalty_left_top", (-8.5, 4), (2.5, 0.04)),
    ("penalty_left_bottom", (-8.5, -4), (2.5, 0.04)),
    ("penalty_right_top", (8.5, 4), (2.5, 0.04)),
    ("penalty_right_bottom", (8.5, -4), (2.5, 0.04)),
    ("goalarea_left_front", (-9, 0), (0.04, 2.5)),
    ("goalarea_right_front", (9, 0), (0.04, 2.5)),
    ("goalarea_left_top", (-10, 2.5), (1, 0.04)),
    ("goalarea_left_bottom", (-10, -2.5), (1, 0.04)),
    ("goalarea_right_top", (10, 2.5), (1, 0.04)),
    ("goalarea_right_bottom", (10, -2.5), (1, 0.04)),
    ("center_mark_x", (0, 0), (0.1, 0.04)),
    ("center_mark_y", (0, 0), (0.04, 0.1)),
    ("penalty_mark_left_x", (-7.5, 0), (0.1, 0.04)),
    ("penalty_mark_left_y", (-7.5, 0), (0.04, 0.1)),
    ("penalty_mark_right_x", (7.5, 0), (0.1, 0.04)),
    ("penalty_mark_right_y", (7.5, 0), (0.04, 0.1)),
])
def test_loaded_line_dimensions(absent_model, name, position, size):
    data = mujoco.MjData(absent_model)
    mujoco.mj_forward(absent_model, data)
    geom = absent_model.geom(f"line_{name}")
    assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(data.geom_xpos[geom.id, :2], position, atol=1e-12)
    np.testing.assert_allclose(geom.size[:2], size, atol=1e-12)
    assert geom.size[2] < 0.001


def test_center_circle_is_flat_ring_with_correct_width(absent_model):
    data = mujoco.MjData(absent_model)
    mujoco.mj_forward(absent_model, data)
    assert absent_model.geom("line_center_circle").type[0] == mujoco.mjtGeom.mjGEOM_MESH
    vertices = mesh_world_vertices(absent_model, data, "line_center_circle")
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    assert radii.min() == pytest.approx(1.96, abs=2e-6)
    assert radii.max() == pytest.approx(2.04, abs=2e-6)
    assert np.ptp(vertices[:, 2]) < 0.001
    for axis in range(2):
        assert vertices[:, axis].min() == pytest.approx(-2.04, abs=2e-6)
        assert vertices[:, axis].max() == pytest.approx(2.04, abs=2e-6)


@pytest.mark.parametrize("side,x", [("left", -11), ("right", 11)])
@pytest.mark.parametrize("edge,y", [("bottom", -7), ("top", 7)])
def test_corner_arcs_are_interior_quarter_circles(absent_model, side, x, edge, y):
    data = mujoco.MjData(absent_model)
    mujoco.mj_forward(absent_model, data)
    name = f"line_corner_{side}_{edge}"
    assert absent_model.geom(name).type[0] == mujoco.mjtGeom.mjGEOM_MESH
    vertices = mesh_world_vertices(absent_model, data, name)
    offsets = vertices[:, :2] - (x, y)
    radii = np.linalg.norm(offsets, axis=1)
    assert radii.min() == pytest.approx(0.66, abs=2e-6)
    assert radii.max() == pytest.approx(0.74, abs=2e-6)
    interior = offsets * (-np.sign(x), -np.sign(y))
    assert interior.min() >= -2e-6
    np.testing.assert_allclose(interior.max(axis=0), (0.74, 0.74), atol=2e-6)
    assert np.ptp(vertices[:, 2]) < 0.001


def test_all_white_lines_are_visual_only(absent_model):
    ids = [i for i in range(absent_model.ngeom) if (mujoco.mj_id2name(absent_model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("line_")]
    assert len(ids) == 28
    assert np.all(absent_model.geom_contype[ids] == 0)
    assert np.all(absent_model.geom_conaffinity[ids] == 0)
    assert np.all(absent_model.geom_group[ids] == 2)
    np.testing.assert_allclose(absent_model.geom_rgba[ids, :3], 1, atol=1e-6)
