import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from conftest import ROOT, contacts_with


def test_official_assets_compile_and_meshes_exist(official_model):
    path = ROOT / "assets/booster_assets/robots/K1/K1_22dof.xml"
    source = ET.parse(path).getroot()
    meshes = source.findall("asset/mesh")
    assert len(meshes) == official_model.nmesh == 24
    for mesh in meshes:
        assert (path.parent / "meshes" / mesh.attrib["file"]).is_file()
    assert official_model.nu == 22
    assert official_model.njnt == 23
    assert official_model.joint("world_joint").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert official_model.opt.timestep == pytest.approx(0.001)


def test_existing_goal_urdf_compiles():
    goal = mujoco.MjModel.from_xml_path(str(ROOT / "models/futsal_goal_3x2_reference.urdf"))
    assert goal.njnt == 0
    assert goal.ngeom >= 3
    assert all(goal.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER)


def test_free_scene_preserves_official_k1_dynamics(official_model, free_model):
    original = official_model
    scene = free_model
    assert scene.nu == original.nu == 22
    assert scene.njnt == original.njnt + 1
    for name in ("timestep", "tolerance", "impratio", "solver", "integrator"):
        np.testing.assert_allclose(getattr(scene.opt, name), getattr(original.opt, name))
    for old_id in range(1, original.nbody):
        name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_BODY, old_id)
        new_id = scene.body(name).id
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_allclose(getattr(scene, field)[new_id], getattr(original, field)[old_id], rtol=0, atol=1e-12)
        if name != "trunk":
            for field in ("body_pos", "body_quat"):
                np.testing.assert_allclose(getattr(scene, field)[new_id], getattr(original, field)[old_id], rtol=0, atol=1e-12)
        old_geoms = np.flatnonzero(original.geom_bodyid == old_id)
        new_geoms = np.flatnonzero(scene.geom_bodyid == new_id)
        assert len(old_geoms) == len(new_geoms)
        for old_geom, new_geom in zip(old_geoms, new_geoms):
            for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction", "geom_solref", "geom_solimp"):
                np.testing.assert_allclose(getattr(scene, field)[new_geom], getattr(original, field)[old_geom], rtol=0, atol=1e-12)
    for old_id in range(original.njnt):
        name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_JOINT, old_id)
        new_id = scene.joint(name).id
        for field in ("jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_actfrcrange", "jnt_actfrclimited"):
            np.testing.assert_allclose(getattr(scene, field)[new_id], getattr(original, field)[old_id], rtol=0, atol=1e-12)
        old_dof, new_dof = original.jnt_dofadr[old_id], scene.jnt_dofadr[new_id]
        count = 6 if original.jnt_type[old_id] == mujoco.mjtJoint.mjJNT_FREE else 1
        for field in ("dof_armature", "dof_damping", "dof_frictionloss"):
            np.testing.assert_allclose(getattr(scene, field)[new_dof:new_dof + count], getattr(original, field)[old_dof:old_dof + count], rtol=0, atol=1e-12)
    for old_id in range(original.nu):
        name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_ACTUATOR, old_id)
        assert mujoco.mj_id2name(scene, mujoco.mjtObj.mjOBJ_ACTUATOR, old_id) == name
        for field in ("actuator_gear", "actuator_forcerange", "actuator_forcelimited", "actuator_gainprm", "actuator_biasprm", "actuator_gaintype", "actuator_biastype"):
            np.testing.assert_allclose(getattr(scene, field)[old_id], getattr(original, field)[old_id], rtol=0, atol=1e-12)
        old_joint = int(original.actuator_trnid[old_id, 0])
        new_joint = int(scene.actuator_trnid[old_id, 0])
        assert mujoco.mj_id2name(scene, mujoco.mjtObj.mjOBJ_JOINT, new_joint) == mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_JOINT, old_joint)


def test_isolation_modes_are_distinct(free_model, fixed_model, absent_model):
    assert free_model.joint("world_joint").type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert fixed_model.nu == absent_model.nu == 0
    assert fixed_model.njnt == absent_model.njnt == 1
    assert fixed_model.body("trunk").id > 0
    assert mujoco.mj_name2id(absent_model, mujoco.mjtObj.mjOBJ_BODY, "trunk") == -1


def test_robot_faces_field_and_feet_start_above_ground(free_model, official_model):
    data = mujoco.MjData(free_model)
    mujoco.mj_forward(free_model, data)
    original_data = mujoco.MjData(official_model)
    mujoco.mj_forward(official_model, original_data)
    camera_name = "head_booster_stereo_rgb_link"
    original_axis = original_data.xmat[official_model.body(camera_name).id].reshape(3, 3)[:, 2]
    scene_axis = data.xmat[free_model.body(camera_name).id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(original_axis, (1, 0, 0), atol=1e-6)
    np.testing.assert_allclose(scene_axis, (-1, 0, 0), atol=1e-6)
    assert data.xpos[free_model.body("trunk").id, 0] == pytest.approx(10.4)
    for side in ("left", "right"):
        body_id = free_model.body(f"{side}_ankle_roll_link").id
        geoms = np.where((free_model.geom_bodyid == body_id) & (free_model.geom_contype != 0))[0]
        assert len(geoms) == 1
        geom_id = geoms[0]
        assert free_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        lower_z = data.geom_xpos[geom_id, 2] - np.abs(rotation[2]) @ free_model.geom_size[geom_id]
        assert lower_z == pytest.approx(0.002, abs=1e-6)
    robot_bodies = {free_model.body(mujoco.mj_id2name(official_model, mujoco.mjtObj.mjOBJ_BODY, i)).id for i in range(1, official_model.nbody)}
    for contact in data.contact:
        assert contact.dist >= -1e-6 or not (free_model.geom_bodyid[contact.geom1] in robot_bodies or free_model.geom_bodyid[contact.geom2] in robot_bodies)


def test_free_robot_contacts_ground(free_model):
    data = mujoco.MjData(free_model)
    ground = free_model.geom("ground").id
    touched = False
    for _ in range(150):
        mujoco.mj_step(free_model, data)
        others = contacts_with(free_model, data, ground)
        touched |= any("ankle_roll" in (mujoco.mj_id2name(free_model, mujoco.mjtObj.mjOBJ_BODY, int(free_model.geom_bodyid[g])) or "") for g in others)
    assert touched
    assert np.isfinite(data.qpos).all()
