from pathlib import Path

import mujoco
import pytest

from k1_goalkeeper.scene import build_scene


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def scene_paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("compiled-scenes")
    return {
        mode: build_scene(directory / f"goalkeeper_{mode}.xml", robot_mode=mode)
        for mode in ("free", "fixed", "absent")
    }


@pytest.fixture(scope="session")
def official_model():
    return mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/booster_assets/robots/K1/K1_22dof.xml")
    )


@pytest.fixture(scope="session")
def free_model(scene_paths):
    return mujoco.MjModel.from_xml_path(str(scene_paths["free"]))


@pytest.fixture(scope="session")
def fixed_model(scene_paths):
    return mujoco.MjModel.from_xml_path(str(scene_paths["fixed"]))


@pytest.fixture(scope="session")
def absent_model(scene_paths):
    return mujoco.MjModel.from_xml_path(str(scene_paths["absent"]))


def ball_state(model, position, velocity=(0, 0, 0), spin=(0, 0, 0)):
    data = mujoco.MjData(model)
    joint = model.joint("ball_free")
    qpos = int(joint.qposadr[0])
    qvel = int(joint.dofadr[0])
    data.qpos[qpos:qpos + 3] = position
    data.qpos[qpos + 3:qpos + 7] = (1, 0, 0, 0)
    data.qvel[qvel:qvel + 3] = velocity
    data.qvel[qvel + 3:qvel + 6] = spin
    mujoco.mj_forward(model, data)
    return data, qpos, qvel


def contacts_with(model, data, geom_id):
    return {
        int(contact.geom2 if contact.geom1 == geom_id else contact.geom1)
        for contact in data.contact
        if geom_id in (contact.geom1, contact.geom2)
    }


def mesh_world_vertices(model, data, name):
    geom = model.geom(name)
    mesh_id = int(geom.dataid[0])
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start:start + count]
    rotation = data.geom_xmat[geom.id].reshape(3, 3)
    return vertices @ rotation.T + data.geom_xpos[geom.id]
