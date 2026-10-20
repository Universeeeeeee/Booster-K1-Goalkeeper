from pathlib import Path
import sys

import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
sys.dont_write_bytecode = True

from robot_perception_sim.camera import load_camera_config
from robot_perception_sim.scene import build_stereo_scene


@pytest.fixture(scope="session")
def config():
    return load_camera_config()


@pytest.fixture(scope="session")
def model(config):
    path = build_stereo_scene(ROOT / "results/tests/scene.xml", config)
    return mujoco.MjModel.from_xml_path(str(path))
