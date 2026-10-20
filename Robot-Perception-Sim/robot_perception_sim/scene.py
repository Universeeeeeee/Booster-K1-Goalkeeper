"""Add the camera component to the existing goalkeeper scene builder."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from k1_goalkeeper.scene import build_scene
from .camera import add_stereo_cameras, load_camera_config

ROOT = Path(__file__).resolve().parents[1]


def build_stereo_scene(output=None, camera_config=None, scene_config=None):
    """Reuse the full field, goal, ball, robot and lights; write only a new scene."""
    output = Path(output or ROOT / "results/stereo/scene.xml").resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Generated scene must stay inside Robot-Perception-Sim")
    config = load_camera_config() if camera_config is None else camera_config
    path = build_scene(output=output, robot_mode="free", config=scene_config)
    root = ET.parse(path).getroot()
    add_stereo_cameras(root, config)
    root.find("visual/global").set("offwidth", str(max(1600, config["image_width_px"])))
    root.find("visual/global").set("offheight", str(max(1000, config["image_height_px"])))
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    mujoco.MjModel.from_xml_path(str(path))
    return path
