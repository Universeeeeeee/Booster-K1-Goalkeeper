"""Opt-in I2 model: conservative shank coverage, unchanged native dynamics."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from .scene import build_scene, numbers


MODEL_VERSION = 'i2_shank_capsules_v1'
# Local-body endpoints and radius, in metres. Both capsules lie inside the
# corresponding official Shank convex hull; the validator checks every plane.
SHANK_FROMTO = (.007, 0., -.05, .007, 0., -.18)
SHANK_RADIUS = .027


def build_training_scene(output, config=None):
    """Build a separate free-base scene; callers opt in by passing its model.

    Original visuals, collision geoms, inertials, joints, actuators, solver and
    contact parameters are retained. No extra exclusions or inflated margins.
    """
    output = Path(output).resolve()
    build_scene(output, config=config)
    tree = ET.parse(output)
    root = tree.getroot()
    root.set('model', f'K1_goalkeeper_{MODEL_VERSION}')
    for side in ('left', 'right'):
        body = root.find(f".//body[@name='{side}_knee_pitch_link']")
        ET.SubElement(body, 'geom', name=f'i2_{side}_shank', type='capsule',
                      fromto=numbers(SHANK_FROMTO), size=str(SHANK_RADIUS),
                      density='0', contype='1', conaffinity='1', group='3',
                      rgba='0 0.6 1 0')
    ET.indent(root)
    tree.write(output, encoding='unicode')
    mujoco.MjModel.from_xml_path(str(output))
    return output
