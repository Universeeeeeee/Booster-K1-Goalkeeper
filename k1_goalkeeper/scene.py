"""Compose the official MJCF and MuJoCo-converted goal, leaving sources intact."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K1_XML = ROOT / "assets/booster_assets/robots/K1/K1_22dof.xml"
GOAL_URDF = ROOT / "models/futsal_goal_3x2_reference.urdf"


def load_config(path=None):
    return json.loads(Path(path or ROOT / "config/simulation.json").read_text())


def numbers(values):
    return " ".join(f"{float(v):.12g}" for v in values)


def convert_goal():
    """Use MuJoCo's URDF importer, explicitly retaining visual-only supports."""
    mujoco.MjModel.from_xml_path(str(GOAL_URDF))
    source = ET.parse(GOAL_URDF).getroot()
    extension = ET.SubElement(source, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false", fusestatic="false")
    model = mujoco.MjModel.from_xml_string(ET.tostring(source, encoding="unicode"))
    with tempfile.TemporaryDirectory() as directory:
        converted = Path(directory) / "goal.xml"
        mujoco.mj_saveLastXML(str(converted), model)
        return ET.parse(converted).getroot().find("worldbody/body")


def line(world, name, x, y, half_x, half_y):
    ET.SubElement(world, "geom", name=f"line_{name}", type="box",
                  pos=numbers([x, y, .001]), size=numbers([half_x, half_y, .0001]),
                  rgba="1 1 1 1", contype="0", conaffinity="0", group="2", density="0")


def arc(world, assets, name, center, radius, start, end, segments):
    """Thin annular strip: accurate visual width, no collision surface."""
    vertices = []
    for angle in np.linspace(start, end, segments + 1):
        for z in [-.0001, .0001]:
            for r in [radius - .04, radius + .04]:
                vertices.append([r * math.cos(angle), r * math.sin(angle), z])
    faces = []
    for i in range(segments):
        a, b = 4 * i, 4 * (i + 1)
        faces.extend([[a+2, a+3, b+3], [a+2, b+3, b+2],
                      [a, b+1, a+1], [a, b, b+1],
                      [a, a+2, b+2], [a, b+2, b],
                      [a+1, b+3, a+3], [a+1, b+1, b+3]])
    n = 4 * segments
    faces.extend([[0, 1, 3], [0, 3, 2], [n, n+3, n+1], [n, n+2, n+3]])
    ET.SubElement(assets, "mesh", name=f"mesh_{name}",
                  vertex=numbers(np.asarray(vertices).ravel()),
                  face=" ".join(str(v) for face in faces for v in face))
    ET.SubElement(world, "geom", name=f"line_{name}", type="mesh", mesh=f"mesh_{name}",
                  pos=numbers([*center, .001]), rgba="1 1 1 1", group="2",
                  contype="0", conaffinity="0", density="0")


def add_field(world, assets):
    ground = world.find("geom[@name='ground']")
    ground.attrib.clear()
    # Keep the official ground contact parameters; ball priority overrides its pairs.
    ground.attrib.update(name="ground", type="plane", pos="0 0 0", size="13 9 1",
                         rgba="0 0 0 0", condim="1", friction="0.4 0.005 0.0001")
    ET.SubElement(world, "geom", name="turf", type="box", pos="0 0 -.011",
                  size="13 9 .01", rgba="0.12 0.36 0.18 1", contype="0",
                  conaffinity="0", group="2", density="0")
    for side, x in [("left", -11), ("right", 11)]:
        line(world, f"boundary_{side}", x, 0, .04, 7)
        sign = -1 if x < 0 else 1
        for area, depth, half_width in [("penalty", 5, 4), ("goalarea", 2, 2.5)]:
            line(world, f"{area}_{side}_front", x-sign*depth, 0, .04, half_width)
            for edge, y in [("top", half_width), ("bottom", -half_width)]:
                line(world, f"{area}_{side}_{edge}", x-sign*depth/2, y, depth/2, .04)
        for axis in ["x", "y"]:
            line(world, f"penalty_mark_{side}_{axis}", x-sign*3.5, 0,
                 .1 if axis == "x" else .04, .04 if axis == "x" else .1)
        for edge, y in [("top", 7), ("bottom", -7)]:
            start = {(False, False): 0, (False, True): -math.pi/2,
                     (True, False): math.pi/2, (True, True): math.pi}[(x > 0, y > 0)]
            arc(world, assets, f"corner_{side}_{edge}", [x, y], .7, start, start+math.pi/2, 64)
    for edge, y in [("top", 7), ("bottom", -7)]:
        line(world, f"boundary_{edge}", 0, y, 11, .04)
    line(world, "halfway", 0, 0, .04, 7)
    line(world, "center_mark_x", 0, 0, .1, .04)
    line(world, "center_mark_y", 0, 0, .04, .1)
    arc(world, assets, "center_circle", [0, 0], 2, 0, 2*math.pi, 256)


def add_camera(world, name, eye, target, fovy=45):
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    up = np.array([0., 1., 0.]) if abs(z[2]) > .99 else np.array([0., 0., 1.])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(world, "camera", name=name, pos=numbers(eye),
                  xyaxes=numbers([*x, *y]), fovy=str(fovy))


def build_scene(output=None, robot_mode="free", config=None):
    if robot_mode not in {"free", "fixed", "absent"}:
        raise ValueError("robot_mode must be free, fixed, or absent")
    config = deepcopy(config if config is not None else load_config())
    radius, mass = config["ball"]["radius"], config["ball"]["mass"]
    if not (0 < radius < 1 and mass > 0):
        raise ValueError("ball radius and mass must be positive (radius < 1 m)")
    suffix = "" if robot_mode == "free" else f"_{robot_mode}"
    output = Path(output or ROOT / f"scenes/goalkeeper{suffix}.xml").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not K1_XML.exists():
        raise FileNotFoundError("K1 assets missing. Run: python scripts/fetch_assets.py")
    root = ET.parse(K1_XML).getroot()
    root.set("model", f"K1_goalkeeper_{robot_mode}")
    root.find("compiler").set("meshdir", os.path.relpath(K1_XML.parent / "meshes", output.parent))
    world, assets = root.find("worldbody"), root.find("asset")
    robot = world.find("body[@name='trunk']")
    robot.set("pos", numbers(config["robot"]["position"]))
    yaw = math.radians(config["robot"]["yaw_degrees"])
    robot.set("quat", numbers([math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))
    for i, geom in enumerate(robot.iter("geom")):
        if "name" not in geom.attrib:
            geom.set("name", f"k1_geom_{i}")
    if robot_mode != "free":
        for tag in ["actuator", "sensor"]:
            node = root.find(tag)
            if node is not None:
                root.remove(node)
        if robot_mode == "absent":
            world.remove(robot)
        else:
            for body in robot.iter("body"):
                for joint in list(body.findall("joint")) + list(body.findall("freejoint")):
                    body.remove(joint)
    add_field(world, assets)
    goal = convert_goal()
    for side, x, angle in [("left", -11, -math.pi/2), ("right", 11, math.pi/2)]:
        instance = deepcopy(goal)
        for node in instance.iter():
            if "name" in node.attrib:
                node.set("name", f"goal_{side}_{node.get('name')}")
        instance.set("name", f"goal_{side}")
        instance.set("pos", numbers([x, 0, 0]))
        instance.set("quat", numbers([math.cos(angle/2), 0, 0, math.sin(angle/2)]))
        for geom in instance.iter("geom"):
            if geom.get("name").endswith("_collision"):
                geom.set("rgba", "0 0 0 0")
        world.append(instance)
    ET.SubElement(assets, "texture", name="ball_checker", type="2d", builtin="checker",
                  rgb1=".95 .95 .95", rgb2=".12 .16 .2", width="128", height="64")
    ET.SubElement(assets, "material", name="ball_material", texture="ball_checker", texrepeat="4 2")
    ball = ET.SubElement(world, "body", name="ball", pos=numbers(config["ball"]["position"]))
    ET.SubElement(ball, "freejoint", name="ball_free")
    inertia = 2 / 3 * mass * radius**2  # thin-shell approximation, to be calibrated
    ET.SubElement(ball, "inertial", pos="0 0 0", mass=str(mass), diaginertia=numbers([inertia]*3))
    ET.SubElement(ball, "geom", name="ball_collision", type="sphere", size=str(radius),
                  material="ball_material", condim="6", priority="1",
                  friction=numbers(config["ball"]["friction"]),
                  solref=numbers(config["ball"]["solref"]), solimp=numbers(config["ball"]["solimp"]))
    add_camera(world, "overview", [0, 0, 27], [0, 0, 0], 42)
    add_camera(world, "goal_front", [5.4, -4.6, 3.0], [10.7, 0, 1.0], 48)
    add_camera(world, "robot_side", [10.1, -3.7, 1.6], [10.4, 0, .65], 38)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1000")
    ET.SubElement(visual, "map", znear=".01", zfar="100")
    ET.SubElement(root, "statistic", center="0 0 .7", extent="28")
    ET.indent(root)
    output.write_text(ET.tostring(root, encoding="unicode") + "\n")
    mujoco.MjModel.from_xml_path(str(output))  # fail here, never deliver an uncompiled scene
    return output
