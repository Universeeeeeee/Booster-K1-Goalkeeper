"""Nominal rectified stereo geometry and MuJoCo/optical frame conversion."""

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/cameras/zedx_wide.json"
CAMERA_NAMES = ("zedx_left_camera", "zedx_right_camera")


def load_camera_config(path=None):
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    for key in ("image_width_px", "image_height_px"):
        value = config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not math.isfinite(config["baseline_m"]) or config["baseline_m"] <= 0:
        raise ValueError("baseline_m must be finite and positive")
    if not math.isfinite(config["horizontal_fov_deg"]) or not 0 < config["horizontal_fov_deg"] < 180:
        raise ValueError("horizontal_fov_deg must be between 0 and 180")
    position = np.asarray(config["mount"]["position_parent_m"], dtype=float)
    quaternion = np.asarray(config["mount"]["quaternion_parent_from_mount_wxyz"], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("mount position must contain three finite values")
    if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or not np.isclose(np.linalg.norm(quaternion), 1, rtol=0, atol=1e-8)):
        raise ValueError("mount quaternion must be a unit quaternion in wxyz order")
    if (config["model"]["projection"] != "pinhole" or not config["model"]["square_pixels"]
            or config["model"]["distortion"] != "ideal_zero"):
        raise ValueError("Only the documented ideal rectified pinhole model is supported")
    return config


def intrinsics(config):
    width, height = config["image_width_px"], config["image_height_px"]
    focal = width / (2 * math.tan(math.radians(config["horizontal_fov_deg"]) / 2))
    return np.array([[focal, 0, (width - 1) / 2],
                     [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=float)


def vertical_fov_deg(config):
    return math.degrees(2 * math.atan(config["image_height_px"] / (2 * intrinsics(config)[1, 1])))


def add_stereo_cameras(root, config):
    """A massless mount uses optical axes; each MuJoCo camera is rotated by Rx(pi)."""
    mount_config = config["mount"]
    head = root.find(f".//body[@name='{mount_config['parent_body']}']")
    if head is None:
        raise ValueError(f"Camera parent body not found: {mount_config['parent_body']}")
    component = DEFAULT_CONFIG.parents[2] / "models/zedx_stereo.xml"
    component_root = ET.parse(component).getroot()
    mesh = component_root.find("asset/mesh")
    mesh.set("file", str(component.parent / mesh.get("file")))
    root.find("asset").append(mesh)
    mount = component_root.find("worldbody/body")
    mount.set("pos", " ".join(map(str, mount_config["position_parent_m"])))
    mount.set("quat", " ".join(map(str, mount_config["quaternion_parent_from_mount_wxyz"])))
    head.append(mount)
    for name, sign in zip(CAMERA_NAMES, (-1, 1)):
        camera = mount.find(f"camera[@name='{name}']")
        camera.set("pos", f"{sign * config['baseline_m'] / 2:.17g} 0 0")
        camera.set("fovy", str(vertical_fov_deg(config)))


def optical_pose(data, camera_name):
    """Return T_WC: column-vector optical coordinates -> world coordinates, metres."""
    camera = data.camera(camera_name)
    transform = np.eye(4)
    transform[:3, :3] = camera.xmat.reshape(3, 3) @ np.diag([1, -1, -1])
    transform[:3, 3] = camera.xpos
    return transform


def project_world(point_world, calibration, world_from_camera):
    """Project a single point; return pixel-centre coordinates and positive optical Z."""
    point = np.asarray(point_world, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("point_world must contain three finite values")
    camera_point = world_from_camera[:3, :3].T @ (point - world_from_camera[:3, 3])
    if camera_point[2] <= 0:
        raise ValueError("point must be in front of the camera")
    homogeneous_pixel = calibration @ camera_point
    return homogeneous_pixel[:2] / homogeneous_pixel[2], float(camera_point[2])


def triangulate_rectified(left_uv, right_uv, calibration, baseline_m):
    """Geometry diagnostic for ideal corresponding pixels, in the left optical frame."""
    left, right = np.asarray(left_uv, dtype=float), np.asarray(right_uv, dtype=float)
    if left.shape != (2,) or right.shape != (2,) or not np.isfinite([left, right]).all():
        raise ValueError("pixels must be finite 2D coordinates")
    if not math.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError("baseline must be finite and positive")
    disparity = left[0] - right[0]
    if disparity <= 1e-6:
        raise ValueError("disparity must be positive and larger than 1e-6 pixels")
    if abs(left[1] - right[1]) > 1e-6:
        raise ValueError("ideal rectified correspondences must lie on the same row")
    z = calibration[0, 0] * baseline_m / disparity
    return np.array([(left[0] - calibration[0, 2]) * z / calibration[0, 0],
                     (left[1] - calibration[1, 2]) * z / calibration[1, 1], z])
