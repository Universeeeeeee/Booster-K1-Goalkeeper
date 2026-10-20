import hashlib
import json

import mujoco
import numpy as np
import pytest
from PIL import Image

from conftest import ROOT
from k1_goalkeeper.scene import build_scene
from robot_perception_sim.camera import (
    CAMERA_NAMES, DEFAULT_CONFIG, intrinsics, load_camera_config, optical_pose,
    project_world, triangulate_rectified, vertical_fov_deg,
)
from robot_perception_sim.stereo import StereoCamera


def test_vendored_camera_asset_fingerprints():
    manifest = json.loads((ROOT / "models/sources.json").read_text())
    for path, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


def test_nominal_calibration(config):
    calibration = intrinsics(config)
    np.testing.assert_allclose(np.diag(calibration), [336.0996183406607] * 2 + [1])
    np.testing.assert_allclose(calibration[:2, 2], [479.5, 299.5])
    assert vertical_fov_deg(config) == pytest.approx(83.50372346447085)


@pytest.mark.parametrize("key,value", [("baseline_m", 0), ("baseline_m", float("nan")),
    ("horizontal_fov_deg", 180), ("image_width_px", 0), ("image_height_px", 1.5)])
def test_invalid_calibration(tmp_path, key, value):
    config = json.loads(DEFAULT_CONFIG.read_text())
    config[key] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_camera_config(path)


@pytest.mark.integration
@pytest.mark.parametrize("yaw,pitch", [(0, 0), (0.4, 0.3), (-0.5, -0.2)])
def test_rigid_mount_axes_and_baseline(model, config, yaw, pitch):
    data = mujoco.MjData(model)
    data.qpos[model.joint("aahead_yaw_joint").qposadr[0]] = yaw
    data.qpos[model.joint("aahead_pitch_joint").qposadr[0]] = pitch
    mujoco.mj_forward(model, data)
    left, right = [optical_pose(data, name) for name in CAMERA_NAMES]
    relative = np.linalg.inv(left) @ right
    np.testing.assert_allclose(relative[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(relative[:3, 3], [0.12, 0, 0], atol=1e-12)
    parent = data.body(config["mount"]["parent_body"])
    np.testing.assert_allclose(left[:3, :3], parent.xmat.reshape(3, 3), atol=1e-12)
    centre = (left[:3, 3] + right[:3, 3]) / 2
    np.testing.assert_allclose(parent.xmat.reshape(3, 3).T @ (centre - parent.xpos),
                               config["mount"]["position_parent_m"], atol=1e-12)
    # Independently derive the expected optical forward axis for the configured world yaw pi.
    expected_forward = [-np.cos(yaw) * np.cos(pitch),
                        -np.sin(yaw) * np.cos(pitch), -np.sin(pitch)]
    np.testing.assert_allclose(left[:3, 2], expected_forward, atol=1e-6)
    assert data.time == 0


@pytest.mark.integration
@pytest.mark.parametrize("point", [[0, 0, 3], [0.6, -0.4, 6], [-2, 1, 10]])
def test_projection_and_disparity(model, config, point):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    left, right = [optical_pose(data, name) for name in CAMERA_NAMES]
    point = np.asarray(point)
    world = left[:3, :3] @ point + left[:3, 3]
    calibration = intrinsics(config)
    uv_left, depth = project_world(world, calibration, left)
    uv_right, _ = project_world(world, calibration, right)
    assert depth == pytest.approx(point[2])
    assert uv_left[0] - uv_right[0] == pytest.approx(calibration[0, 0] * 0.12 / point[2])
    np.testing.assert_allclose(triangulate_rectified(uv_left, uv_right, calibration, 0.12), point, atol=1e-11)


@pytest.mark.parametrize("right", [[10, 10], [11, 10], [9, 12], [float("nan"), 10]])
def test_invalid_ideal_correspondence(config, right):
    with pytest.raises(ValueError):
        triangulate_rectified([10, 10], right, intrinsics(config), 0.12)


@pytest.mark.integration
def test_original_scene_dynamics_unchanged(model):
    baseline_path = build_scene(ROOT / "results/tests/original_scene.xml")
    original = mujoco.MjModel.from_xml_path(str(baseline_path))
    assert (model.nq, model.nv, model.nu, model.njnt) == (original.nq, original.nv, original.nu, original.njnt)
    housing = model.geom("zedx_housing_visual")
    assert housing.contype[0] == housing.conaffinity[0] == 0
    assert model.body("zedx_mount").mass[0] == 0
    for field in ("qpos0", "jnt_range", "jnt_axis", "dof_armature", "dof_damping",
                  "actuator_gear", "actuator_forcerange", "actuator_trnid", "sensor_type"):
        np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
    for index in range(1, original.nbody):
        name = original.body(index).name
        before, after = original.body(name), model.body(name)
        for field in ("mass", "inertia", "ipos", "iquat", "pos", "quat"):
            np.testing.assert_allclose(getattr(after, field), getattr(before, field), atol=1e-12)
    for index in range(original.ngeom):
        before = original.geom(index)
        after = model.geom(before.name)
        for field in ("size", "pos", "quat", "friction", "solref", "solimp", "contype", "conaffinity"):
            np.testing.assert_allclose(getattr(after, field), getattr(before, field), atol=1e-12)
    data_original, data_camera = mujoco.MjData(original), mujoco.MjData(model)
    mujoco.mj_forward(original, data_original)
    mujoco.mj_forward(model, data_camera)
    mass_original, mass_camera = np.empty((original.nv, original.nv)), np.empty((model.nv, model.nv))
    mujoco.mj_fullM(original, mass_original, data_original.qM)
    mujoco.mj_fullM(model, mass_camera, data_camera.qM)
    np.testing.assert_allclose(mass_camera, mass_original, atol=1e-12)
    np.testing.assert_allclose(data_camera.qfrc_bias, data_original.qfrc_bias, atol=1e-12)


@pytest.mark.render
def test_synchronized_rgb_and_ball_visibility(model, config):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos = data.qpos.copy()
    with StereoCamera(model, config) as camera:
        frame = camera.capture(data)
        renderer = camera.renderer
        renderer.enable_segmentation_rendering()
        for name in CAMERA_NAMES:
            renderer.update_scene(data, camera=name)
            segmentation = renderer.render()
            ball = ((segmentation[:, :, 0] == model.geom("ball_collision").id)
                    & (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM))
            assert ball.sum() > 20
            ys, xs = np.where(ball)
            predicted, _ = project_world(data.body("ball").xpos, intrinsics(config), optical_pose(data, name))
            assert np.linalg.norm([xs.mean() - predicted[0], ys.mean() - predicted[1]]) < 1
    for side in ("left", "right"):
        assert frame[side].shape == (600, 960, 3)
        assert frame[side].dtype == np.uint8
    np.testing.assert_array_equal(data.qpos, qpos)
    assert frame["timestamp_s"] == data.time == 0


@pytest.mark.render
@pytest.mark.parametrize("yaw,pitch", [(0, 0), (0.4, 0.3), (-0.5, -0.2)])
def test_analytic_projection_matches_rendered_markers(model, config, yaw, pitch):
    data = mujoco.MjData(model)
    data.qpos[model.joint("aahead_yaw_joint").qposadr[0]] = yaw
    data.qpos[model.joint("aahead_pitch_joint").qposadr[0]] = pitch
    mujoco.mj_forward(model, data)
    left_pose = optical_pose(data, CAMERA_NAMES[0])
    # Independent RGB marker measurement; no projection/triangulation helper finds the pixels.
    cases = [(0.173, -0.081, 2), (1.7, -0.5, 2.5), (-4.5, 1.0, 5), (0.2, 0.3, 10)]
    results = []
    output = ROOT / f"results/tests/projection_{yaw}_{pitch}"
    output.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=600, width=960) as renderer:
        for index, point in enumerate(cases):
            world = left_pose[:3, :3] @ point + left_pose[:3, 3]
            for name in CAMERA_NAMES:
                pose = optical_pose(data, name)
                renderer.update_scene(data, camera=name)
                # Isolate the projection experiment from ground/robot occlusion.
                # Visibility in the complete scene is tested separately above.
                renderer.scene.ngeom = 0
                marker = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_BOX,
                                   np.array([0.025, 0.025, 0.0001]), world,
                                   left_pose[:3, :3].ravel(), np.array([1, 0, 1, 1], dtype=np.float32))
                marker.emission = 1
                renderer.scene.ngeom += 1
                rgb = renderer.render()
                mask = (rgb[:, :, 0] > 180) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] > 180)
                ys, xs = np.where(mask)
                assert xs.size > 0, f"marker not visible at {point}, {name}"
                expected, _ = project_world(world, intrinsics(config), pose)
                measured = np.array([xs.mean(), ys.mean()])
                error = float(np.linalg.norm(measured - expected))
                Image.fromarray(rgb).save(output / f"{index}_{name}.png")
                results.append({"point_left_optical_m": list(point), "camera": name,
                                "expected_uv_px": expected.tolist(), "measured_uv_px": measured.tolist(),
                                "error_px": error})
                assert error < 1, (point, name, measured, expected)
    (output / "errors.json").write_text(json.dumps(results, indent=2) + "\n")
