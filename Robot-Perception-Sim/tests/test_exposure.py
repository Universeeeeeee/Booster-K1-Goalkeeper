from copy import deepcopy

import mujoco
import numpy as np
import pytest
from PIL import Image

from conftest import ROOT
from robot_perception_sim.stereo import StereoCamera


@pytest.mark.render
def test_static_exposure_matches_instantaneous_render(model, config):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    instant_config = deepcopy(config)
    instant_config["exposure_time_ms"] = 0
    with StereoCamera(model, instant_config) as camera:
        instant = camera.capture(data)
    with StereoCamera(model, config) as camera:
        exposed = camera.capture(data)
    assert exposed["exposure_end_s"] - exposed["exposure_start_s"] == pytest.approx(0.010)
    assert exposed["timestamp_s"] == exposed["exposure_start_s"] == data.time
    for side in ("left", "right"):
        np.testing.assert_array_equal(exposed[side], instant[side])


@pytest.mark.render
def test_exposure_blurs_moving_ball_without_stepping(config):
    config = {**config, "image_width_px": 320, "image_height_px": 200}
    model = mujoco.MjModel.from_xml_string('''
    <mujoco>
      <visual><global offwidth="320" offheight="200"/></visual>
      <asset><material name="red" rgba="1 0 0 1" emission="1"/></asset>
      <worldbody>
        <camera name="zedx_left_camera" pos="-0.06 0 0" quat="0 1 0 0" fovy="83.50372346447085"/>
        <camera name="zedx_right_camera" pos="0.06 0 0" quat="0 1 0 0" fovy="83.50372346447085"/>
        <body pos="0 0 2"><freejoint/><geom type="sphere" size="0.025" material="red"/></body>
      </worldbody>
    </mujoco>''')
    data = mujoco.MjData(model)
    data.time = 1
    data.qvel[0] = 20  # 20 m/s transverse motion; no new physics/control loop.
    mujoco.mj_forward(model, data)
    before = {field: getattr(data, field).copy() for field in ("qpos", "qvel", "qacc", "ctrl")}
    with StereoCamera(model, {**config, "exposure_time_ms": 0}) as camera:
        instant = camera.capture(data)
    with StereoCamera(model, config) as camera:
        exposed = camera.capture(data)
    output = ROOT / "results/tests/exposure"
    output.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        def bounds(rgb):
            ys, xs = np.where((rgb[:, :, 0] > 5) & (rgb[:, :, 1] < 5) & (rgb[:, :, 2] < 5))
            assert len(xs)
            return xs.min(), xs.max()
        x0, x1 = bounds(instant[side])
        b0, b1 = bounds(exposed[side])
        assert b1 - b0 > x1 - x0 + 6
        assert b1 > x1 + 6  # Trail follows the positive velocity.
        focal_px = config["image_width_px"] / (2 * np.tan(np.deg2rad(config["horizontal_fov_deg"]) / 2))
        expected_extension_px = focal_px * 20 * 0.010 / 2
        assert b1 - x1 == pytest.approx(expected_extension_px, abs=2)
        Image.fromarray(instant[side]).save(output / f"{side}_0ms.png")
        Image.fromarray(exposed[side]).save(output / f"{side}_10ms.png")
    assert exposed["exposure_start_s"] == 1
    assert exposed["exposure_end_s"] == pytest.approx(1.01)
    assert data.time == 1
    for field, expected in before.items():
        np.testing.assert_array_equal(getattr(data, field), expected)
