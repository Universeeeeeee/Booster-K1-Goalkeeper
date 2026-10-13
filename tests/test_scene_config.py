import mujoco
import numpy as np
import pytest

from k1_goalkeeper.scene import build_scene, load_config


def test_configured_initial_states_reach_compiled_model(tmp_path):
    config = load_config()
    config["robot"]["position"] = [10, -.4, .6]
    config["robot"]["yaw_degrees"] = 90
    config["ball"]["position"] = [6, .5, .3]
    path = build_scene(tmp_path / "custom.xml", config=config)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body("trunk").xpos, [10, -.4, .6])
    np.testing.assert_allclose(data.body("ball").xpos, [6, .5, .3])
    optical_axis = data.body("head_booster_stereo_rgb_link").xmat.reshape(3, 3)[:, 2]
    np.testing.assert_allclose(optical_axis, [0, 1, 0], atol=1e-6)
    assert load_config()["robot"]["position"] == [10.4, 0, .55369]


@pytest.mark.parametrize("key,value", [("radius", 0), ("mass", -1), ("radius", float("nan"))])
def test_invalid_ball_parameters_fail_before_writing_scene(tmp_path, key, value):
    config = load_config()
    config["ball"][key] = value
    output = tmp_path / "invalid.xml"
    with pytest.raises(ValueError):
        build_scene(output, config=config)
    assert not output.exists()
