"""Verify the source manifest against independently loaded compiled MuJoCo fields."""
import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_isaac_source", PROJECT / "scripts/audit_isaac_source.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


@pytest.fixture(scope="module")
def contract():
    # Keep the generated scene under the project's results provenance boundary.
    import tempfile
    (PROJECT / "results/g1").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=PROJECT / "results/g1") as directory:
        scene = audit_module.build_scene(Path(directory) / "scene.xml", robot_mode="free")
        yield scene, audit_module.audit(scene)


def test_named_mapping(contract):
    _, report = contract
    robot = report["robot"]
    assert len(robot["robot_joint_order"]) == 22
    assert len(set(robot["policy_action_order"])) == 20
    assert [robot["robot_joint_order"][i] for i in robot["policy_to_robot_indices"]] == robot["policy_action_order"]
    assert robot["unactuated_by_policy"] == ["aahead_yaw_joint", "aahead_pitch_joint"]
    assert robot["free_base"]


def test_compiled_mass_and_ball_inertia(contract):
    scene, report = contract
    model = mujoco.MjModel.from_xml_path(str(scene))
    for body in report["bodies"]:
        assert body["mass_kg"] == model.body_mass[body["id"]]
        np.testing.assert_array_equal(body["inertia_principal_kg_m2"], model.body_inertia[body["id"]])
    ball = report["ball"]
    assert ball["mass_kg"] == pytest.approx(0.43)
    assert ball["radius_m"] == pytest.approx(0.11)
    np.testing.assert_allclose(ball["inertia_principal_kg_m2"], [2 / 3 * 0.43 * 0.11**2] * 3)
    for goal in report["goals"].values():
        assert goal["clear_width_m"] == pytest.approx(3.0)
        assert goal["clear_height_m"] == pytest.approx(2.0)


def test_named_axis_limits_and_motor_delivery(contract):
    scene, report = contract
    model = mujoco.MjModel.from_xml_path(str(scene))
    for row in report["robot"]["joints"]:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, row["name"])
        motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, row["name"])
        np.testing.assert_array_equal(row["axis_body"], model.jnt_axis[joint])
        np.testing.assert_array_equal(row["range_rad"], model.jnt_range[joint])
        np.testing.assert_array_equal(row["motor_force_range"], model.actuator_forcerange[motor])
        assert model.actuator_trnid[motor, 0] == joint
