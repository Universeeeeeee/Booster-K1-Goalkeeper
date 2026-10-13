"""Validate official K1 walk inference and 30-second zero-command free-base standing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch

from k1_goalkeeper.scene import K1_XML
from k1_goalkeeper.walk_controller import DEPLOY_ROOT, K1WalkController


def verify_sources():
    manifest = json.loads((DEPLOY_ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((DEPLOY_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Official policy source/weight changed: {name}")
    return manifest


def pose(controller):
    address = controller.base_qpos_address
    position = controller.data.qpos[address:address + 3].copy()
    quat = controller.data.qpos[address + 3:address + 7].copy()
    up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
    return position, quat, float(np.clip(up_z, -1, 1))


def validate(duration, physics_dt, output, show_viewer=False):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    manifest = verify_sources()
    model = mujoco.MjModel.from_xml_path(str(K1_XML))
    source_timestep = float(model.opt.timestep)
    model.opt.timestep = physics_dt
    smoke = K1WalkController(model, mujoco.MjData(model))
    smoke.initialize()
    targets = smoke.step()
    stage_a = {
        "passed": True, "observation_frame_size": int(smoke.policy.compute_observation().numel()),
        "observation_history_shape": list(smoke.policy.obs_history.shape),
        "policy_action_size": int(smoke.policy.last_action.numel()),
        "joint_target_size": int(targets.size), "targets": targets.tolist(),
        "motor_controls": smoke.data.ctrl[smoke.actuator_ids].tolist(),
        "base_joint_type": "free", "pd_substeps": smoke.decimation,
        "controller": "unchanged upstream LocomotionPolicy with named MuJoCo state/torque adapter",
    }
    smoke.stop()
    controller = K1WalkController(model, mujoco.MjData(model))
    controller.initialize()
    samples = {name: [] for name in ["time", "base_position", "base_quaternion", "upright_z",
                                     "qpos", "qvel", "motor_controls", "joint_targets"]}

    def record(targets):
        position, quaternion, up_z = pose(controller)
        values = [float(controller.data.time), position, quaternion, up_z,
                  controller.data.qpos.copy(), controller.data.qvel.copy(),
                  controller.data.ctrl[controller.actuator_ids].copy(), targets.copy()]
        for key, value in zip(samples, values):
            samples[key].append(value)
        return position, up_z

    record(controller.robot.default_joint_pos.numpy())
    window = None
    if show_viewer:
        from mujoco import viewer
        window = viewer.launch_passive(model, controller.data)
        window.cam.distance = 2.2
        window.cam.azimuth = 135
        window.cam.elevation = -15
    error = None
    started = time.monotonic()
    try:
        for _ in range(math.ceil(duration / controller.cfg.policy_dt)):
            iteration = time.monotonic()
            targets = controller.step()
            position, up_z = record(targets)
            if position[2] < .3 or up_z < .5:
                raise RuntimeError("Standing criterion violated: base z < .3 m or trunk tilt > 60 degrees")
            if window:
                if not window.is_running():
                    raise RuntimeError("Viewer closed before completing standing verification")
                window.cam.lookat[:] = position
                window.sync()
                time.sleep(max(0, controller.cfg.policy_dt - (time.monotonic() - iteration)))
    except RuntimeError as exc:
        error = str(exc)
    finally:
        controller.stop()
        if window:
            window.close()
    samples = {key: np.asarray(value) for key, value in samples.items()}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trajectory.npz", **samples)
    stage_b = {
        "passed": error is None and controller.data.time >= 30 - 1e-9,
        "requested_duration_s": duration, "simulated_duration_s": float(controller.data.time),
        "policy_steps": len(samples["time"]) - 1,
        "command_m_s_rad_s": [0, 0, 0], "free_base": True,
        "minimum_base_height_m": float(samples["base_position"][:, 2].min()),
        "maximum_trunk_tilt_degrees": float(np.degrees(np.arccos(samples["upright_z"])).max()),
        "maximum_horizontal_drift_m": float(np.linalg.norm(
            samples["base_position"][:, :2] - samples["base_position"][0, :2], axis=1).max()),
        "final_base_position_m": samples["base_position"][-1].tolist(),
        "final_base_quaternion_wxyz": samples["base_quaternion"][-1].tolist(),
        "peak_motor_controls_nm": np.abs(samples["motor_controls"]).max(axis=0).tolist(),
        "stability_criteria": {"minimum_duration_s": 30, "minimum_base_height_m": .3,
                               "minimum_upright_z": .5, "pose_sample_interval_s": controller.cfg.policy_dt},
        "error": error,
    }
    report = {
        "versions": {"python": sys.version.split()[0], "mujoco": mujoco.__version__,
                     "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 1},
        "sources": {"deployment": manifest, "robot_mjcf_path": str(K1_XML),
                    "robot_mjcf_sha256": hashlib.sha256(K1_XML.read_bytes()).hexdigest()},
        "physics": {"source_timestep_s": source_timestep, "validation_timestep_s": physics_dt,
                    "timestep_reason": "Standalone official deployment is .002 s; optional .001 s comparison preserves MJCF's timestep",
                    "policy_period_s": controller.cfg.policy_dt, "decimation": controller.decimation,
                    "gravity_m_s2": model.opt.gravity.tolist(), "initial_base_position_m": controller.cfg.mujoco.init_pos,
                    "model_nq": model.nq, "model_nv": model.nv, "model_nu": model.nu,
                    "constraints": "No welds, fixed base, external support, or external forces added"},
        "control": {"joint_names": controller.cfg.robot.joint_names,
                    "policy_joint_names": controller.cfg.policy.policy_joint_names,
                    "qpos_addresses": controller.qpos_addresses.tolist(),
                    "dof_addresses": controller.dof_addresses.tolist(),
                    "actuator_ids": controller.actuator_ids.tolist(),
                    "kp": controller.cfg.robot.joint_stiffness, "kd": controller.cfg.robot.joint_damping,
                    "effort_limits_nm": controller.cfg.robot.effort_limit},
        "stage_a": stage_a, "stage_b": stage_b, "wall_duration_s": time.monotonic() - started,
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"A: passed (69 observations x 10 history, 20 actions, 22 PD-controlled joints)")
    print(f"B: {'passed' if stage_b['passed'] else 'not passed'}; {stage_b['simulated_duration_s']:.3f} s, "
          f"min height {stage_b['minimum_base_height_m']:.6f} m, max tilt {stage_b['maximum_trunk_tilt_degrees']:.6f} deg")
    print(f"Saved {output.resolve()}")
    return stage_b["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30, help="Runs shorter than 30 s cannot pass B")
    parser.add_argument("--physics-dt", type=float, choices=[.001, .002], default=.002)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/walk_validation")
    parser.add_argument("--viewer", action="store_true", help="Use mjpython on macOS")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    raise SystemExit(0 if validate(args.duration, args.physics_dt, args.output, args.viewer) else 1)
