"""Passive measurements around the existing official K1 walking controller."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from .scene import K1_XML
from .walk_controller import K1WalkController


@dataclass(frozen=True)
class Segment:
    name: str
    duration: float
    command: tuple[float, float, float]

    def __post_init__(self):
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("segment duration must be positive and finite")
        if not np.isclose(self.duration / .02, round(self.duration / .02), rtol=0, atol=1e-8):
            raise ValueError("segment duration must be a multiple of the .02 s policy period")
        if len(self.command) != 3 or not np.isfinite(self.command).all():
            raise ValueError("command must contain three finite values")


def body_state(controller):
    q, v = controller.base_qpos_address, controller.base_dof_address
    data = controller.data
    position = data.qpos[q:q+3].copy()
    quat = data.qpos[q+3:q+7].copy()
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    rpy = np.array([np.arctan2(rotation[2, 1], rotation[2, 2]),
                    np.arcsin(np.clip(-rotation[2, 0], -1, 1)),
                    np.arctan2(rotation[1, 0], rotation[0, 0])])
    return {"base_position": position, "base_quaternion": quat,
            "body_linear_velocity": rotation.T @ data.qvel[v:v+3],
            "body_angular_velocity": data.qvel[v+3:v+6].copy(), "rpy": rpy}


def trajectory_hash(trajectory):
    digest = hashlib.sha256()
    for name in sorted(trajectory):
        array = np.ascontiguousarray(trajectory[name])
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def recovery_metrics(trajectory, velocity, start, end, completed):
    time, position, rpy = trajectory["time"], trajectory["base_position"], trajectory["rpy"]
    up_z = np.cos(rpy[:, 0]) * np.cos(rpy[:, 1])
    result = {"verified": False, "stop_delay_s": None}
    actual_end = min(end, time[-1])
    last = (time > max(start, actual_end-2) + 1e-10) & (time <= actual_end + 1e-10)
    if not last.any():
        return result
    average = velocity[last].mean(axis=0)
    result.update(mean_body_velocity=average.tolist(),
                  displacement_world=(position[np.searchsorted(time, actual_end+1e-9, side="right")-1]
                                      - position[min(len(time)-1, np.searchsorted(time, start-1e-9))]).tolist())
    # Require a whole second of consecutive, slow .2 s bins, rather than only a cancelling average.
    for bin_start in np.arange(start, actual_end-1+1e-9, .2):
        stopped = True
        for offset in range(5):
            mask = (time > bin_start+offset*.2+1e-10) & (time <= bin_start+(offset+1)*.2+1e-10)
            if not mask.any() or np.any(np.abs(velocity[mask].mean(axis=0)) > [.05, .05, .1]):
                stopped = False
                break
        if stopped:
            result["stop_delay_s"] = float(bin_start-start)
            break
    result["verified"] = bool(completed and end-start >= 3 and result["stop_delay_s"] is not None and
                              np.all(np.abs(average) <= [.05, .05, .1]) and
                              position[last, 2].min() >= .45 and
                              up_z[last].min() >= math.cos(math.radians(10)))
    return result


def summarize_case(name, segments, trajectory, physics_dt, initial_yaw=0, seed=0, error=None, fall_events=None):
    time = trajectory["time"]
    position = trajectory["base_position"]
    rpy = trajectory["rpy"]
    up_z = np.cos(rpy[:, 0]) * np.cos(rpy[:, 1])
    velocity = np.column_stack([trajectory["body_linear_velocity"][:, :2],
                                trajectory["body_angular_velocity"][:, 2]])
    # Both instantaneous errors and time-averaged bias are retained; gait oscillation is not hidden.
    phase_results = []
    start = 0.0
    for index, segment in enumerate(segments):
        end = start + segment.duration
        selected = trajectory["segment_index"] == index
        steady_start = start + min(2.0, segment.duration / 2)
        steady = selected & (time > steady_start + 1e-10)
        row = {**asdict(segment), "start_time_s": start, "end_time_s": end,
               "steady_window_start_s": steady_start,
               "completed": bool(time[-1] >= end - 1e-9),
               "physics_samples": int(selected.sum()), "steady_physics_samples": int(steady.sum())}
        if selected.any():
            before = max(0, int(np.flatnonzero(selected)[0]) - 1)
            after = int(np.flatnonzero(selected)[-1])
            row["displacement_world"] = (position[after] - position[before]).tolist()
            row["yaw_change_radians"] = float(np.unwrap(rpy[before:after+1, 2])[-1] - rpy[before, 2])
            row["torque_saturation_fraction"] = trajectory["saturated"][selected].mean(axis=0).tolist()
            row["any_saturation_step_fraction"] = float(trajectory["saturated"][selected].any(axis=1).mean())
            row["peak_raw_torque_nm"] = np.abs(trajectory["raw_torque"][selected]).max(axis=0).tolist()
            row["peak_motor_controls_nm"] = np.abs(trajectory["motor_controls"][selected]).max(axis=0).tolist()
            row["minimum_base_height_m"] = float(position[selected, 2].min())
            row["maximum_trunk_tilt_degrees"] = float(np.degrees(np.arccos(np.clip(up_z[selected], -1, 1))).max())
        if steady.any():
            measured = velocity[steady]
            command = np.asarray(segment.command)
            row["mean_body_velocity"] = measured.mean(axis=0).tolist()
            row["mean_tracking_error"] = (measured.mean(axis=0) - command).tolist()
            row["tracking_rmse"] = np.sqrt(np.mean((measured - command)**2, axis=0)).tolist()
            row["steady_torque_saturation_fraction"] = trajectory["saturated"][steady].mean(axis=0).tolist()
            first, last = np.flatnonzero(steady)[[0, -1]]
            first = max(0, int(first) - 1)
            elapsed = float(time[last] - time[first])
            displacement = position[last] - position[first]
            row["steady_displacement_world"] = displacement.tolist()
            row["finite_difference_world_velocity"] = (displacement / elapsed).tolist()
            row["finite_difference_yaw_rate"] = float((np.unwrap(rpy[first:last+1, 2])[-1] - rpy[first, 2]) / elapsed)
            fd_velocity = np.diff(position[first:last+1], axis=0) / np.diff(time[first:last+1])[:, None]
            rotations = np.empty((last-first, 3, 3))
            for i, quat in enumerate(trajectory["base_quaternion"][first+1:last+1]):
                mujoco.mju_quat2Mat(rotations[i].reshape(9), quat)
            mean_fd_body = np.einsum("nij,ni->nj", rotations, fd_velocity).mean(axis=0)
            row["finite_difference_body_linear_velocity"] = mean_fd_body.tolist()
            row["velocity_position_consistency_error"] = (mean_fd_body[:2] - measured.mean(axis=0)[:2]).tolist()
            world_angular = np.einsum("nij,nj->ni", rotations, trajectory["body_angular_velocity"][first+1:last+1])
            row["mean_world_vertical_angular_velocity"] = float(world_angular[:, 2].mean())
            active = np.abs(command) > 1e-12
            gains = np.divide(measured.mean(axis=0), command, out=np.zeros(3), where=active)
            row["response_gain"] = [float(gains[i]) if active[i] else None for i in range(3)]
            row["response_present"] = bool(np.all((gains[active] >= .5) & (gains[active] <= 1.5)))
            tolerance = np.maximum([.05, .05, .08], .3 * np.abs(command))
            # Zero-command cross-axis tolerances are deliberately tighter for translation.
            tolerance[~active] = np.asarray([.05, .05, .1])[~active]
            row["tracks_command"] = bool(np.all(np.abs(measured.mean(axis=0) - command) <= tolerance))
            row["tracking_tolerance"] = tolerance.tolist()
        else:
            row["response_present"] = row["tracks_command"] = False
        phase_results.append(row)
        start = end
    stable = error is None and not fall_events
    completed = bool(stable and time[-1] >= start - 1e-9)
    recovery = {"verified": False, "stop_delay_s": None}
    recoveries = []
    for index, row in enumerate(phase_results):
        if not any(row["command"]) and index > 0 and any(segments[index-1].command):
            metrics = recovery_metrics(trajectory, velocity, row["start_time_s"], row["end_time_s"], stable and row["completed"])
            recoveries.append({"segment": row["name"], **metrics})
            row["zero_recovery"] = metrics
            if index == len(segments)-1:
                recovery = metrics
    moving = [r for r in phase_results if np.any(np.asarray(r["command"]) != 0)]
    return {"name": name, "physics_dt": physics_dt, "initial_yaw": initial_yaw, "seed": seed,
            "requested_duration_s": start, "simulated_duration_s": float(time[-1]),
            "completed": completed, "error": error, "fall_events": fall_events or [],
            "minimum_base_height_m": float(position[:, 2].min()),
            "maximum_trunk_tilt_degrees": float(np.degrees(np.arccos(np.clip(up_z, -1, 1))).max()),
            "segments": phase_results, "zero_recovery": recovery, "zero_recoveries": recoveries,
            "mobility_usable": bool(completed and moving and recovery["verified"] and
                                    all(r["verified"] for r in recoveries) and
                                    all(r["response_present"] and r["tracks_command"] for r in moving)),
            "trajectory_sha256": trajectory_hash(trajectory)}


def run_case(name, segments, physics_dt=.002, initial_yaw=0, seed=0, output=None, simulation=None):
    if not segments or physics_dt not in [.001, .002] or not math.isfinite(initial_yaw):
        raise ValueError("provide segments, a .001/.002 timestep, and finite initial yaw")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if simulation is None:
        model = mujoco.MjModel.from_xml_path(str(K1_XML))
        model.opt.timestep = physics_dt
        data = mujoco.MjData(model)
        controller = K1WalkController(model, data)
    else:
        if simulation.controller is None or not np.isclose(simulation.model.opt.timestep, physics_dt):
            raise ValueError("scene recorder needs k1_walk and matching physics_dt")
        simulation.reset()
        model, data, controller = simulation.model, simulation.data, simulation.controller
    for segment in segments:
        vx, vy, yaw = segment.command
        cfg = controller.vel_command
        if not (-cfg.vx_backward_max <= vx <= cfg.vx_forward_max and abs(vy) <= cfg.vy_max and abs(yaw) <= cfg.vyaw_max):
            raise ValueError("velocity command exceeds official task bounds")
    if simulation is None:
        controller.initialize()
        q = controller.base_qpos_address
        data.qpos[q+3:q+7] = [math.cos(initial_yaw/2), 0, 0, math.sin(initial_yaw/2)]
        mujoco.mj_forward(model, data)
        controller.update_state()
        controller.policy.reset()
    else:
        initial_yaw = float(body_state(controller)["rpy"][2])
    keys = ["time", "segment_index", "command", "base_position", "base_quaternion",
            "body_linear_velocity", "body_angular_velocity", "rpy", "raw_torque",
            "motor_controls", "applied_torque", "saturated"]
    contacts = []
    if simulation is not None:
        keys += ["world_linear_velocity", "world_angular_velocity", "initial_body_position",
                 "initial_body_rotation", "goal_position", "goal_rotation", "ball_position",
                 "ball_velocity", "unexpected_contacts"]
        initial = body_state(controller)
        initial_rotation = np.empty(9)
        mujoco.mju_quat2Mat(initial_rotation, initial["base_quaternion"])
        initial_rotation = initial_rotation.reshape(3, 3)
        goal_id = model.body("goal_right").id
        goal_origin = data.xpos[goal_id].copy()
        goal_rotation = data.xmat[goal_id].reshape(3, 3).copy()
        ball_joint = model.joint("ball_free")
        ball_q, ball_v = int(ball_joint.qposadr[0]), int(ball_joint.dofadr[0])
        robot_id = model.body("trunk").id
        def category(body):
            while body:
                if body == robot_id:
                    return "robot"
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                if name == "ball":
                    return "ball"
                if name.startswith("goal_"):
                    return "goal"
                body = model.body_parentid[body]
            return "environment"
        categories = [category(body) for body in model.geom_bodyid]
    rows = {key: [] for key in keys}
    phase = [-1]
    command = np.zeros(3)
    targets = controller.robot.default_joint_pos.numpy().copy()
    limits = controller.robot.effort_limit.numpy()
    kp, kd = controller.robot.joint_stiffness.numpy(), controller.robot.joint_damping.numpy()
    falls = []

    def record(raw, control, applied):
        state = body_state(controller)
        values = {"time": float(data.time), "segment_index": phase[0], "command": command.copy(), **state,
                  "raw_torque": raw.copy(), "motor_controls": control.copy(), "applied_torque": applied.copy(),
                  "saturated": np.abs(raw) >= limits - 1e-6}
        if simulation is not None:
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, state["base_quaternion"])
            rotation = rotation.reshape(3, 3)
            flags = np.zeros(3, dtype=bool)
            pairs = [("ball", "robot"), ("goal", "robot"), ("ball", "goal")]
            for contact in data.contact[:data.ncon]:
                pair = tuple(sorted([categories[contact.geom1], categories[contact.geom2]]))
                if pair in pairs and contact.dist <= 0:
                    index = pairs.index(pair)
                    flags[index] = True
                    if not any(event["pair"] == list(pair) for event in contacts):
                        contacts.append({"time": float(data.time), "pair": list(pair),
                                         "geoms": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                                                   for g in [contact.geom1, contact.geom2]]})
            v = controller.base_dof_address
            values.update(world_linear_velocity=data.qvel[v:v+3].copy(),
                          world_angular_velocity=rotation @ state["body_angular_velocity"],
                          initial_body_position=initial_rotation.T @ (state["base_position"]-initial["base_position"]),
                          initial_body_rotation=initial_rotation.T @ rotation,
                          goal_position=goal_rotation.T @ (state["base_position"]-goal_origin),
                          goal_rotation=goal_rotation.T @ rotation,
                          ball_position=data.qpos[ball_q:ball_q+3].copy(),
                          ball_velocity=data.qvel[ball_v:ball_v+6].copy(), unexpected_contacts=flags)
        for key in keys:
            rows[key].append(values[key])
        return state

    record(np.zeros(22), np.zeros(22), np.zeros(22))
    native_step = mujoco.mj_step
    native_control = controller.ctrl_step

    def observe_control(dof_targets):
        nonlocal targets
        targets = dof_targets.detach().cpu().numpy().copy()
        native_control(dof_targets)

    def observe_step(model, data):
        nonlocal targets
        if simulation is not None:
            targets = simulation.targets.detach().cpu().numpy()
        # This formula only measures pre-limit demand; the unchanged controller computes/applies PD.
        raw = kp * (targets - data.qpos.astype(np.float32)[controller.qpos_addresses]) - kd * data.qvel.astype(np.float32)[controller.dof_addresses]
        control = data.ctrl[controller.actuator_ids].astype(np.float32).copy()
        native_step(model, data)
        controller._check_simulation()
        state = record(raw, control, data.qfrc_actuator[controller.dof_addresses].astype(np.float32))
        if contacts:
            raise RuntimeError("unexpected scene contact; stage D stops before impact testing")
        roll, pitch = state["rpy"][:2]
        if state["base_position"][2] < .3 or math.cos(roll)*math.cos(pitch) < .5:
            falls.append({"time": float(data.time), "segment": segments[phase[0]].name,
                          "base_height_m": float(state["base_position"][2]),
                          "tilt_degrees": float(np.degrees(np.arccos(np.clip(math.cos(roll)*math.cos(pitch), -1, 1))))})
            raise RuntimeError("fall detected from free-base height/pose")

    error = None
    try:
        with patch.object(controller, "ctrl_step", observe_control), patch.object(mujoco, "mj_step", observe_step):
            for index, segment in enumerate(segments):
                phase[0] = index
                command = np.asarray(segment.command, dtype=float)
                controller.vel_command.lin_vel_x, controller.vel_command.lin_vel_y, controller.vel_command.ang_vel_yaw = command
                for _ in range(round(segment.duration / controller.cfg.policy_dt)):
                    if simulation is None:
                        controller.step()
                    else:
                        simulation.advance_policy_step()
    except RuntimeError as exc:
        error = str(exc)
    finally:
        controller.stop()
    trajectory = {key: np.asarray(value, dtype=np.float32 if key in ["raw_torque", "motor_controls", "applied_torque"] else None)
                  for key, value in rows.items()}
    summary = summarize_case(name, segments, trajectory, physics_dt, initial_yaw, seed, error, falls)
    if simulation is not None:
        summary["scene"] = {"control_mode": simulation.control_mode,
                            "physics_steps": simulation.physics_steps, "policy_steps": simulation.policy_steps,
                            "unexpected_contact_events": contacts,
                            "goal_origin_world": goal_origin.tolist(), "goal_rotation_world": goal_rotation.tolist(),
                            "initial_body_rotation_world": initial_rotation.tolist(),
                            "ball_maximum_horizontal_displacement_m": float(np.linalg.norm(
                                trajectory["ball_position"][:, :2]-trajectory["ball_position"][0, :2], axis=1).max()),
                            "ball_actuator_force_final_maximum": float(np.abs(data.qfrc_actuator[ball_v:ball_v+6]).max())}
    summary["joint_names"] = controller.cfg.robot.joint_names
    summary["effort_limits_nm"] = limits.tolist()
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "trajectory.npz", **trajectory)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary, trajectory
