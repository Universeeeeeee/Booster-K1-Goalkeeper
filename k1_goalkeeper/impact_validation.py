"""Stage E measurements using the existing scene loop and official controller."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .detection import GoalDetector
from .scene_simulation import SceneSimulation
from .simulation import Shot


@dataclass(frozen=True)
class ImpactCase:
    name: str
    target_body: str = "trunk"
    speed: float = 2.
    offset_y: float = 0.
    robot_mode: str = "free"
    command: tuple = (0., 0., 0.)
    moving_duration: float = 0.
    miss: bool = False
    duration: float = 6.
    seed: int = 0
    target_geom_index: int = 0

    def __post_init__(self):
        if self.robot_mode not in ("free", "fixed", "absent"):
            raise ValueError("robot_mode must be free/fixed/absent")
        if not np.isfinite([self.speed, self.offset_y, self.moving_duration, self.duration]).all() or self.speed <= 0 or self.duration <= 0 or self.moving_duration < 0:
            raise ValueError("speed/duration must be positive finite; moving_duration nonnegative")
        if len(self.command) != 3 or not np.isfinite(self.command).all():
            raise ValueError("command must contain three finite values")
        if self.robot_mode != "free" and (any(self.command) or self.moving_duration):
            raise ValueError("fixed/absent controls cannot move K1")
        if self.target_geom_index < 0:
            raise ValueError("target_geom_index must be nonnegative")


def ball_contact_force_world(contact, force6, ball_geom):
    """Contact force on the ball, with contact basis stored as matrix rows."""
    if ball_geom not in (contact.geom1, contact.geom2):
        raise ValueError("contact does not contain the ball")
    sign = 1 if contact.geom2 == ball_geom else -1
    return sign * np.asarray(contact.frame).reshape(3, 3).T @ np.asarray(force6)[:3]


def _robot_body(model, body):
    while body:
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body)) == "trunk":
            return True
        body = model.body_parentid[body]
    return False


def _launch(model, data, case, ball_geom):
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, case.target_body)
    if body < 0:
        raise ValueError("target body unavailable; absent control requires an exact launch override")
    candidates = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and
                  ((model.geom_contype[g] & model.geom_conaffinity[ball_geom]) or
                   (model.geom_contype[ball_geom] & model.geom_conaffinity[g]))]
    if case.target_geom_index >= len(candidates):
        raise ValueError(f"{case.target_body}: no selected effective collision geometry; not verifiable")
    geom = candidates[case.target_geom_index]
    center = data.geom_xpos[geom].copy()
    rotation = data.geom_xmat[geom].reshape(3, 3)
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(rotation[0]) @ model.geom_size[geom]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        axis_x = rotation[0, 2]
        extent = model.geom_size[geom, 0]*np.sqrt(max(0, 1-axis_x**2)) + model.geom_size[geom, 1]*abs(axis_x)
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        extent = -float((model.mesh_vert[start:start+count] @ rotation.T)[:, 0].min())
    else:
        raise ValueError("protocol supports existing K1 box/cylinder/mesh colliders")
    radius = model.geom_size[ball_geom, 0]
    position = center.copy()
    position[0] -= extent + radius + .3
    position[1] += case.offset_y + (.8 if case.miss else 0)
    position[2] = max(position[2], radius+.001)
    rolling = case.target_body.endswith("ankle_roll_link")
    velocity = [case.speed, 0, 0 if rolling else -model.opt.gravity[2]*.3/(2*case.speed)]
    spin = [0, case.speed/radius if rolling else 0, 0]
    return {"position": position.tolist(), "velocity": velocity, "angular_velocity": spin,
            "target_geom_id": int(geom), "target_body": case.target_body,
            "target_geom_center_world": center.tolist(), "nominal_free_flight_s": .3/case.speed,
            "construction": "0.3m gap from collider front plus ball radius; ballistic aim or rolling foot input"}


def run_impact(model, case, output=None, launch=None, command_provider=None, step_observer=None):
    """Settle, launch once, observe; optional F hooks leave default E behavior intact.

    Commands use the current scene before each step; the observer runs after it.
    Launch follows command selection, so a 50 Hz hook first reacts to a t=2 shot
    at t=2.02, without seeing future state.
    """
    from .velocity_validation import body_state, recovery_metrics, trajectory_hash
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(case.seed)
    np.random.seed(case.seed)
    if not np.isclose(model.opt.timestep, .001, rtol=0, atol=1e-12):
        raise ValueError("impact validation requires 1 ms physics timestep")
    sim = SceneSimulation(model, control_mode="k1_walk" if case.robot_mode == "free" else "zero_torque")
    data, controller = sim.data, sim.controller
    robot_exists = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk") >= 0
    if robot_exists != (case.robot_mode != "absent") or (case.robot_mode == "fixed" and model.nu):
        raise ValueError("case robot_mode does not match compiled model")
    ball_geom = model.geom("ball_collision").id
    joint = model.joint("ball_free")
    q, v = int(joint.qposadr[0]), int(joint.dofadr[0])
    dt = float(model.opt.timestep)
    warnings = [mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL]
    counts = np.array([data.warning[w].number for w in warnings])
    detector = GoalDetector(radius=float(model.geom_size[ball_geom, 0]))
    keys = ["time", "ball_position", "ball_velocity", "ball_angular_velocity_body", "base_position",
            "base_quaternion", "body_linear_velocity", "body_angular_velocity", "rpy", "qpos", "qvel",
            "raw_torque", "motor_controls", "applied_torque", "saturated", "command",
            "ball_contact_force_world", "robot_contact_force_world", "momentum_residual_ns"]
    rows = {k: [] for k in keys}
    contact_rows, episodes, active = [], [], {}
    falls, numerical_error, policy_stop = [], None, None
    current_command = np.zeros(3)
    launch_time = 2 + case.moving_duration
    launch_step = math.ceil(launch_time/dt)
    launch_time = launch_step*dt
    stop_time = launch_time + (1 if any(case.command) else 0)
    end_time = launch_time + case.duration
    launch_info = None
    limits = controller.robot.effort_limit.numpy() if controller else np.empty(0)

    def sample(raw, controls, force, robot_force, residual):
        if controller:
            state = body_state(controller)
        elif robot_exists:
            rotation = data.body("trunk").xmat.reshape(3, 3)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rotation.reshape(9))
            state = {"base_position": data.body("trunk").xpos.copy(), "base_quaternion": quat,
                     "body_linear_velocity": np.zeros(3), "body_angular_velocity": np.zeros(3),
                     "rpy": np.array([np.arctan2(rotation[2, 1], rotation[2, 2]),
                                      np.arcsin(np.clip(-rotation[2, 0], -1, 1)), np.arctan2(rotation[1, 0], rotation[0, 0])])}
        else:
            state = {k: np.empty(0) for k in ["base_position", "base_quaternion", "body_linear_velocity", "body_angular_velocity", "rpy"]}
        values = {"time": float(data.time), "ball_position": data.qpos[q:q+3].copy(),
                  "ball_velocity": data.qvel[v:v+3].copy(), "ball_angular_velocity_body": data.qvel[v+3:v+6].copy(),
                  **state, "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "raw_torque": raw,
                  "motor_controls": controls, "applied_torque": data.qfrc_actuator[controller.dof_addresses].copy() if controller else np.empty(0),
                  "saturated": np.abs(raw) >= limits-1e-6, "command": current_command.copy(),
                  "ball_contact_force_world": force, "robot_contact_force_world": robot_force,
                  "momentum_residual_ns": residual}
        for k in keys:
            rows[k].append(values[k])
        if controller:
            z, (roll, pitch) = state["base_position"][2], state["rpy"][:2]
            tilt = float(np.degrees(np.arccos(np.clip(math.cos(roll)*math.cos(pitch), -1, 1))))
            if (z < .3 or tilt > 60) and not falls:
                falls.append({"time": float(data.time), "height_m": float(z), "tilt_degrees": tilt})
        return state

    sample(np.zeros(len(limits)), np.zeros(len(limits)), np.zeros(3), np.zeros(3), np.zeros(3))
    try:
        for step in range(math.ceil(end_time/dt)):
            if controller:
                current_command = np.asarray(case.command if 2 <= data.time+1e-9 < stop_time else (0, 0, 0), dtype=float)
                if command_provider is not None:
                    current_command = np.asarray(command_provider(sim), dtype=float)
                sim.set_velocity(current_command)
            if step == launch_step:
                mujoco.mj_kinematics(model, data)
                launch_info = _launch(model, data, case, ball_geom) if launch is None else dict(launch)
                shot = Shot(case.name, tuple(launch_info["position"]), tuple(launch_info["velocity"]),
                            tuple(launch_info["angular_velocity"]), case.duration, case.seed)
                data.qpos[q:q+7] = (*shot.position, 1, 0, 0, 0)
                data.qvel[v:v+6] = (*shot.velocity, *shot.angular_velocity)
                mujoco.mj_forward(model, data)
                launch_info["time"] = float(data.time)
                launch_info["initial_overlap"] = bool(any(c.dist < 0 and ball_geom in (c.geom1, c.geom2)
                                                          and _robot_body(model, model.geom_bodyid[c.geom2 if c.geom1 == ball_geom else c.geom1])
                                                          for c in data.contact[:data.ncon]))
                if launch_info["initial_overlap"]:
                    raise ValueError("shot must start without robot overlap")
                detector.update(data.time, data.qpos[q:q+3])
                # The row at launch time becomes the delivered launch state, not the parked ball.
                rows["ball_position"][-1] = data.qpos[q:q+3].copy()
                rows["ball_velocity"][-1] = data.qvel[v:v+3].copy()
                rows["ball_angular_velocity_body"][-1] = data.qvel[v+3:v+6].copy()
                rows["qpos"][-1] = data.qpos.copy()
                rows["qvel"][-1] = data.qvel.copy()
            before_time, before_velocity = float(data.time), data.qvel[v:v+3].copy()
            before_joints = data.qpos[controller.qpos_addresses].astype(np.float32).copy() if controller else np.empty(0)
            before_velocities = data.qvel[controller.dof_addresses].astype(np.float32).copy() if controller else np.empty(0)
            sim.step()
            if step_observer is not None:
                step_observer(sim)
            if not np.isclose(data.time, before_time+dt, rtol=0, atol=1e-9) or not all(np.isfinite(x).all() for x in [data.qpos, data.qvel, data.qacc, data.ctrl]) or not np.array_equal(counts, [data.warning[w].number for w in warnings]):
                raise RuntimeError("MuJoCo nonfinite state, BAD warning, or unexpected time reset")
            raw = (controller.robot.joint_stiffness.numpy()*(sim.targets.numpy()-before_joints)
                   - controller.robot.joint_damping.numpy()*before_velocities) if controller else np.empty(0)
            controls = data.ctrl[controller.actuator_ids].copy() if controller else np.empty(0)
            total_force, robot_force, pairs = np.zeros(3), np.zeros(3), {}
            if launch_info is not None:
                for index, contact in enumerate(data.contact[:data.ncon]):
                    if contact.efc_address < 0 or ball_geom not in (contact.geom1, contact.geom2):
                        continue
                    force6 = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, force6)
                    if force6[0] <= 1e-6:
                        continue
                    other = int(contact.geom2 if contact.geom1 == ball_geom else contact.geom1)
                    body = int(model.geom_bodyid[other])
                    kind = "robot" if _robot_body(model, body) else ("ground" if body == 0 else "goal")
                    force = ball_contact_force_world(contact, force6, ball_geom)
                    total_force += force
                    if kind == "robot":
                        robot_force += force
                    contact_rows.append([before_time, float(data.time), other, body, *contact.pos, *force,
                                         force6[0], *force6[3:], float(contact.dist)])
                    if other not in pairs:
                        pairs[other] = {"kind": kind, "force": np.zeros(3), "normal": 0., "peak": 0.}
                    pairs[other]["force"] += force
                    pairs[other]["normal"] += force6[0]
                    pairs[other]["peak"] += np.linalg.norm(force)
                for other, contact in pairs.items():
                    if other not in active:
                        event = {"kind": contact["kind"], "geom_id": other,
                                 "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                                 "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])),
                                 "start_time": before_time, "end_time": float(data.time), "duration_s": 0.,
                                 "impulse_on_ball_world_ns": [0., 0., 0.], "normal_impulse_ns": 0., "peak_force_n": 0.,
                                 "ball_velocity_before": before_velocity.tolist(), "ball_velocity_after": data.qvel[v:v+3].tolist()}
                        episodes.append(event)
                        active[other] = event
                    event = active[other]
                    event["end_time"] = float(data.time)
                    event["duration_s"] += dt
                    event["normal_impulse_ns"] += float(contact["normal"]*dt)
                    event["impulse_on_ball_world_ns"] = (np.asarray(event["impulse_on_ball_world_ns"])+contact["force"]*dt).tolist()
                    event["peak_force_n"] = max(event["peak_force_n"], float(contact["peak"]))
                    event["ball_velocity_after"] = data.qvel[v:v+3].tolist()
                active = {g: event for g, event in active.items() if g in pairs}
                detector.update(data.time, data.qpos[q:q+3])
            mass = model.body_mass[model.geom_bodyid[ball_geom]]
            residual = mass*(data.qvel[v:v+3]-before_velocity)-(total_force+mass*model.opt.gravity)*dt if launch_info else np.zeros(3)
            sample(raw, controls, total_force, robot_force, residual)
            if falls:
                break  # Record a physical fall as an experimental result; do not change the policy.
    except RuntimeError as exc:
        message = str(exc)
        if controller is not None and "safety fallback" in message:
            policy_stop = message
        else:
            numerical_error = {"message": message, "last_valid_time": float(rows["time"][-1]),
                               "warning_counts": {w.name: int(data.warning[w].number) for w in warnings}}
    finally:
        if controller:
            controller.stop()
    trajectory = {k: np.asarray(value) for k, value in rows.items()}
    trajectory["contacts"] = np.asarray(contact_rows, dtype=float).reshape(-1, 15)
    robot_contacts = [c for c in episodes if c["kind"] == "robot"]
    after = trajectory["time"] >= launch_time-1e-9
    completed = bool(trajectory["time"][-1] >= end_time-1e-9)
    recovery = None
    maintained = None
    preimpact = None
    if controller:
        launch_index = min(len(trajectory["time"])-1, np.searchsorted(trajectory["time"], launch_time-1e-9))
        preimpact = {"base_position_world": trajectory["base_position"][launch_index].tolist(),
                     "body_linear_velocity": trajectory["body_linear_velocity"][launch_index].tolist(),
                     "body_angular_velocity": trajectory["body_angular_velocity"][launch_index].tolist(),
                     "displacement_since_initial_world": (trajectory["base_position"][launch_index]-trajectory["base_position"][0]).tolist()}
    if controller:
        velocity = np.column_stack([trajectory["body_linear_velocity"][:, :2], trajectory["body_angular_velocity"][:, 2]])
        last_contact = max((c["end_time"] for c in robot_contacts), default=launch_time)
        recovery = recovery_metrics(trajectory, velocity, max(last_contact, stop_time), end_time,
                                    completed and numerical_error is None and not falls and policy_stop is None)
        tilts = np.degrees(np.arccos(np.clip(np.cos(trajectory["rpy"][:, 0])*np.cos(trajectory["rpy"][:, 1]), -1, 1)))
        maintained = bool(after.any() and not falls and not numerical_error and not policy_stop and completed
                          and trajectory["base_position"][after, 2].min() >= .45 and tilts[after].max() <= 10)
    event = detector.event
    summary = {"case": asdict(case), "launch": launch_info, "preimpact_robot": preimpact, "physics_dt": dt, "policy_dt": .02 if controller else None,
               "physics_steps": sim.physics_steps, "policy_steps": sim.policy_steps, "completed": completed,
               "simulated_duration_s": float(trajectory["time"][-1]), "contacts": episodes,
               "effective_contact": bool(robot_contacts), "fall_events": falls, "numerical_error": numerical_error,
               "policy_stop": policy_stop, "recovery": recovery, "balance_maintained": maintained,
               "balance_recovered": (bool(robot_contacts and recovery["verified"]) if controller else None),
               "goal": bool(event.outcome == "Goal") if event else (False if completed else None),
               "exit_event": event.to_dict() if event else None,
               "maximum_momentum_residual_ns": float(np.abs(trajectory["momentum_residual_ns"]).max()),
               "joint_names": controller.cfg.robot.joint_names if controller else [],
               "effort_limits_nm": limits.tolist(),
               "trajectory_sha256": trajectory_hash(trajectory)}
    if controller:
        summary.update(minimum_base_height_m=float(trajectory["base_position"][:, 2].min()),
                       maximum_tilt_degrees=float(tilts.max()),
                       peak_body_angular_velocity_rad_s=np.abs(trajectory["body_angular_velocity"]).max(axis=0).tolist(),
                       torque_saturation_fraction=trajectory["saturated"][after].mean(axis=0).tolist() if after.any() else [],
                       peak_raw_torque_nm=np.abs(trajectory["raw_torque"]).max(axis=0).tolist())
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/"trajectory.npz", **trajectory)
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary, trajectory
