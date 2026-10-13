"""Repeatable shot initialization and MuJoCo contact/trajectory recording."""

from dataclasses import asdict, dataclass
import math
import random

import mujoco
import numpy as np

from .detection import GoalDetector


@dataclass(frozen=True)
class Shot:
    name: str = "custom"
    position: tuple[float, float, float] = (7.0, 0.0, 0.111)
    velocity: tuple[float, float, float] = (8.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration: float = 3.0
    seed: int = 0

    def __post_init__(self):
        for vector in (self.position, self.velocity, self.angular_velocity):
            if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
                raise ValueError("Position and velocities must be three finite numbers")
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Duration must be positive and finite")


def preset_shots(seed=0, duration=3.0):
    """Test inputs; collision/outcome depends on selected robot mode and physics."""
    return {
        "center_low": Shot("center_low", (7, 0, 0.111), (8, 0, 0), (0, 8 / 0.11, 0), duration, seed),
        "left": Shot("left", (7, -1, 0.111), (8, 0, 0), (0, 8 / 0.11, 0), duration, seed),
        "right": Shot("right", (7, 1, 0.111), (8, 0, 0), (0, 8 / 0.11, 0), duration, seed),
        "high": Shot("high", (8, 0, 1), (8, 0, 4), duration=duration, seed=seed),
        "near_post": Shot("near_post", (8, 1.37, 0.111), (8, 0, 0), (0, 8 / 0.11, 0), duration, seed),
        "miss_left": Shot("miss_left", (8, -2, 0.111), (8, 0, 0), duration=duration, seed=seed),
        "miss_right": Shot("miss_right", (8, 2, 0.111), (8, 0, 0), duration=duration, seed=seed),
        "over_crossbar": Shot("over_crossbar", (8, 0, 2.7), (8, 0, 1), duration=duration, seed=seed),
        "post": Shot("post", (9, 1.54, 0.5), (8, 0, 0), duration=duration, seed=seed),
        "crossbar": Shot("crossbar", (9, 0, 2.04), (8, 0, 1.22625), duration=duration, seed=seed),
    }


def random_shots(count, seed=0, duration=3.0):
    rng = random.Random(seed)
    return [Shot(f"random_{i:04d}", (rng.uniform(5, 8), rng.uniform(-3, 3), rng.uniform(0.111, 1)),
                 (rng.uniform(6, 12), rng.uniform(-3, 3), rng.uniform(0, 4)),
                 tuple(rng.uniform(-20, 20) for _ in range(3)), duration, seed)
            for i in range(count)]


def run_shot(model, shot, robot_mode="free"):
    """Reset to the scene's initial pose and simulate without a standing controller."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_collision")
    if joint < 0 or geom < 0:
        raise ValueError("Scene must contain ball_free and ball_collision")
    qpos = int(model.jnt_qposadr[joint])
    dof = int(model.jnt_dofadr[joint])
    ball_body = int(model.geom_bodyid[geom])
    data.qpos[qpos:qpos + 7] = (*shot.position, 1, 0, 0, 0)
    data.qvel[dof:dof + 6] = (*shot.velocity, *shot.angular_velocity)
    data.ctrl[:] = 0

    def check_state():
        for warning in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQACC):
            if data.warning[warning].number:
                raise RuntimeError(f"MuJoCo unstable state: {warning.name}")
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("MuJoCo produced a non-finite state")

    mujoco.mj_forward(model, data)
    check_state()
    detector = GoalDetector(radius=float(model.geom_size[geom, 0]))

    def sample():
        return {"time": float(data.time), "position": data.qpos[qpos:qpos + 3].tolist(),
                "velocity": data.qvel[dof:dof + 3].tolist(),
                "angular_velocity_body": data.qvel[dof + 3:dof + 6].tolist()}

    def state():
        return {**sample(), "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
                "ctrl": data.ctrl.tolist()}

    goal_roots = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                  for name in ("goal_left", "goal_right")}

    def classify(body):
        initial_body = body
        while body > 0:
            if body in goal_roots:
                return "goal"
            if body == ball_body:
                return "ball"
            body = int(model.body_parentid[body])
        return "robot" if initial_body else "ground"

    initial = state()
    trajectory = [sample()]
    event = detector.update(data.time, trajectory[-1]["position"])
    contacts, active_pairs = [], set()
    robot_contact = goal_contact = False
    for _ in range(math.ceil(shot.duration / model.opt.timestep)):
        if event is not None:
            break
        before_velocity = data.qvel[dof:dof + 3].tolist()
        before_time = float(data.time)
        mujoco.mj_step(model, data)
        check_state()
        pairs = set()
        for contact in data.contact[:data.ncon]:
            if contact.efc_address < 0 or geom not in (contact.geom1, contact.geom2):
                continue
            other = int(contact.geom2 if contact.geom1 == geom else contact.geom1)
            pair = (geom, other)
            pairs.add(pair)
            kind = classify(int(model.geom_bodyid[other]))
            robot_contact |= kind == "robot"
            goal_contact |= kind == "goal"
            if pair not in active_pairs:
                contacts.append({"time": before_time, "kind": kind,
                                 "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                                 "position": contact.pos.tolist(),
                                 "ball_velocity_before": before_velocity,
                                 "ball_velocity_after": data.qvel[dof:dof + 3].tolist()})
        active_pairs = pairs
        trajectory.append(sample())
        event = detector.update(data.time, trajectory[-1]["position"])

    outcome = event.outcome if event else ("Robot Contact" if robot_contact else "Timeout")
    return {"shot": asdict(shot), "robot_mode": robot_mode,
            "control_status": {"free": "uncontrolled", "fixed": "fixed_isolation", "absent": "absent"}[robot_mode],
            "outcome": outcome, "termination_reason": event.reason if event else "duration_reached",
            "goal": bool(event and event.outcome == "Goal"), "robot_contact": bool(robot_contact),
            "goal_contact": bool(goal_contact), "exit_event": event.to_dict() if event else None,
            "initial_state": initial, "final_state": state(), "trajectory": trajectory,
            "contacts": contacts, "timestep": float(model.opt.timestep)}
