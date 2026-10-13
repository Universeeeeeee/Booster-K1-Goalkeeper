"""Named action backends for I1; target PD does not load the walking policy."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from gymnasium import spaces
import mujoco
import numpy as np


DEPLOY_ROOT = Path(__file__).resolve().parents[1] / "assets/booster_deploy"
BAD_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL,
)


def _official_spec():
    """Read authenticated task literals without importing Torch or a policy."""
    manifest = json.loads((DEPLOY_ROOT / "manifest.json").read_text())
    trees = []
    for relative in ("booster_deploy/robots/k1.py", "tasks/locomotion/robots/k1/__init__.py"):
        source = (DEPLOY_ROOT / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != manifest["files"][relative]:
            raise ValueError(f"official configuration hash mismatch: {relative}")
        trees.append(ast.parse(source))
    robot = next(n.value for n in trees[0].body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "K1_CFG" for t in n.targets))
    names = tuple(ast.literal_eval(next(k.value for k in robot.keywords if k.arg == "joint_names")))
    task = next(n for n in trees[1].body if isinstance(n, ast.ClassDef)
                and n.name == "K1WalkControllerCfg")
    replacement = next(n.value for n in task.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "robot" for t in n.targets))
    parameters = {k.arg: np.asarray(ast.literal_eval(k.value), dtype=np.float32)
                  for k in replacement.keywords}
    velocity = next(n.value for n in task.body if isinstance(n, ast.AnnAssign)
                    and n.target.id == "vel_command")
    bounds = {k.arg: ast.literal_eval(k.value) for k in velocity.keywords}
    if len(names) != 22 or len(set(names)) != 22:
        raise ValueError("official K1 configuration must have 22 unique joint names")
    return names, parameters, bounds


@dataclass(frozen=True)
class JointMapping:
    joint_names: tuple
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    actuator_ids: np.ndarray
    base_qpos_address: int
    base_dof_address: int
    ball_qpos_address: int
    ball_dof_address: int
    default_positions: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort: np.ndarray
    joint_ranges: np.ndarray


def _free_addresses(model, name):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"training backend requires free joint {name}")
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def _mapping(model, names, parameters):
    if not np.isclose(model.opt.timestep, .001, rtol=0, atol=1e-12):
        raise ValueError("training backends require a 0.001 s physics timestep")
    joints, motors = [], []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"missing K1 hinge joint: {name}")
        motor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        matching = np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                  & (model.actuator_trnid[:, 0] == joint))
        if motor < 0 or matching.tolist() != [motor]:
            raise ValueError(f"K1 joint needs one matching named motor actuator: {name}")
        if (not np.array_equal(model.actuator_gear[motor], [1, 0, 0, 0, 0, 0])
                or model.actuator_dyntype[motor] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[motor] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_gainprm[motor, 0] != 1
                or model.actuator_biastype[motor] != mujoco.mjtBias.mjBIAS_NONE):
            raise ValueError(f"K1 actuator must be a unit-gear torque motor: {name}")
        if not model.jnt_limited[joint]:
            raise ValueError(f"K1 joint requires finite XML angle limits: {name}")
        joints.append(joint)
        motors.append(motor)
    joints = np.asarray(joints, dtype=int)
    ranges = model.jnt_range[joints].copy()
    if not np.isfinite(ranges).all() or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError("K1 joint ranges must be finite and increasing")
    base_q, base_v = _free_addresses(model, "world_joint")
    ball_q, ball_v = _free_addresses(model, "ball_free")
    return JointMapping(names, joints, model.jnt_qposadr[joints].copy(),
                        model.jnt_dofadr[joints].copy(), np.asarray(motors, dtype=int),
                        base_q, base_v, ball_q, ball_v, parameters["default_joint_pos"].copy(),
                        parameters["joint_stiffness"].copy(), parameters["joint_damping"].copy(),
                        parameters["effort_limit"].copy(), ranges)


def _validate_action(action, action_space):
    value = np.asarray(action, dtype=np.float32)
    if value.shape != action_space.shape or not np.isfinite(value).all():
        raise ValueError(f"action must contain finite values with shape {action_space.shape}")
    if not action_space.contains(value):
        raise ValueError("action exceeds backend bounds")
    return value.copy()


class JointTargetBackend:
    """Absolute 22-joint targets, official task PD, and one 1 ms step."""

    def __init__(self, model):
        names, parameters, _ = _official_spec()
        self.model = model
        self.mapping = _mapping(model, names, parameters)
        self.data = mujoco.MjData(model)
        self.action_space = spaces.Box(self.mapping.joint_ranges[:, 0].astype(np.float32),
                                       self.mapping.joint_ranges[:, 1].astype(np.float32),
                                       dtype=np.float32)
        self.default_action = self.mapping.default_positions.copy()
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mapping = self.mapping
        q = mapping.base_qpos_address
        self.data.qpos[q:q+7] = self.model.qpos0[q:q+7].astype(np.float32)
        self.data.qpos[mapping.qpos_addresses] = mapping.default_positions
        mujoco.mj_forward(self.model, self.data)
        self._targets = self.default_action.copy()
        self.physics_steps = self.policy_steps = 0
        self._expected_time = float(self.data.time)
        self._warnings = np.asarray([self.data.warning[w].number for w in BAD_WARNINGS])

    def set_action(self, action):
        self._targets = _validate_action(action, self.action_space)

    def _check_state(self):
        if (not all(np.isfinite(value).all() for value in
                    (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl))
                or not np.isclose(self.data.time, self._expected_time, rtol=0, atol=1e-9)
                or not np.array_equal(self._warnings,
                                      [self.data.warning[w].number for w in BAD_WARNINGS])):
            raise RuntimeError("MuJoCo nonfinite state, BAD warning, or unexpected time reset")

    def step_physics(self):
        self._check_state()
        mapping = self.mapping
        position = self.data.qpos.astype(np.float32)[mapping.qpos_addresses]
        velocity = self.data.qvel.astype(np.float32)[mapping.dof_addresses]
        self.data.ctrl[mapping.actuator_ids] = np.clip(
            mapping.kp * (self._targets - position) - mapping.kd * velocity,
            -mapping.effort, mapping.effort)
        self._expected_time += self.model.opt.timestep
        before = float(self.data.time)
        try:
            mujoco.mj_step(self.model, self.data)
        finally:
            if np.isclose(self.data.time, before + self.model.opt.timestep, rtol=0, atol=1e-9):
                self.physics_steps += 1
        self._check_state()


class BodyVelocityBackend:
    """Body velocity commands delivered to the unchanged official walk policy."""

    def __init__(self, model):
        names, parameters, bounds = _official_spec()
        self.model = model
        self.mapping = _mapping(model, names, parameters)
        self.action_space = spaces.Box(
            np.asarray([-bounds["vx_backward_max"], -bounds["vy_max"], -bounds["vyaw_max"]], dtype=np.float32),
            np.asarray([bounds["vx_forward_max"], bounds["vy_max"], bounds["vyaw_max"]], dtype=np.float32),
            dtype=np.float32)
        self.default_action = np.zeros(3, dtype=np.float32)
        from .scene_simulation import SceneSimulation
        self.simulation = SceneSimulation(model, control_mode="k1_walk")
        self.data = self.simulation.data

    @property
    def physics_steps(self):
        return self.simulation.physics_steps

    @property
    def policy_steps(self):
        return self.simulation.policy_steps

    def reset(self):
        self.simulation.reset()

    def set_action(self, action):
        value = _validate_action(action, self.action_space)
        # Box bounds are float32; avoid their outward rounding violating exact task bounds.
        cfg = self.simulation.controller.vel_command
        command = np.clip(value.astype(float),
                          [-cfg.vx_backward_max, -cfg.vy_max, -cfg.vyaw_max],
                          [cfg.vx_forward_max, cfg.vy_max, cfg.vyaw_max])
        self.simulation.set_velocity(command)

    def step_physics(self):
        self.simulation.step()


def create_action_backend(model, mode="joint_targets"):
    if mode == "joint_targets":
        return JointTargetBackend(model)
    if mode == "body_velocity":
        return BodyVelocityBackend(model)
    raise ValueError("action mode must be joint_targets or body_velocity")
