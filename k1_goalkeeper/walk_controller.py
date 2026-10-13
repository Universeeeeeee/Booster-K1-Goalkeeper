"""Named MuJoCo state/control adapter for the unchanged official K1 policy."""

from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

DEPLOY_ROOT = Path(__file__).resolve().parents[1] / "assets/booster_deploy"
sys.path.insert(0, str(DEPLOY_ROOT))

from booster_deploy.controllers.base_controller import BaseController
from booster_deploy.utils.isaaclab import math as lab_math
from tasks.locomotion.robots.k1 import K1WalkTaskCfg


class K1WalkController(BaseController):
    """Run one official 50 Hz policy step and its torque-controlled substeps."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        cfg = K1WalkTaskCfg()
        cfg.policy.device = "cpu"
        cfg.policy.checkpoint_path = str(
            DEPLOY_ROOT / "tasks/locomotion" / cfg.policy.checkpoint_path
        )
        physics_dt = float(model.opt.timestep)
        ratio = cfg.policy_dt / physics_dt if physics_dt > 0 else 0
        self.decimation = round(ratio)
        if self.decimation < 1 or not np.isclose(ratio, self.decimation, rtol=0, atol=1e-9):
            raise ValueError("physics timestep must divide the official 0.02 s policy period")
        cfg.mujoco.physics_dt = physics_dt
        cfg.mujoco.decimation = self.decimation

        names = cfg.robot.joint_names
        if len(set(names)) != len(names):
            raise ValueError("K1 joint configuration contains duplicate joint names")
        joint_ids, actuator_ids = [], []
        for name in names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"missing K1 joint: {name}")
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"K1 joint must be a hinge: {name}")
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            matching = np.flatnonzero(
                (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (model.actuator_trnid[:, 0] == joint_id)
            )
            if actuator_id < 0 or matching.tolist() != [actuator_id]:
                raise ValueError(f"K1 joint needs one matching named motor actuator: {name}")
            if (
                not np.array_equal(model.actuator_gear[actuator_id], [1, 0, 0, 0, 0, 0])
                or model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE
                or model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_gainprm[actuator_id, 0] != 1
                or model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_NONE
            ):
                raise ValueError(f"K1 actuator must be a unit-gear torque motor: {name}")
            joint_ids.append(joint_id)
            actuator_ids.append(actuator_id)
        self.joint_ids = np.asarray(joint_ids, dtype=int)
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_addresses = model.jnt_dofadr[self.joint_ids].copy()
        self.actuator_ids = np.asarray(actuator_ids, dtype=int)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
        if base_id < 0 or model.jnt_type[base_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("K1 requires its free base joint named world_joint")
        self.base_qpos_address = int(model.jnt_qposadr[base_id])
        self.base_dof_address = int(model.jnt_dofadr[base_id])
        self._bad_warnings = [
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
            mujoco.mjtWarning.mjWARN_BADCTRL,
        ]
        self._warning_counts = self._read_warning_counts()
        self._expected_time = float(data.time)
        super().__init__(cfg)

    def _read_warning_counts(self):
        return np.asarray([self.data.warning[w].number for w in self._bad_warnings])

    def _check_simulation(self):
        if (
            not all(np.isfinite(a).all() for a in (self.data.qpos, self.data.qvel, self.data.qacc))
            or not np.isfinite(self.data.time)
            or not np.isclose(self.data.time, self._expected_time, rtol=0, atol=1e-9)
            or not np.array_equal(self._read_warning_counts(), self._warning_counts)
        ):
            self.stop()
            raise RuntimeError("MuJoCo has nonfinite state, a BAD warning, or an unexpected time reset")

    def initialize(self):
        """Reset to the task's initial pose with the base remaining free."""
        q, v = self.base_qpos_address, self.base_dof_address
        self.data.qpos[q:q + 3] = np.asarray(self.cfg.mujoco.init_pos, dtype=np.float32)
        self.data.qpos[q + 3:q + 7] = np.asarray(self.cfg.mujoco.init_quat, dtype=np.float32)
        self.data.qpos[self.qpos_addresses] = self.robot.default_joint_pos.numpy()
        self.data.qvel[v:v + 6] = 0
        self.data.qvel[self.dof_addresses] = 0
        self.data.ctrl[self.actuator_ids] = 0
        mujoco.mj_forward(self.model, self.data)
        self._warning_counts = self._read_warning_counts()
        self._expected_time = float(self.data.time)
        self.vel_command.lin_vel_x = 0.0
        self.vel_command.lin_vel_y = 0.0
        self.vel_command.ang_vel_yaw = 0.0
        self.update_state()
        self.start()

    def update_state(self):
        """Populate official robot-order float32 tensors using joint addresses."""
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)
        q, v = self.base_qpos_address, self.base_dof_address
        robot_data = self.robot.data
        robot_data.joint_pos.copy_(torch.from_numpy(qpos[self.qpos_addresses]))
        robot_data.joint_vel.copy_(torch.from_numpy(qvel[self.dof_addresses]))
        robot_data.feedback_torque.copy_(torch.from_numpy(
            self.data.qfrc_actuator[self.dof_addresses].astype(np.float32)
        ))
        robot_data.root_pos_w.copy_(torch.from_numpy(qpos[q:q + 3]))
        robot_data.root_quat_w.copy_(torch.from_numpy(qpos[q + 3:q + 7]))
        robot_data.root_lin_vel_b.copy_(lab_math.quat_apply_inverse(
            robot_data.root_quat_w, torch.from_numpy(qvel[v:v + 3])
        ))
        robot_data.root_ang_vel_b.copy_(torch.from_numpy(qvel[v + 3:v + 6]))

    def compute_targets(self):
        """Infer targets without advancing physics; preserve official policy history."""
        if not self.is_running:
            raise RuntimeError("K1 controller is stopped; call initialize() before step()")
        self._check_simulation()
        self.update_state()
        targets = self.policy_step()
        if not self.is_running:
            raise RuntimeError("official K1 safety fallback stopped the policy")
        return targets

    def apply_control(self, dof_targets):
        """Write one official PD update; the caller owns the next physics step."""
        self._check_simulation()
        targets = dof_targets.detach().cpu().numpy()
        if not np.isfinite(targets).all():
            self.stop()
            raise RuntimeError("K1 policy produced nonfinite joint targets")
        position = self.data.qpos.astype(np.float32)[self.qpos_addresses]
        velocity = self.data.qvel.astype(np.float32)[self.dof_addresses]
        self.data.ctrl[self.actuator_ids] = np.clip(
            self.robot.joint_stiffness.numpy() * (targets - position)
            - self.robot.joint_damping.numpy() * velocity,
            -self.robot.effort_limit.numpy(), self.robot.effort_limit.numpy()
        )
        self._expected_time += self.model.opt.timestep

    def finish_physics_step(self):
        self._check_simulation()

    def ctrl_step(self, dof_targets):
        for _ in range(self.decimation):
            self.apply_control(dof_targets)
            mujoco.mj_step(self.model, self.data)
            self.finish_physics_step()

    def step(self):
        """Infer official targets and advance exactly one 0.02 s policy period."""
        targets = self.compute_targets()
        self.ctrl_step(targets)
        self.update_state()
        return targets.detach().cpu().numpy().copy()
