"""Scene-owned physics loop with an optional official K1 walking controller."""
import mujoco
import numpy as np


class SceneSimulation:
    def __init__(self, model, data=None, control_mode="zero_torque"):
        if control_mode not in ("zero_torque", "k1_walk"):
            raise ValueError("control_mode must be zero_torque or k1_walk")
        self.model = model
        self.data = mujoco.MjData(model) if data is None else data
        self.control_mode = control_mode
        self.controller = None
        if control_mode == "k1_walk":
            base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
            if base < 0 or model.jnt_type[base] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("k1_walk requires a free K1 base")
            if not np.isclose(model.opt.timestep, .001, rtol=0, atol=1e-12):
                raise ValueError("scene k1_walk requires a 0.001 s physics timestep")
            from .walk_controller import K1WalkController
            self.controller = K1WalkController(model, self.data)
            q = self.controller.base_qpos_address
            self.controller.cfg.mujoco.init_pos = model.qpos0[q:q+3].tolist()
            self.controller.cfg.mujoco.init_quat = model.qpos0[q+3:q+7].tolist()
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.physics_steps = self.policy_steps = 0
        if self.controller is not None:
            self.controller.initialize()
            self.targets = self.controller.robot.default_joint_pos.clone()
        else:
            self.targets = None
            mujoco.mj_forward(self.model, self.data)

    def set_velocity(self, command):
        if self.controller is None:
            raise ValueError("velocity commands require k1_walk control_mode")
        if len(command) != 3 or not np.isfinite(command).all():
            raise ValueError("command must contain three finite body velocities")
        vx, vy, yaw = command
        cfg = self.controller.vel_command
        if not (-cfg.vx_backward_max <= vx <= cfg.vx_forward_max
                and abs(vy) <= cfg.vy_max and abs(yaw) <= cfg.vyaw_max):
            raise ValueError("velocity command exceeds official task bounds")
        cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_yaw = command

    def step(self):
        if self.controller is not None:
            if not self.controller.is_running:
                raise RuntimeError("K1 controller is stopped; reset the scene before stepping")
            if self.physics_steps % self.controller.decimation == 0:
                self.targets = self.controller.compute_targets()
                self.policy_steps += 1
            self.controller.apply_control(self.targets)
        else:
            self.data.ctrl[:] = 0
        before = float(self.data.time)
        try:
            mujoco.mj_step(self.model, self.data)
        finally:
            # Passive recorders may reject a contact after the delivered physics step.
            if np.isclose(self.data.time, before + self.model.opt.timestep, rtol=0, atol=1e-9):
                self.physics_steps += 1
        if self.controller is not None:
            self.controller.finish_physics_step()

    def advance_policy_step(self):
        if self.controller is None:
            raise ValueError("policy steps require k1_walk control_mode")
        for _ in range(self.controller.decimation):
            self.step()
