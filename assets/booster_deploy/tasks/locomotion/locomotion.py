from __future__ import annotations

from dataclasses import MISSING
import os

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.policy_runner import create_policy_runner


class LocomotionPolicy(Policy):
    """Walking policy with observation history."""

    def __init__(self, cfg: T1LocomotionPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        self.device = torch.device(cfg.device)

        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)

        self._model = create_policy_runner(
            policy_path,
            self.device,
            use_actor_module=self.cfg.use_actor_module,
        )

        self.actor_obs_history_length = cfg.actor_obs_history_length
        self.action_scale = torch.as_tensor(
            cfg.action_scale, dtype=torch.float32, device=self.device)
        if self.action_scale.ndim == 0:
            self.action_scale = self.action_scale.repeat(
                len(self.cfg.policy_joint_names))
        if self.action_scale.numel() != len(self.cfg.policy_joint_names):
            raise ValueError(
                f"action_scale length {self.action_scale.numel()} does not "
                f"match policy joint count "
                f"{len(self.cfg.policy_joint_names)}")

        self.robot.data.to(self.device)
        self.default_joint_pos = self.robot.default_joint_pos.to(self.device)
        self.filtered_dof_target = self.default_joint_pos.clone()
        self.arm_action_fix_index = None
        if self.cfg.arm_action_fix_joint_name is not None:
            self.arm_action_fix_index = self.robot.cfg.joint_names.index(
                self.cfg.arm_action_fix_joint_name)
        self.obs_history = None
        self.last_action = torch.zeros(
            len(self.cfg.policy_joint_names),
            dtype=torch.float32,
            device=self.device,
        )
        self.real2sim_joint_map = torch.tensor([
            self.robot.cfg.joint_names.index(name)
            for name in self.cfg.policy_joint_names
        ], dtype=torch.long, device=self.device)

    def reset(self) -> None:
        self.obs_history = None
        self.last_action.zero_()
        self.filtered_dof_target.copy_(self.default_joint_pos)

    def _initialize_obs_history(self, obs: torch.Tensor) -> torch.Tensor:
        if self.cfg.history_init == "repeat":
            return obs.repeat(self.actor_obs_history_length, 1)
        return torch.zeros(
            self.actor_obs_history_length,
            obs.numel(),
            dtype=obs.dtype,
            device=obs.device,
        )

    def _forward_model(self, obs_history: torch.Tensor) -> torch.Tensor:
        # ``create_policy_runner`` already selects the actor submodule for
        # TorchScript checkpoints when ``use_actor_module`` is enabled.  The
        # returned runner is therefore the single inference entry point for
        # both K1 and the other locomotion policies.
        return self._model(obs_history.flatten()).reshape(-1)

    def _action_to_targets(self, action: torch.Tensor) -> torch.Tensor:
        action_real = torch.zeros_like(self.default_joint_pos)
        action_real.scatter_reduce_(
            0,
            self.real2sim_joint_map,
            action,
            reduce="sum",
        )
        if self.arm_action_fix_index is not None:
            action_real[self.arm_action_fix_index] -= (
                self.cfg.arm_action_fix_offset)

        action_scale_real = torch.zeros_like(self.default_joint_pos)
        action_scale_real.scatter_reduce_(
            0,
            self.real2sim_joint_map,
            self.action_scale,
            reduce="sum",
        )
        dof_target = self.default_joint_pos + action_real * action_scale_real
        if self.cfg.action_filter < 1.0:
            self.filtered_dof_target.lerp_(
                dof_target, self.cfg.action_filter)
            return self.filtered_dof_target.clone()
        return dof_target

    def compute_observation(self) -> torch.Tensor:
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        base_quat = self.robot.data.root_quat_w
        base_ang_vel = self.robot.data.root_ang_vel_b

        gravity_w = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        if self.cfg.enable_safety_fallback:
            if projected_gravity[2] > -0.5:
                print(
                    "\nFalling detected, stopping policy for safety. "
                    "You can disable safety fallback by setting "
                    f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                    "to False."
                )
                self.controller.stop()

        command = self.controller.vel_command
        if command is None:
            command_obs = torch.zeros(3, dtype=torch.float32, device=self.device)
        else:
            command_obs = torch.tensor(
                [command.lin_vel_x, command.lin_vel_y, command.ang_vel_yaw],
                dtype=torch.float32,
                device=self.device,
            )

        mapped_default_pos = self.default_joint_pos[self.real2sim_joint_map]
        mapped_dof_pos = dof_pos[self.real2sim_joint_map]
        mapped_dof_vel = dof_vel[self.real2sim_joint_map]

        return torch.cat([
            base_ang_vel,
            projected_gravity,
            command_obs,
            mapped_dof_pos - mapped_default_pos,
            mapped_dof_vel * self.cfg.obs_dof_vel_scale,
            self.last_action,
        ], dim=0)

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation().clamp(
            -self.cfg.clip_observation,
            self.cfg.clip_observation,
        )

        if self.obs_history is None:
            self.obs_history = self._initialize_obs_history(obs)
            if self.cfg.history_init == "zeros":
                # Match the original locomotion policy: nine zero frames
                # followed by the first live observation.
                self.obs_history[-1] = obs
        else:
            self.obs_history = self.obs_history.roll(shifts=-1, dims=0)
            self.obs_history[-1] = obs

        with torch.no_grad():
            action = self._forward_model(self.obs_history)
            action = action.clamp(-self.cfg.clip_action, self.cfg.clip_action)

        expected_actions = len(self.cfg.policy_joint_names)
        if action.numel() != expected_actions:
            raise RuntimeError(
                f"policy returned {action.numel()} actions; expected "
                f"{expected_actions}")
        self.last_action.copy_(action)
        return self._action_to_targets(action)


@configclass
class T1LocomotionPolicyCfg(PolicyCfg):
    constructor = LocomotionPolicy
    checkpoint_path: str = MISSING  # type: ignore
    actor_obs_history_length: int = 10
    action_scale: float = 0.25
    obs_dof_vel_scale: float = 1.0
    clip_action: float = 100.0
    clip_observation: float = 100.0
    history_init: str = "zeros"
    use_actor_module: bool = False
    action_filter: float = 1.0
    arm_action_fix_offset: float = 0.0
    arm_action_fix_joint_name: str | None = None
    policy_joint_names: list[str] = MISSING  # type: ignore


@configclass
class K1LocomotionPolicyCfg(T1LocomotionPolicyCfg):
    constructor = LocomotionPolicy
    history_init: str = "repeat"
    obs_dof_vel_scale: float = 0.1
    use_actor_module: bool = True
    action_filter: float = 0.8
    arm_action_fix_offset: float = 0.2
    arm_action_fix_joint_name: str | None = "right_elbow_pitch_joint"


class T2LocomotionPolicy(LocomotionPolicy):
    """T2 policy with the heading observations used during training."""

    def reset(self) -> None:
        super().reset()
        # Start the history from the robot's actual pose rather than assuming
        # that the previous action was zero.  This is used only for T2's
        # first observation: inference() replaces last_action with the
        # model's real output after the first policy step.
        mapped_dof_pos = self.robot.data.joint_pos[self.real2sim_joint_map]
        mapped_default_pos = self.default_joint_pos[self.real2sim_joint_map]
        self.last_action.copy_(
            (mapped_dof_pos - mapped_default_pos) / self.action_scale
        )
        self.heading_target = None
        self.last_command = torch.zeros(3, dtype=torch.float32, device=self.device)

    def compute_observation(self) -> torch.Tensor:
        observation = super().compute_observation()
        command = torch.as_tensor(
            [self.controller.vel_command.lin_vel_x,
             self.controller.vel_command.lin_vel_y,
             self.controller.vel_command.ang_vel_yaw],
            dtype=torch.float32, device=self.device)
        standing = torch.linalg.vector_norm(command) < 0.01
        w, x, y, z = self.robot.data.root_quat_w
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        command_changed = torch.any(torch.abs(command - self.last_command) > 1.0e-6)
        if self.heading_target is None or standing or command_changed:
            self.heading_target = yaw.clone()
        else:
            self.heading_target = torch.atan2(
                torch.sin(self.heading_target + self.controller.cfg.policy_dt * command[2]),
                torch.cos(self.heading_target + self.controller.cfg.policy_dt * command[2]),
            )
        self.last_command.copy_(command)
        heading_error = torch.atan2(
            torch.sin(self.heading_target - yaw),
            torch.cos(self.heading_target - yaw),
        ).clamp(-0.5, 0.5).unsqueeze(0)
        return torch.cat([observation, heading_error, standing.float().reshape(1)], dim=0)


@configclass
class T2LocomotionPolicyCfg(T1LocomotionPolicyCfg):
    constructor = T2LocomotionPolicy
    checkpoint_path: str = MISSING  # type: ignore
    actor_obs_history_length: int = 10
    obs_dof_vel_scale: float = 1.0
    clip_action: float = 10.0
    clip_observation: float = 100.0
    history_init: str = "repeat"
    action_scale: list[float] = [
        0.2608, 0.2608, 0.2212, 0.2773, 0.2773, 0.1740,
        0.2773, 0.2773, 0.3155, 0.2773, 0.2773, 0.3155,
        0.3155, 0.3155, 0.3155, 0.3155, 0.3155, 0.3155,
        0.3155, 0.1059, 0.1059, 0.0950, 0.0950,
    ]
