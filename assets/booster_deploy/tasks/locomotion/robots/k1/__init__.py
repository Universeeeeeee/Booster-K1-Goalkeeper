from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.k1 import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...locomotion import K1LocomotionPolicyCfg


@configclass
class K1WalkControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(  # type: ignore
        default_joint_pos=[
            0, 0,
            0.2, -1.25, 0, -0.5,
            0.2, 1.25, 0, 0.5,
            -0.15, 0, 0, 0.3, -0.15, 0,
            -0.15, 0, 0, 0.3, -0.15, 0,
        ],
        joint_stiffness=[
            4.0, 4.0,
            20.0, 20.0, 20.0, 20.0,
            20.0, 20.0, 20.0, 20.0,
            100.0, 100.0, 100.0, 100.0, 65.0, 65.0,
            100.0, 100.0, 100.0, 100.0, 65.0, 65.0,
        ],
        joint_damping=[
            1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
        ],
        effort_limit=[
            6.0, 6.0,
            14.0, 14.0, 14.0, 14.0,
            14.0, 14.0, 14.0, 14.0,
            30.0, 20.0, 15.0, 35.0, 24.0, 15.0,
            30.0, 20.0, 15.0, 35.0, 24.0, 15.0,
        ],
    )
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_forward_max=1.6,
        vx_backward_max=0.3,
        vy_max=0.3,
        vyaw_max=1.8,
    )
    policy: K1LocomotionPolicyCfg = K1LocomotionPolicyCfg(
        policy_joint_names=[
            "aaleft_shoulder_pitch_joint",
            "aaright_shoulder_pitch_joint",
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_elbow_pitch_joint",
            "right_elbow_pitch_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_elbow_yaw_joint",
            "right_elbow_yaw_joint",
            "left_knee_pitch_joint",
            "right_knee_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        ],
    )


@configclass
class K1WalkTaskCfg(K1WalkControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "robots/k1/models/k1_walk.pt"


register_task("k1_walk", K1WalkTaskCfg())
