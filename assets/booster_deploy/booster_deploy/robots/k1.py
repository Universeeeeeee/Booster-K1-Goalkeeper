from ..controllers.controller_cfg import PrepareStateCfg, RobotCfg


K1_CFG = RobotCfg(
    name="Booster_K1",
    prepare_mode="walking",
    joint_names=[
        "aahead_yaw_joint", "aahead_pitch_joint",
        "aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_elbow_pitch_joint", "left_elbow_yaw_joint",
        "aaright_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_elbow_pitch_joint", "right_elbow_yaw_joint",
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ],
    body_names=[
        "trunk", "aahead_yaw_link", "aahead_pitch_link",
        "aaleft_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_elbow_pitch_link", "left_elbow_yaw_link",
        "aaright_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_elbow_pitch_link", "right_elbow_yaw_link",
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_pitch_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_pitch_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    ],
    joint_stiffness=[
        4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
        80., 80.0, 80., 80., 30., 30., 80., 80.0, 80., 80., 30., 30.,
    ],
    joint_damping=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
    ],
    default_joint_pos=[
        0, 0, 0.0, -1.3, 0, -0., 0.0, 1.3, 0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0., -0.0, 0, 0, 0.0, -0.0, 0.,
    ],
    effort_limit=[
        6, 6, 14, 14, 14, 14, 14, 14, 14, 14,
        30, 35, 20, 40, 20, 20, 30, 35, 20, 40, 20, 20,
    ],
    sim_joint_names=[
        "aahead_yaw_joint", "aaleft_shoulder_pitch_joint",
        "aaright_shoulder_pitch_joint", "left_hip_pitch_joint",
        "right_hip_pitch_joint", "aahead_pitch_joint", "left_shoulder_roll_joint",
        "right_shoulder_roll_joint", "left_hip_roll_joint", "right_hip_roll_joint",
        "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_hip_yaw_joint",
        "right_hip_yaw_joint", "left_elbow_yaw_joint", "right_elbow_yaw_joint",
        "left_knee_pitch_joint", "right_knee_pitch_joint", "left_ankle_pitch_joint",
        "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    ],
    sim_body_names=[
        "trunk", "aahead_yaw_link", "aaleft_shoulder_pitch_link",
        "aaright_shoulder_pitch_link", "left_hip_pitch_link", "right_hip_pitch_link",
        "aahead_pitch_link", "left_shoulder_roll_link", "right_shoulder_roll_link",
        "left_hip_roll_link", "right_hip_roll_link", "left_elbow_pitch_link",
        "right_elbow_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link",
        "left_elbow_yaw_link", "right_elbow_yaw_link", "left_knee_pitch_link",
        "right_knee_pitch_link", "left_ankle_pitch_link", "right_ankle_pitch_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
    ],
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40., 40., 50., 20., 20., 40., 50., 20., 20.,
            350., 350., 180., 350., 250., 250., 350., 350., 180., 350., 250., 250.,
        ],
        damping=[
            1.5, 1.5, 0.5, 1.5, 0.2, 0.2, 0.5, 1.5, 0.2, 0.2,
            7.5, 7.5, 3., 5.5, 5.0, 5.0, 7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],
        joint_pos=[
            0, 0, 0.0, -1.3, 0, -0., 0.0, 1.3, 0, 0.,
            -0.0, 0, 0, 0.105, -0.10, 0., -0.0, 0, 0, 0.105, -0.10, 0.,
        ],
    ),
)
