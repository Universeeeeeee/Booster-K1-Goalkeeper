# 资产审查与控制缺口

审查依据是实际加载的 MuJoCo 模型、官方仓库源码，以及用户原始球门 URDF。运行 `python scripts/audit_assets.py` 可重新生成完整机器可读报告 `reports/asset_audit.json`，其中包含每个电机的名称、顺序、关节范围、力矩上限、每个碰撞几何及 SHA-256。

## 来源与完整性

- K1 来源：[BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets)，固定提交 `12a97516f57469078707afb587a5f84c1cb110d1`。使用 `robots/K1/K1_22dof.xml`，没有用 locomotion 或 parallel 模型替代。
- 官方 MJCF 引用的 24 个 STL 全部存在并可加载。MJCF、对应 URDF 和 24 个 STL 的实际文件 SHA-256 均与该提交的 Git blob 比较一致。
- 球门位于 `models/futsal_goal_3x2_reference.urdf`，从项目根目录移入，内容未改。保留原文件，不把 URDF 当作 MJCF `<include>` 片段。它是用户提供的参考模型，不是主办方官方 CAD。
- 资产工作目录仅检出 `robots/K1/`、许可证和上游 README；T1、T2、动作CSV和上游 Python 包未检出。审查所需的上游动作元数据通过固定提交的 Git blob 读取，不需要保留这些工作目录。
- 足球采用原生 sphere，避免为可选外观引入 DAE 转换依赖。没有导入 PhysicsFC 或 simatch 的足球网格、接触参数，也没有声称外观经过 FIFA 认证。

## K1 实际编译结果

在 Python 3.13.13、MuJoCo 3.3.7、NumPy 2.3.3 环境加载：`nq=29`、`nv=28`、`nu=22`，即自由基座加 22 个铰接关节。加载的刚体总质量为 19.666 kg。保留官方关节、惯性、执行器、IMU 和默认接触参数；官方 timestep 为 0.001 s，求解器 Newton，未指定 integrator，因此为 Euler。

22 个电机的排列与加载模型中的 22 个非自由关节排列一致，均为 gear=1 的力矩 motor，无内置位置 PD。`MjData.ctrl` 的顺序如下；左右对应项的上限相同。

| 索引 | 电机/关节 | 力矩范围 N·m |
|---|---|---|
| 0 | aahead_yaw_joint | ±6 |
| 1 | aahead_pitch_joint | ±6 |
| 2 | aaleft_shoulder_pitch_joint | ±14 |
| 3 | left_shoulder_roll_joint | ±14 |
| 4 | left_elbow_pitch_joint | ±14 |
| 5 | left_elbow_yaw_joint | ±14 |
| 6 | aaright_shoulder_pitch_joint | ±14 |
| 7 | right_shoulder_roll_joint | ±14 |
| 8 | right_elbow_pitch_joint | ±14 |
| 9 | right_elbow_yaw_joint | ±14 |
| 10 | left_hip_pitch_joint | ±68 |
| 11 | left_hip_roll_joint | ±43 |
| 12 | left_hip_yaw_joint | ±38.3 |
| 13 | left_knee_pitch_joint | ±112 |
| 14 | left_ankle_pitch_joint | ±38.3 |
| 15 | left_ankle_roll_joint | ±38.3 |
| 16 | right_hip_pitch_joint | ±68 |
| 17 | right_hip_roll_joint | ±43 |
| 18 | right_hip_yaw_joint | ±38.3 |
| 19 | right_knee_pitch_joint | ±112 |
| 20 | right_ankle_pitch_joint | ±38.3 |
| 21 | right_ankle_roll_joint | ±38.3 |

### 碰撞覆盖

机器人共有 20 个启用碰撞的几何体，地面另计：

- 躯干：box 和 cylinder；头部：两个 mesh。
- 每侧手臂：肩 roll 的 cylinder、肘 pitch 和肘 yaw 的 mesh。
- 每侧腿：hip roll 与 hip yaw 的 cylinder；膝部 cylinder 与小腿局部 box；足底 box。
- `aaleft_shoulder_pitch_link`、`aaright_shoulder_pitch_link`、左右 `hip_pitch_link`、左右 `ankle_pitch_link` 只有视觉 mesh，没有直接碰撞体。

因此已有足底、躯干和部分手臂接触能力，但没有完整覆盖视觉表面。大腿/小腿的代理几何也只覆盖局部；启用碰撞的 mesh 由 MuJoCo 的凸包接触处理。机器人与球接触验证只能证明测试位置存在接触，不能保证足球在任意视觉表面都不会穿过。为遵守复用官方模型的要求，本阶段没有补建碰撞体。

### 初始姿态与朝向

官方零关节姿态的基座 z=1 m，两个足底碰撞 box 下缘 z=0.44831 m，原始姿态悬空。将基座 z 调为 0.55369 m 后，足底下缘为 0.002 m。该几何摆放只是消除初始穿模，不是站立控制。

通过加载模型检查 `head_booster_stereo_rgb_link` 的光学 +Z 方向，机器人局部向量约为 `[1, 3.2e-7, 3.2e-7]`，据此确定局部 +X 对应朝前。右侧守门位绕世界 Z 转 π，朝向世界 −X。此检查用于姿态摆放，不实现摄像头感知。

## 球门检查

原始 URDF 直接被 MuJoCo 编译为固定模型，`nq=nv=0`。默认导入仅保留三个碰撞 cylinder，丢弃 visual；统一场景转换需显式保留 visual，才能显示后支架。固定性由编译后的自由度与世界附着验证，不依赖 Gazebo `static` 标签。

从实际加载几何计算：立柱中心 x=±1.54 m、半径 0.04 m，净宽 `3.08−0.08=3.0 m`；横梁中心 z=2.04 m、半径 0.04 m，下缘 2.0 m。原点为门线地面中心，局部 +Y 朝球场；左门绕 Z 转 −π/2，右门 +π/2，并平移到世界 x=−11/+11 m。后部支架仅 visual，无碰撞，也无球网。

## 原始资产仓库中的控制器缺口与独立动力学基线

审查固定提交的仓库文件、README、motion helper 和官方 actuator 定义，没有找到站立控制器、PD 增益或策略权重。仓库中的两个 K1 CSV 是 50 Hz 重定向动作轨迹，不能代替闭环平衡控制。

还有一个接口注意点：`src/booster_assets/motions.py` 的 CSV 元数据索引 2、6 分别写为 `left_shoulder_pitch_joint`、`right_shoulder_pitch_joint`，实际 MJCF 名称含 `aa` 前缀。后续接入动作时必须显式核对映射，不能按名字直接加载并认为兼容。

审查脚本对原始官方自由基座模型施加零电机力矩，从足底 2 mm 间隙起运行 2 s：基座 z 从 0.55369 m 降至约 0.208525 m，躯干向上轴的世界 Z 分量约 −0.07619，状态保持有限。这个独立实验明确展示无控制器模型会倒下，不能验收动态站立。

此处与 `reports/asset_audit.json` 的控制缺口结论仅针对 `booster_assets` 仓库，不包括后续获取的 `booster_deploy`。另一个官方仓库提供了 `k1_walk.pt`、任务PD和默认关节姿态，现已完成独立 A、B 验证，见 [行走策略验证](walk_validation.md)。接球抗扰、守门场景接入和自动扑救仍待验证；固定基座仅用于静态可视化、隔离碰撞测试。
