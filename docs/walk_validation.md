# 官方 K1 行走策略：第一轮 A、B 验证

此报告保留独立 K1 模型的推理、PD 和零速度站立结果，A、B 已通过。后续阶段C的基础速度控制结果另见 [速度验证](velocity_validation.md)；D场景接入与E足球撞击尚未执行，原场景入口仍是零力矩基线。

## 来源与运行

官方代码和 `k1_walk.pt` 来自 [BoosterRobotics/booster_deploy](https://github.com/BoosterRobotics/booster_deploy/tree/7bb1462e2ac742eba5c01659b815e2e99fbe617f)，固定提交 `7bb1462e2ac742eba5c01659b815e2e99fbe617f`。只导出 K1 权重和必要依赖，共22个原始文件，保留许可证；没有 T1/T2 权重或模型。`assets/booster_deploy/manifest.json` 记录每个文件的 SHA-256，运行前逐项核验。

```bash
.venv/bin/python -m pip install -r requirements-walk.txt
.venv/bin/python scripts/fetch_walk_policy.py
.venv/bin/python scripts/validate_walk.py
.venv/bin/python scripts/validate_walk.py --physics-dt 0.001 --output reports/walk_validation_1ms
```

需要实时观察时，在 macOS 使用：

```bash
.venv/bin/mjpython scripts/validate_walk.py --viewer --output results/walk_viewer
```

默认运行30秒仿真；不显示窗口时不等待实际30秒墙钟时间。`--duration` 可做短时检查，但不足30秒不会通过 B。关闭交互窗口导致验证提前结束，也不会通过 B。

## 保留的官方行为

`k1_goalkeeper/walk_controller.py` 直接调用官方 `K1WalkTaskCfg`、`BaseController` 和 `LocomotionPolicy`，没有另写观测或动作处理。MuJoCo 状态通过关节名称取得 `jnt_qposadr`/`jnt_dofadr`，执行器通过名称及 joint transmission 双重核对。控制只写22个 K1 电机，不覆盖其他物体的控制或外力。

| 项目 | 官方值/处理 |
|---|---|
| 策略关节 | 官方20关节顺序，头部两关节保持默认姿态并由PD控制 |
| 单帧观测 | 69维：机身角速度、投影重力、速度命令、20关节位置偏差、速度、上一动作 |
| 历史 | 10帧，第一帧重复初始化；690维输入官方 actor |
| 动作/观测裁剪 | ±100 |
| 动作缩放 | 0.25 |
| 关节速度观测缩放 | 0.1 |
| 目标滤波 | 0.8 |
| 右肘pitch补偿 | 在缩放前减0.2，即未滤波目标偏移−0.05 rad |
| 策略周期 | 0.02 s（50 Hz） |
| PD与力矩限幅 | 使用行走任务的覆盖参数，不使用基础 `K1_CFG` 的另一组默认值 |

官方部署代码的 physics_dt=0.002、decimation=10；原始 MJCF 为0.001。本轮默认使用官方2 ms配置，并另外在1 ms、decimation=20下验证。只修改独立运行时模型的 timestep，不修改 XML、关节、惯性、电机参数或原守门配置。每个物理子步重新计算官方 PD 和力矩限幅。

初始化采用官方默认关节姿态、基座 `(0,0,0.6)` 和单位四元数。基座保持 free joint，不添加焊接、固定关节、外部支撑或外力；官方摔倒检测保持启用。

## 实际结果

环境：Python3.13.13、MuJoCo3.3.7、PyTorch2.9.0、NumPy2.3.3，CPU单线程。每组1500次策略调用，零速度命令，连续30秒物理仿真。

| 指标 | 官方2 ms配置 | 原模型1 ms对照 |
|---|---:|---:|
| A：权重、推理、PD | 通过 | 通过 |
| B：自由基座30秒未摔倒 | 通过 | 通过 |
| 最低基座高度 | 0.547707 m | 0.548429 m |
| 最大机身倾角 | 1.323821° | 1.658750° |
| 最大水平漂移 | 0.010380 m | 0.008132 m |

站立判据为持续至少30秒、基座高度不低于0.3 m、机身朝上轴Z分量不低于0.5（倾角不超过60°），且不出现停止、非有限值、MuJoCo BAD警告或自动重置。姿态指标从按名称找到的自由关节位置和四元数计算，每20 ms保存；物理和PD分别每2/1 ms推进。

`reports/walk_validation/validation.json` 保存2 ms结果、版本、哈希、映射、PD及验收判据；`trajectory.npz` 保存状态、目标和电机输出。1 ms对照保存在 `reports/walk_validation_1ms/`。

独立审查还直接运行了原官方 `MujocoController` 30秒，并与名称映射控制器逐步比对；关节/基座状态、观测历史、目标和电机输出差值均为零。自动化测试包含短时原控制器对照、插入额外关节/足球、执行器重排，以及通过官方IMU交叉核对机身角速度。源文件均与固定提交的原始字节相同。

## 后续边界

零速度站立通过不等于已验证行走、侧移、停止恢复或接球抗扰。接入守门场景时还需核验起始位置/朝向、球接触参数和稳定性记录；原场景与射门脚本本轮没有启用策略。
