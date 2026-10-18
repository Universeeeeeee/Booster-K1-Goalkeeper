# K1 足球守门 MuJoCo 仿真基线

基于 Booster K1 官方 22 DoF 模型和参考球门 URDF，提供完整球场、足球接触、可重复射门、整球越线判定与轨迹记录。已接入官方 `k1_walk` 行走策略，支持仿真真值驱动的主动横向站位；预测、抬臂防守和站立训练实验见下方文档。

## 安装

Python ≥3.11，在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/audit_assets.py
.venv/bin/python scripts/build_scene.py
```

机器人资产来自 [booster_assets](https://github.com/BoosterRobotics/booster_assets)，固定提交 `12a97516f57469078707afb587a5f84c1cb110d1`；获取与审查脚本校验版本和源文件。上游许可证保留在资产仓库内。

使用官方行走策略需额外安装依赖并获取权重：

```bash
.venv/bin/python -m pip install -r requirements-walk.txt
.venv/bin/python scripts/fetch_walk_policy.py
.venv/bin/python scripts/validate_walk.py
```

## 启动与观察

macOS 交互窗口使用 `mjpython`；其他平台可用 `.venv/bin/python`：

```bash
# 官方策略站立
.venv/bin/mjpython scripts/visualize.py --control-mode k1_walk --camera goal_front

# 站立 2 秒后横移 2 秒，再自动停止
.venv/bin/mjpython scripts/visualize.py --control-mode k1_walk --velocity 0 0.2 0 --command-duration 2 --camera goal_front

# 静态截图，不推进物理
.venv/bin/python scripts/visualize.py --camera overview --output reports/overview.png
```

Space 暂停，R 重置。相机可选 `overview`、`goal_front`、`robot_side`。

机器人模式：`free`（默认，自由基座）、`fixed`（静态刚体，用于隔离接触）、`absent`（移除机器人）。默认控制为 `zero_torque`，自由模式下会倒下；`k1_walk` 仅支持 `free` 模式和 1 ms 物理步长。静态截图不能用于验证站立。

## 射门与判定

```bash
# 无机器人，运行 10 个预设射门
.venv/bin/python scripts/shoot_ball.py --suite --robot-mode absent --output results/shot_suite.json

# 固定机器人，自定义球的位置与速度
.venv/bin/python scripts/shoot_ball.py --robot-mode fixed --position 8 0 0.5 --velocity 8 0 0 --duration 3 --output results/custom.json

# 固定种子的随机射门
.venv/bin/python scripts/shoot_ball.py --random 10 --seed 42 --robot-mode absent --output results/random.json
```

可用 `--preset center_low` 运行单球；自定义位置、速度和角速度不能与预设、套件或随机模式混用。每球重置初始状态，JSON 记录配置、轨迹、接触事件和判定结果，默认保存到 `results/shots.json`。此入口的自由机器人使用零力矩控制。

进球要求**整球越过门线且位于门框净开口内**，首次完整出界事件锁存。结果包括 `Goal`、`Miss`、`Robot Contact`、`Timeout`；接触不等于成功扑救，接触后仍可能进球。

## 场景与限制

- 配置入口：[config/simulation.json](config/simulation.json)，支持 `--config path.json`。单位为 m、kg、s、rad；X 沿场长、Y 沿场宽、Z 向上，右门在 X=11，K1 面向 −X。
- 草坪 26×18 m，有效场地 22×14 m，球门净宽 3 m、净高 2 m；足球半径 0.11 m、质量 0.43 kg。
- 默认物理步长 1 ms，使用 Newton 求解器与 Euler 积分。足球采用薄球壳惯量近似，接触参数未经实球标定，无空气阻力或马格努斯力。
- 机器人碰撞体仅覆盖部分视觉表面；球门固定，无球网、实体边墙或完整裁判规则。
- 行走策略存在低速死区和短时响应不足。主动站位与预测依赖仿真真值；现有实验不能证明完整扑救、实球预测精度或真实机器人部署能力。

## 验证与文档

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_assets.py --output reports/asset_audit.json
```

未安装行走依赖或权重时，策略专用测试会跳过。各阶段的复现命令、测量结果和适用边界见对应报告，原始输出保存在 `reports/`。

| 内容 | 文档 |
|---|---|
| 资产、模型与碰撞覆盖 | [资产审查](docs/asset_audit.md)、[碰撞审计](docs/collision_audit.md) |
| 行走与场景集成（A–D） | [行走策略](docs/walk_validation.md)、[速度控制](docs/velocity_validation.md)、[场景集成](docs/goalkeeper_validation.md) |
| 足球碰撞与抗扰（E） | [抗扰报告](docs/impact_validation.md) |
| 主动站位与预测（F–G） | [站位闭环](docs/positioning_validation.md)、[预测与可达性](docs/prediction_validation.md) |
| 抬臂防守实验（H） | [防守原型](docs/defense_validation.md)、[移停与冲击验证](docs/defense_followup_validation.md) |
| 训练模型与接口 | [训练模型](docs/training_model.md)、[训练接口](docs/training_interface.md) |
| 站立 PPO 基线 | [站立训练](docs/standing_training.md) |
