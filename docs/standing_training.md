# 站立任务与最小 PPO 训练闭环

本轮在 I2 独立模型上接入站立奖励和单环境 PPO，完成采样、更新、保存、重新加载与独立评测。核心实现为 [standing_task.py](../k1_goalkeeper/standing_task.py)、[train_standing.py](../scripts/train_standing.py) 和 [evaluate_standing.py](../scripts/evaluate_standing.py)。

## 任务与输入输出

- 物理步长 1 ms，策略周期 20 ms，训练回合 5 s，保持原有跌倒终止与正常时间上限截断。
- 22 维动作归一化到 `[-1, 1]`，实际目标为官方参考姿态加 `0.15 × action` rad。该邻域已检查处于关节限位内；它是首版站立实验的动作范围。
- 78 维观测依次为机身重力方向（3）、机身线速度（3）、角速度（3）、相对参考位置（3）、关节角偏移（22）、关节速度（22）、上一动作（22）。角速度除以 5，位置误差分别除以 `[0.2, 0.2, 0.1]` m，关节偏移除以 0.15 rad，关节速度除以 10，最后裁到 `[-10, 10]`。所有缩放固定，无需额外保存运行均值。
- Actor 和本轮 PPO Critic 都使用上述当前状态。I1 的特权与离线数据接口保留给诊断，本轮没有实现非对称 Critic。
- 使用当前仿真真值观测；尚未加入传感器估计、噪声、遮挡或反馈延迟。

每回合在 1.0–1.1 s 对躯干施加一次世界坐标水平力。训练时 X/Y 分量各自均匀采样于 `[-10, 10]` N。扰动力计划不进入策略观测。测试验证：不同未来推力下，施力前的观测逐项一致；实际交付脉冲的积分与预期冲量一致。

## 奖励

下表各项是每秒奖励率，乘实际交付的 `elapsed_dt` 后求和；跌倒额外一次性扣 5。`g` 为机身坐标重力方向，`e` 为世界位置误差，`Δq` 为相对参考姿态的关节角，`a` 为归一化动作，`τ_limit` 为官方软件力矩限幅。

| 项目 | 奖励率 |
|---|---|
| 直立 | `1.5 × exp(-sum(g_xy²) / 0.04)` |
| 高度 | `0.5 × exp(-(e_z / 0.04)²)` |
| 站位 | `0.5 × exp(-sum(e_xy²) / 0.04)` |
| 机身运动 | `-0.1 × sum(v²) - 0.05 × sum(ω²)` |
| 姿态偏移 | `-0.5 × mean((Δq / 0.15)²)` |
| 关节运动 | `-0.002 × mean(q̇²)` |
| 力矩 | `-0.02 × mean((τ / τ_limit)²)` |
| 动作变化 | `-0.05 × mean((a - a_previous)²)` |

这些权重是首版任务设计。物理表现与奖励分别评测，奖励上升不能代替站立能力改善。若底层返回无效 transition，任务立即抛出异常，阻止缓存状态被 PPO 保存或用于 bootstrap。正常时间上限保留 `truncated` 语义，由 SB3 对有效终态 bootstrap；跌倒使用 `terminated`。

## 实验记录

正式目录：[run_01](../reports/standing_training_validation/run_01/protocol.json)。算法使用固定版本 `stable-baselines3==2.9.0` 的 PPO；这是本轮最小基线选择。[SB3 PPO 文档](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

| 配置 | 数值 |
|---|---|
| 设备与随机种子 | CPU 单线程，seed=7 |
| 策略/价值网络 | 各两层 64 单元 MLP |
| 学习率、折扣、GAE | 0.0003、0.99、0.95 |
| 每轮采样、batch、epoch | 512、128、5 |
| PPO clip、KL 上限 | 0.2、0.02 |
| 初始 log 标准差 | −0.7 |
| 实际采样 | 16,384 步，32 轮，160 个优化 epoch |
| 完整训练回合 | 79，其中 38 个达到 5 s、41 个跌倒；最后部分回合不计入完整回合统计 |
| 采样与优化耗时 | 约 39 s，约 421 decision steps/s；不含环境准备和独立评测 |

策略权重发生实际更新，参数与采样缓冲区保持有限数值。`initial.zip` 保存训练前网络，`final.zip` 保存训练后网络；保存前和重新加载后的 26 组当前状态动作逐项一致，见 [training_result.json](../reports/standing_training_validation/run_01/training_result.json) 和 `reload_check.npz`。

训练日志为 `monitor.monitor.csv`、`episodes.jsonl` 与 `logs/progress.csv`。每回合记录结束原因、奖励各分项及 Monitor 汇总；逐轮记录 PPO 损失、KL、策略标准差等。训练期间采样使用探索动作，正式评测使用确定性动作，两者分别报告。

## 独立评测协议

评测工况与判据在训练开始前写入协议：

1. 无扰动站立 30 s。
2. 世界 X 正向、X 负向、Y 正向、Y 负向各一次 15 N、0.1 s 的推力，各运行 10 s。

每个工况分别运行固定参考姿态、训练前网络、训练后网络，各重复两次，共 30 次执行。扰动力幅度与固定方向区别于训练时随机采样的分布。此集合只覆盖这些固定初态、方向和幅度，不是广泛抗扰能力统计。

预设稳定判据：完成规定时长，最大倾角不超过 10°，最大水平漂移不超过 0.1 m。另独立记录回报、跌倒、实际力矩、正接触力自碰撞、数值警告及交付冲量。每 1 ms 保存完整 qpos/qvel、实际关节力矩和外力；每 20 ms 保存观测、动作、奖励和结束标志。评测在另一个进程从 checkpoint 加载网络，不复用训练中的网络对象。

最终评测与独立交付复核见 [evaluation/results.json](../reports/standing_training_validation/run_01/evaluation/results.json)。

| 工况 | 参考姿态最大倾角 | 训练前最大倾角 | 训练后最大倾角 |
|---|---:|---:|---:|
| 无扰动，30 s | 3.346° | 3.509° | 2.714° |
| X 正向推力 | 4.436° | 4.488° | 4.163° |
| X 负向推力 | 8.264° | 8.448° | 6.552° |
| Y 正向推力 | 3.342° | 3.507° | 2.711° |
| Y 负向推力 | 3.342° | 3.502° | 2.740° |

三组控制均为 **5/5 工况稳定**，30 次执行全部完成对应时长，无跌倒、BAD 数值警告或正接触力自碰撞；重复执行的物理与 transition 数组逐项一致。训练后模型五个工况的最大倾角均低于训练前网络和固定参考姿态。水平漂移范围为参考姿态 0.0295–0.0636 m、训练前 0.0307–0.0650 m、训练后 0.0265–0.0533 m；负 Y 推力下训练后漂移约 0.0308 m，比参考姿态约 0.0299 m 略大。

无扰动回报由训练前约 73.157 变为训练后约 73.412；但 X 正向推力下训练后回报约 24.230，仍略低于参考姿态约 24.243。因此结论限定为：**该小规模实验跑通了训练闭环，在这组固定工况的倾角指标上有改善；稳定通过率与参考姿态相同，不能据此推广到广泛抗扰或守门能力。** 本轮只有一个训练随机种子。

交付时独立重读 60 份 NPZ，重新核对数组与文件哈希，并直接检查 1 ms 时间序列、外力脉冲时序、动作范围、checkpoint 输出和 PPO 日志有限性。开始前登记的 1,341 个已有文件 SHA-256 均保持一致。结果见 [delivery_audit.json](../reports/standing_training_validation/run_01/delivery_audit.json)。

## 验证与复验

新增 11 项语义检查，覆盖奖励偏好及积分、非法动作/推力、动作交付、未来推力隔离、脉冲冲量、无效 transition 拦截和 PPO 时间上限信号。全量 **412 项测试通过**，原有 401 项保留。6 条 Gymnasium 提示沿用 I1；测试末尾的 QVEL 警告来自既有故意构造极端速度的用例。

安装仅新增兼容当前 Torch/Gymnasium 的 SB3，`pip check` 通过。复验从项目根目录执行，并为每次实验使用新目录：

```sh
cd /path/to/booster
.venv/bin/python -m pip install -r requirements-ppo.txt
.venv/bin/python scripts/train_standing.py --output reports/standing_reproduction --steps 16384 --seed 7
.venv/bin/python scripts/evaluate_standing.py --run reports/standing_reproduction --output reports/standing_reproduction/evaluation
.venv/bin/python -m pytest -q
```

本轮完成站立任务的训练与评测闭环。移动、足球拦截、实机部署和更广泛的物理参数随机化仍需后续任务与独立验证。
