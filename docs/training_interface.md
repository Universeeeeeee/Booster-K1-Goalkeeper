# I1：K1 单环境训练接口

本阶段建立 Gymnasium 单环境接口，供未来统一站立、移动、拦截与恢复策略使用。没有训练策略、迭代现有守门算法、补碰撞体、随机化任务、开发并行训练或调优奖励。

实现入口：[training_env.py](../k1_goalkeeper/training_env.py)。接口遵循 [Gymnasium Env 契约](https://gymnasium.farama.org/api/env/)，依赖固定为 [Gymnasium 1.4.0](https://pypi.org/project/gymnasium/1.4.0/)。原 MuJoCo 场景生成器、进球判定器与官方控制器直接复用。

## 使用方式

```sh
cd /path/to/booster
.venv/bin/python -m pip install -r requirements-training.txt
# body_velocity 参考模式另需已有 requirements-walk.txt 和 A–H 官方权重。
```

```python
from k1_goalkeeper.training_env import K1GoalkeeperEnv
from k1_goalkeeper.simulation import Shot

with K1GoalkeeperEnv(action_mode="joint_targets") as env:
    actor_obs, info = env.reset(seed=7, options={"task": "standing"})
    critic_state = env.get_privileged_state()  # 单独交给 Critic
    actor_obs, reward, terminated, truncated, info = env.step(env.default_action)

    actor_obs, info = env.reset(seed=7, options={
        "task": "single_shot",
        "shot": Shot(position=(7, .8, .111), velocity=(4, 0, 0),
                     angular_velocity=(0, 4/.11, 0)),
        "launch_time": 2.0,
        "time_limit": 8.0,
    })
```

`reset` 返回 `(actor_obs, info)`；`step` 返回 `(actor_obs, reward, terminated, truncated, info)`。回合结束后必须显式 reset。无自动重置、无隐藏 warmup；reset 后物理时间为 0，官方策略历史、滤波、上一动作及计数重新初始化。

物理步长固定 **1 ms**。默认一次 action 推进 **20 个物理步 / 20 ms**，每物理步检查状态、接触和进球。若提前终止或到达时间上限，本次 step 只推进必要子步。`control_dt` 可选择正的整数毫秒；body_velocity 内部官方策略始终每 20 ms 推理，与外层决策周期分开。

## 可替换动作接口

| 模式 | 动作空间 | 语义与用途 | 依赖和限制 |
|---|---|---|---|
| `joint_targets`，默认 | 22 维 float32，XML 关节范围内 | 按官方机器人关节名称顺序给绝对角度，单位 rad；每毫秒执行官方任务 PD 及力矩限幅 | 不加载 `k1_walk.pt`，不导入 Torch 或行走控制器；零向量不代表站立，应使用 `default_action` 获取参考姿态 |
| `body_velocity` | 3 维 float32 | 机身 `[vx, vy, yaw_rate]`，单位 m/s、m/s、rad/s；复用官方行走控制 | 显式依赖已验证的 `k1_walk`；只是参考后端，不能表达全身防守动作 |

速度边界沿用官方任务配置：vx∈[-0.3,1.6]、vy∈[-0.3,0.3]、yaw_rate∈[-1.8,1.8]。边界声明不表示 I1 已验证全部速度或动作安全。

两种后端都读取关节名称、`jnt_qposadr`、`jnt_dofadr` 及对应 motor，不假设机器人占满状态和控制数组。官方默认姿态、KP/KD、软件限幅从已校验 SHA-256 的官方配置读取；关节模式只读取静态字面量。球的自由关节与其它对象不会被误控制。

后端契约为 `model / data / mapping / action_space / default_action` 及 `reset / set_action / step_physics`，实现见 [training_actions.py](../k1_goalkeeper/training_actions.py)。`backend_factory(model)` 可替换实现，须使用同一个 MuJoCo model、一维 Box 和固定的 22 个 K1 关节语义；reset 可更新 data/mapping 引用，但不能改变空间契约。未来选择绝对目标、目标增量或其它控制结构时可替换此接口，本轮没有确定最终策略网络或动作结构。

## 三条数据通道

**Actor：**`reset/step` 只返回一个 float32 数组。本轮明确使用**当前仿真真值**，尚未实现真实传感器估计、视觉、遮挡、噪声和时延。当前真值不是未来轨迹，但也不能称为实机可直接获取的观测。

| 字段 | 维度 | 坐标与含义 |
|---|---:|---|
| `projected_gravity` | 3 | 世界单位重力方向转入机身坐标 |
| `body_linear_velocity` | 3 | 基座世界线速度转入机身坐标 |
| `body_angular_velocity` | 3 | MuJoCo 自由关节本地角速度 |
| `goal_relative_position` | 3 | 右门坐标 `[world_y, 11-world_x, world_z]`，单位 m |
| `world_heading_cos_sin` | 2 | 世界 yaw 的 cos/sin；不是球门局部 yaw 或球门方位角 |
| `joint_position_offset` | 22 | 相对官方参考姿态的真实关节角，rad |
| `joint_velocity` | 22 | 真实关节角速度，rad/s |
| `ball_relative_position` | 3 | 当前球位置减基座位置后转入机身坐标，m |
| `ball_relative_linear_velocity` | 3 | 世界球与基座线速度之差转入机身坐标，m/s；不含旋转坐标导数的 `-ω×r` 项 |
| `ball_observation_valid` | 1 | 正常真值观测为 1；数值失败回传缓存观测并要求丢弃 transition |
| `previous_action` | 3 或 22 | 上一次接受的外层动作；reset 使用全零历史填充 |

总维数：机身速度模式 **68**，关节目标模式 **87**。可从 `observation_layout` 查各字段起止索引，从 `mapping.joint_names` 查关节顺序。所有返回数组独立复制。

**Critic：**显式调用 `get_privileged_state()`，与 `privileged_observation_space` 一致。包含当前完整 qpos/qvel/qacc、K1 实际执行器力矩、球本地角速度、body 质量与惯量。没有射门计划、待发射球的初始参数、未来轨迹或最终结果标签。不把此字典放入 Actor 数组或普通 info。

**离线评测：**`get_evaluation_state()` 可读取实际状态、接触诊断、射门配置、计划发射时刻、已交付发球、已经发生的出口事件及最后有效物理状态。该通道用于诊断，不能拼入 Actor 或 Critic。最终进球标签在 rollout 完成后写入离线汇总，环境不调用 G 预测器或真实未来 rollout。

验证脚本分别保存 `actor.npz`、`critic.npz`、`evaluation.npz` 和 `evaluation_summary.json`。Actor 文件内 action/reward/结束标志属于 transition 元数据，**不是 observation 特征**。reset 行有 `is_transition=False`；其后的行是对应 action 的返回观测，训练数据配对需要用前一行观测作为输入。Critic 只记录有效状态，按时间与 transition_valid 对齐，不能默认失效回合的数组行数相同。

## Reset、事件与奖励

Reset options 仅支持：

- `task="standing"`：球停放在场内远处，不安排发射。
- `task="single_shot"`：`shot` 为现有 `Shot` 或字段字典；`launch_time` 默认为 2 s。发射可位于任意 1 ms 子步，必须早于时间上限。
- `time_limit`：回合总物理时长，默认为构造参数 `episode_duration=10`，必须是正的整数毫秒。`Shot.duration` 是原结构的元数据，不替代这里的时间上限。

`seed` 接入 Gymnasium 自有 NumPy Generator，不重设全局 NumPy RNG。本阶段任务为固定输入，不声称 seed 已实现射门随机化；I3 再使用这个 Generator 构造随机任务。

待发射参数不写入 Actor/Critic/info。球从停放位置移至发射位置时重建 `GoalDetector`，避免将 teleport 当作过线。实际运动每毫秒复用原全球越线、左右门开口和场地出口逻辑；机器人碰球本身不结束回合。

| 原因 | `terminated` | `truncated` | Bootstrap/数据处理 |
|---|---|---|---|
| 全球进门、出界、跌倒 | True | False | 不 bootstrap；事件位置和时刻保留 |
| 正常时间上限 | False | True | 当前状态有效，允许 bootstrap |
| 数值异常、BAD warning 或意外时间重置 | False | True | 返回上一 decision 的有效 Actor 观测，`discard_transition=True`、`transition_valid=False`，禁止 bootstrap |
| 正常继续 | False | False | 当前有效状态允许 bootstrap |

跌倒基线使用高度<0.3 m 或倾角>60°。未来恢复任务需要在 I3 重新定义可恢复倒地与真正终止的边界；当前测试不宣称已经实现恢复任务。官方 policy safety fallback 也不会被禁用。

`info` 提供实际物理步数、决策步数、elapsed_dt、当前诊断、返回观测时间、最后有效物理时间及有效性。数值异常下这些时间可能不同，不能把缓存观测当作新的真实终态。离线 getter 同时保留故障当前状态和最后有效状态。reset 失败时不允许继续 step；奖励回调报错也必须重新 reset。

默认奖励只提供未调优的事件基线：右门进球 -1、跌倒 -1，其余 0。碰球和 timeout 不奖励，不将接触等同成功挡球。可替换 `reward_fn(transition)->dict[str, finite_scalar]`；输入包含当前 Actor、当前 Critic、实际事件和有效性，不包含离线计划或未来标签。返回各分项及总和。当前稀疏奖励不是训练完成方案，不应据此直接启动大规模训练。

## 验证与复现

正式验证数据：[results.json](../reports/training_interface_validation/results.json)，冻结方案：[protocol.json](../reports/training_interface_validation/protocol.json)，原接口审计：[training_interface_audit.md](training_interface_audit.md)。

**5 个正式工况、10 次执行全部通过，三通道数组重复一致。**结果如下：

| 工况 | 实际结果 | 终止/截断 |
|---|---|---|
| 机身速度模式零命令站立 | 30 s；最大倾角 1.191° | 时间上限截断 |
| 关节目标模式默认姿态 | 30 s；最大倾角 3.346° | 时间上限截断 |
| 机身速度模式单次射门 | t≈3.028 s 整球进右门，越线 x=11.11 m | terminated |
| 关节目标模式球门线工况 | t≈0.027 s 整球进右门，越线 x=11.11 m | terminated |
| 机身速度模式接触工况 | t≈3.631 s 开始真实接触，86 个正接触力样本；继续至 8 s | 时间上限截断 |

站立结论只覆盖固定初态和固定默认输入，不是对未来关节策略的移动、抗扰或统一守门能力保证。关节目标的进球工况从球门线附近发射，用于检查接口终止与插值，不是可比的守门成功率实验。接触工况不宣称成功扑救。

两种模式的 Gymnasium 检查通过；每种四次交替 reset 的初态、Actor、Critic、info 及官方策略历史一致，而离线待发射参数确实不同。30 份 NPZ 已独立重读，核验哈希、时间、实际执行器力矩、命名控制映射和力矩上限。1,070 份 A–H 原文件及官方资产的哈希不变。

原 322 项测试保留，新增 71 项，**393 项全部通过**，见 [全量测试输出](../reports/training_interface_validation/tests_after.txt)。6 条 Gymnasium 建议对应上述物理单位动作和无限 Box 边界。另一个原生 QVEL 警告来自未修改的 `test_mujoco_silent_unstable_reset_is_error` 故意输入 `1e100 m/s`，其来源在 [E0](collision_audit.md) 已独立确认；正式 I1 工况的四类 BAD warning 计数均为 0。

脚本依次检查两模式固定输入站立、两模式单次进球终止、真正球–机器人接触后继续执行；每例两遍，比对三通道数组哈希。另做两模式 Gymnasium 检查及不同待发球参数、四次交替 reset 的 Actor/Critic/info/官方策略历史隔离证明。所有这些是单环境串行验收，没有训练或吞吐量评估。

```sh
cd /path/to/booster
# 正式目录已冻结；复现请使用新目录。
.venv/bin/python scripts/validate_training_env.py --output reports/training_interface_reproduction
.venv/bin/python -m pytest -q
```

Gymnasium 检查会提示原始物理动作没有归一化、原始观测 Box 使用无限界。这是保留物理单位和开放观测范围的选择，需由未来训练配置明确归一化，不表示 NaN 被接受。正常状态和动作必须有限。

## I2 所需具体工作

1. 创建独立训练模型版本及来源/参数差异清单。保持官方 MJCF 和历史场景原样，记录碰撞、质量、惯量、关节/执行器与接触参数的模型指纹。
2. 按未来动作实际接触路径确定必要区域。重点检查肩 pitch 根部、hip pitch、ankle pitch 等仅视觉部位，以及原生躯干/臂/腿/足代理的连续性。手部应依据实际远端资产确认形状，不能虚构腕关节或放大几何。现有 20 个有效 K1 geom 不代表整片视觉外壳完整覆盖。
3. 必要补充几何须依据已有真实资产，采用合理尺寸；检查所有动作范围的自碰撞、机器人–球/门/地面接触及初始穿透。射门发射位置合法性也应进入后续任务生成校验，I1 本身不提供任意输入的碰撞清除器。
4. 复核足球质量/半径、薄壳惯量假设、滚动/滑动/扭转摩擦、反弹和接触时间参数。现参数是仿真基线，尚非完整实物标定；需要落球、滚动、斜碰和不同身体部位的受控实验，记录接触力、冲量与动量/能量变化。
5. 处理 H0 发现的距离查询后端差异：用实际求解器接触、几何资产和独立测量交叉核验，不能仅靠距离 getter 认证覆盖或安全。原生和扩展模式应独立出结果。
6. 扩展模型后验收质量/惯量无意外变化、1 ms 稳定性、足地支撑、自碰撞、30 s 参考站立、全球过线一致性，以及同一发球状态下的原生/训练模型配对。Actor/Critic 空间、reset、终止和三通道确定性也必须重新核验。

I1 的完成不表示 I2 物理模型、I3 连续守门/恢复任务或 I4 并行训练已就绪。
