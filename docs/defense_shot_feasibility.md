# 现有 G 空中球：事后时序与 native 覆盖诊断

本报告只读原 G 正式结果，服务于实验设计；没有执行 H4，也没有运行新的臂动作。完整 2 s 抬臂姿态无法在这四例发球后的穿越时刻前到位，但**未完全到位不等于不能接触球**。H1 是否通过、是否执行 H4，仍由根任务依据 H1 动态门槛决定。

机器可读证据：[shot_feasibility_audit.json](/path/to/booster/reports/defense_validation/shot_feasibility_audit.json)。原始 G 与配对 absent 轨迹均先核验 `trajectory_hash` 与原 summary 一致；22 份读取的轨迹、summary、源码和 H0 引用文件 SHA-256 前后保持一致。

## 实际穿越时刻与高度

发球绝对仿真时刻均为约 2.000 s。对原 1 ms 轨迹中 `ball_x - reference_x(t)` 的首个前向符号变化做相邻样本线性插值，使用机器人原实际运动的 base_x(t)。下表每格为 **发球后秒数 / 球中心世界高度 m**，不是在线预测：

| 原 G 正式案例 | G 保守 deadline：base_x−0.81 | native 最前投影界减球半径 | 机器人中心：base_x(t) |
|---|---:|---:|---:|
| `air_high_g_active` | 0.700220 / 1.589597 | 0.772787 / 1.458540 | 0.801462 / 1.392514 |
| `air_wide_g_active` | 0.700220 / 0.845614 | 0.772787 / 0.637454 | 0.801462 / 0.540961 |
| `air_high_left_g_active` | 0.511323 / 1.322101 | 0.575811 / 1.275178 | 0.601314 / 1.245363 |
| `air_low_right_g_active` | 0.943089 / 1.186866 | 1.026042 / 0.856189 | 1.062453 / 0.705099* |

G 的 `.81 m` 平面是全身与球外接范围的保守 deadline，不能称为真实前碰撞面。另一列在私有数据中载入原 qpos，仅调用 `mj_kinematics`，对编译 native mesh 顶点及原 box/cylinder 等几何计算世界 x 最前支撑界，再减球半径 0.11 m。它也是无限投影平面，前端极值可能来自不同 y/z 的形状，**穿越该平面不证明接触**；真实接触要看原接触记录。

`air_low_right` 在绝对 t=3.050 s（发球后 1.050 s）已有原 `right_shoulder_roll_link` 的 native 有效接触，随后仍进球。星号中心值是**接触后的真实轨迹**，不能作为未受扰动的来球。配对 absent 球对同一 G 中心参考路径的事后穿越为发球后 1.058858 s、z=0.706739 m，仅用于反事实诊断。其余三例到这些平面前没有原机器人接触，原球轨迹与 absent 相符。

原始来源精确路径（每目录内 `summary.json`、`trajectory.npz`，各文件哈希与插值样本索引见 JSON）：

| 案例 | G 目录 | 配对 absent 目录 |
|---|---|---|
| air_high | [G](/path/to/booster/reports/prediction_validation/air_high_g_active/trajectory.npz) | [absent](/path/to/booster/reports/positioning_validation/air_high_absent/trajectory.npz) |
| air_wide | [G](/path/to/booster/reports/prediction_validation/air_wide_g_active/trajectory.npz) | [absent](/path/to/booster/reports/positioning_validation/air_wide_absent/trajectory.npz) |
| air_high_left | [G](/path/to/booster/reports/prediction_validation/air_high_left_g_active/trajectory.npz) | [absent](/path/to/booster/reports/prediction_validation/air_high_left_absent/trajectory.npz) |
| air_low_right | [G](/path/to/booster/reports/prediction_validation/air_low_right_g_active/trajectory.npz) | [absent](/path/to/booster/reports/prediction_validation/air_low_right_absent/trajectory.npz) |

## 发球后才启动冻结的 2 s rise

[冻结门槛](/path/to/booster/reports/defense_validation/h0_gate.json)的初始动作是 0.05 rad、2 s quintic rise、2 s hold、2 s return。曲线为 `s(u)=10u³−15u⁴+6u⁵`。这里假设 H1 通过后再设计发球响应，未将 H1 的无接触平衡实验当成已验证的球防守动作。

原循环在发球前先选命令；首个 50 Hz postlaunch 观察为 t=2.020 s。但 t=2.000 s 已缓存发球前的 invalid forecast，10 Hz G 刷新首次看到发球后的当前状态实际在 **t=2.100 s**。四例原 high-level records 都直接验证了这一点。

[H0 时延](/path/to/booster/reports/defense_validation/predictor_latency.json)的 `air_center_g_active` 当前状态完整刷新 median/p95/max 为 21.302/21.691/21.844 ms；这里仅引用为空中球 runtime 参考，**没有声称逐例实测或最坏耗时保证**。在假设仿真时序映射为实际控制时序的预算计算中，以实际 100 ms 观察等待加 median runtime，动作最早在约 t=2.121302 s 开始，完整 rise 约 t=4.121302 s 才结束。原离线仿真并未自动把这段 wall runtime 加进 `data.time`；这只是设计预算。

| 案例 | native 前投影时 rise 命令幅度比例 | 中心时 rise 命令幅度比例 | 中心时初始 0.05 rad 残差 |
|---|---:|---:|---:|
| air_high | 19.88% | 22.00% | 0.010999 rad |
| air_wide | 19.88% | 22.00% | 0.010999 rad |
| air_high_left | 8.10% | 9.33% | 0.004663 rad |
| air_low_right | 41.12% | 44.50% | 0.022248 rad |

这些是冻结插值曲线的**命令比例**，不是实际关节响应或碰撞覆盖。H1 hold 最后 1 s 的响应门槛不能替代 rise 瞬态跟踪证据。即使等待/runtime 都为零，四例穿越也均早于发球后 2 s；这一结论只能说明完整姿态未准备好。部分 rise、默认姿态或原 native 肩已可能接触，是否有效挡球仍需未来通过门槛后独立验证。

## 分开判断高度、侧向覆盖和既有动作限制

[H0 native 几何审计](/path/to/booster/reports/defense_validation/geometry_audit.json)的 13 个小角度静态候选最高臂几何 z 为 0.761587–0.766910 m，参考 base z=0.553690 m。它描述这一组固定 root 静态姿态，不能直接充当所有动态姿态或任意抬臂的上界。

另有与臂关节角及 root 旋转无关的保守上界 `arm_top_z ≤ instantaneous_trunk_z + 0.663224298 m`，来自 native 几何与臂链偏移的三角不等式。在每个实际原穿越时刻，使用球底 `ball_z−0.11` 与原实际 base z 对比：

| 案例 | 中心时原 base z | 球底−通用臂上界 | 在此时、此 root 高度排除垂直重叠？ |
|---|---:|---:|---|
| air_high | 0.550932 | +0.068357 m | 是 |
| air_wide | 0.550932 | −0.783196 m | 否 |
| air_high_left | 0.550931 | −0.078791 m | 否 |
| air_low_right | 0.550916 | −0.619042 m | 否；已有原 native 接触 |

`air_high` 在 native 前投影时也有 +0.134384 m 垂直间隔；中心时只有 base z 达到约 0.619289 m，通用界才不再排除重叠，达到该高度仍不保证接触。这些是**原测量 root 高度和特定时刻的条件结论**，不是对新动作改变 root、球继续下降或整个飞行窗口的全局不可能证明。

`air_high_left` 虽然球底明显高于小候选静态高度，但通用任意臂界已经不排除重叠；不能把小候选结果扩展为所有抬臂不可能。`air_wide` 中心球 y=0.600 m、原 base y约0.00119 m，主要是侧向覆盖/走位问题；原 G 首次刷新 target 处冻结几何可覆盖，current/decision 处不可覆盖且 reachability=false，不能归因于高度绝对不可达。`air_low_right` 原 G 在已有 native 肩发生接触却仍进球，说明接触与阻止进球必须分开评价。

G 仅执行原姿态/原侧移动作，未执行新臂动作。这份诊断既不能为未实施的新动作预设成功，也不能为它预设失败。H1 的平衡、实际响应、安全和恢复证据通过后，根任务再决定是否做 H4，以及是否需要不同的冻结动作时序或经过预先声明的发球场景。原未来轨迹只能用于本次 posthoc 检查，不能进入在线控制器或用来提前启动响应。
