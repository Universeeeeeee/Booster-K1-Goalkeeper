# H3：移停后抬臂、恢复行走与低速 native 臂冲击

速度标注说明：本报告的 1/2 m/s 工况名指水平初速度 vx；弹道瞄准的初始 vz 分别为 1.4715/0.73575 m/s，完整速度大小为 1.779132/2.131039 m/s。冻结协议保存完整三维向量，接触测量与物理执行不变。

H3 的四个配对组全部通过：两向移停和 1/2 m/s 冲击，各原步行 baseline 与 both-roll 臂残差两遍，共 **16 次正式实验**。每种模式的两遍完整轨迹 SHA-256 一致，所有安全、实际关节响应及恢复门槛通过，没有 guard、fall、数值错误、新自碰撞或非足部地面接触。原受保护 840 份文件在结束后再次独立核验一致。

这是既有 native 臂碰撞几何上的低速冲击与运动恢复验证，**不是高球防守成功证明，也不是实时控制证明**。它没有执行 H4，没有新增碰撞壳、修改官方 actor/history/filter/PD、KP/KD/effort，或修改 A–G。

## 冻结方案与复现

H1 全部零路径、0.05/0.15/0.30 rad 四家族两遍通过后，才冻结并执行 H3。选用已通过的 both-roll ±0.15 rad：左 shoulder roll +0.15、右 −0.15；保持原 `DefensePulse` 2 s quintic rise、2 s hold、2 s return，峰值残差 slew 0.140625 rad/s。物理 1 ms、策略 20 ms 不变。

- [冻结 protocol](/path/to/booster/reports/defense_validation/followup/protocol.json)：时序、固定发球输入、H1 依据、源码/helper 哈希与安全门槛。
- [正式 results](/path/to/booster/reports/defense_validation/followup/results.json)：四组、16 次实验与重复哈希，`gate_passed=true`。
- [抬臂前停稳 addendum](/path/to/booster/reports/defense_validation/followup/pre_arm_stop_addendum.json)：冻结 protocol 不变，独立对原 E 数组补查 t=3..6。
- [交付结果独立核验](/path/to/booster/reports/defense_validation/followup/delivered_verification.json)：重载 16 份完整 archive 核验内部哈希，检查每个 tick 的 `nominal+offset==delivered`、未覆盖目标逐位一致，附实际接触和恢复行走值。

复现命令（已有正式结果会先核验并复用；重做实验使用另一个输出目录）：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_defense_followup.py --freeze-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_defense_followup.py --phase motion
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/validate_defense_followup.py --phase impact
```

实现：[defense_followup.py](/path/to/booster/k1_goalkeeper/defense_followup.py)、[validate_defense_followup.py](/path/to/booster/scripts/validate_defense_followup.py)。复用原 `PostFilteredArmController`、`DefensePulse`、E `run_impact`、H `contact_state`、`recovery_metrics`。E 完整 qpos/qvel/torque/contact 记录保留；仅新增策略 tick 的 nominal/delivered/offset 和逐物理步 active self/nonfoot/external pair 行，没有重开发策略 history 或第二套完整 q/torque recorder。

安全阈值沿用 H1：base z≥0.42 m、tilt≤15°、选中非零残差目标在 XML 内缩 0.05 rad 范围且在 PD 前检查、实际关节不越 XML 范围、臂 raw PD 不连续超过 0.10 s 饱和、原实际 effort 上限与 1e−5 Nm 容差。非有限状态归真数值错误；主动 balance/contact/target guard 单独标记为 censored。冲击组允许 robot-ball 接触，robot-goal 与其它外部接触仍停止。自碰撞和非足部地面接触按同一物理步配对 baseline 比较。

## 两向移停、抬臂与恢复后行走

两个方向均执行 body vy=±0.2 m/s 的 t=2..3 走位；t=3..6 零命令；t=6..12 臂 pulse；t=12..17 恢复；t=17..18 body vx=0.4 m/s；t=18..23 再恢复。baseline 使用相同全部 body 命令，不加臂残差。

原 `recovery_metrics` 对全部八次 t=3..6 均 verified：+vy 的停稳延迟 0.6 s，−vy 为 0.4 s，均早于 t=6 抬臂。这是独立追加前置检查，没有改冻结方案。正式 t=12..17 与 t=18..23 恢复也全部 verified；有臂例相应停稳延迟分别为 0 与 0.4 s。

| 两向有臂例（r0/r1 相同） | hold 最后 1 s 左/右 signed 响应比例 | t≥2 最低 base z | t≥2 最大 tilt |
|---|---:|---:|---:|
| vy=+0.2 | 1.171601 / 1.147080 | 0.512397 m | 2.681912° |
| vy=−0.2 | 1.179357 / 1.150862 | 0.517840 m | 2.671834° |

响应使用 E 原 qpos 与相同时间 baseline 的有符号 median 差，除以 ±0.15 rad；两肩均超过冻结的 0.5 门槛。这一比例包括策略对实际姿态扰动的反应，不是纯残差到实际角度的静态比例。

恢复后 t=17..18 的实际行走：

| 原先走位方向 | baseline / arms 平均 body vx | baseline / arms 在 t17 body x 的位移 |
|---|---:|---:|
| vy=+0.2 | 0.168217 / 0.168801 m/s | 0.167913 / 0.168506 m |
| vy=−0.2 | 0.189305 / 0.193556 m/s | 0.189026 / 0.193269 m |

实际正向运动与 matched baseline 接近，支持恢复后可以再次行走。这个仅 1 s 的加速段平均速度低于命令 0.4 m/s，不能宣称已稳定跟踪 0.4 m/s。随后 t=18..23 最后 2 s 的有臂平均 `(body vx, body vy, yaw-rate)` 分别为 `(0.0000646, 0.0001144, −0.0006735)` 与 `(0.0001115, −0.0004102, 0.0019334)`，满足原恢复门槛。

## 两档低速 native 冲击

固定取已通过 H1 `both_roll_0.15_r0` 原 trajectory 的 t=6 qpos，在独立数据中只做 FK，然后调用原 E `_launch` 指向 `left_elbow_yaw_link` 的原 native mesh `k1_geom_14`（组合场景 geom ID 44）。冻结位置为 `(9.884708, −0.258913, 0.503166)` m；速度输入分别 `(1,0,1.4715)` 与 `(2,0,0.73575)` m/s，body 速度命令为零，t=6 发球，t=14 结束。baseline 和有臂使用同一 launch。

这份原 H1 pose 用于**预先冻结实验输入**，没有作为在线控制未来状态。1 m/s 两遍必须先通过实际 native 目标接触、安全、响应、恢复与重复哈希，才执行 2 m/s。

下表 r0/r1 数值相同。active duration 是所有目标 geom 的正法向力 episode 时长之和；span 是从首个 episode 开始至最后结束的时间，包含间歇，二者不混用：

| 发球速度/模式 | 首接触绝对 t | active duration / span | 法向冲量 | peak force | 最后目标接触后世界 vx |
|---|---:|---:|---:|---:|---:|
| 1 m/s baseline | 6.328 s | 0.023 / 0.050 s | 0.502699 N·s | 70.465 N | +0.150160 m/s |
| 1 m/s arms | 6.316 s | 0.043 / 0.064 s | 0.584831 N·s | 66.015 N | −0.033115 m/s |
| 2 m/s baseline | 6.171 s | 0.080 / 0.119 s | 1.025017 N·s | 102.867 N | −0.014842 m/s |
| 2 m/s arms | 6.162 s | 0.101 / 0.131 s | 1.153958 N·s | 98.235 N | −0.368829 m/s |

两档有臂的目标作用于球的总世界冲量分别为 `(-0.444239, +0.043536, +0.475733)` 与 `(-1.018605, +0.058439, +0.598877)` N·s。球最初世界 vx 为正；第一接触 episode 结束时有臂 vx 仍为 +0.413284/+0.849264 m/s，多次接触后才变为表中的负值。因此不能把“首次碰到球”直接写成“立即反弹”。

四种模式全部确认为 `left_elbow_yaw_link` 真实力承载接触，没有 miss/unverifiable。四种模式在 t=14 前均 `goal=false`、`exit_event=null`，且 baseline 也如此。**H3 是指定 native 臂的低速冲击实验，不是射门阻挡成功率评估**；本报告没有将这些未进球结果统计为新增扑救。

1/2 m/s 有臂的 t=6..7 左/右实际 signed 响应比例为 `1.189839/1.159173`、`1.180659/1.167064`。t=9..14 原恢复检查全部 verified，停稳延迟为 0；有臂最后 2 s 平均 `(body vx, body vy, yaw-rate)` 分别为 `(0.0000145, −0.0000137, 0.0000820)` 和 `(0.0000151, −0.0000326, 0.0001655)`。两档有臂最低 base z 为 0.550839/0.550610 m，最大 tilt 为 1.009157/1.060795°。

## 结论边界

H3 证据支持：在冻结的既有 native 模型、官方步行策略与这些时序/低速输入下，±0.15 rad 双肩 roll 可以在移停后产生实际响应、返回后恢复行走，并承受已测到的 native 臂冲击且恢复。它不证明高球覆盖、发球后 2 s rise 能及时完成、其它姿态/速度的冲击稳定性或硬件实时性。[原 G 空中球事后诊断](/path/to/booster/docs/defense_shot_feasibility.md)仍独立成立；后续 H4 的在线决策与实际防守结果由根任务另行验证。
