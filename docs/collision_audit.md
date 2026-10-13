# E0：官方 K1 碰撞几何与 QVEL 警告审计

审计对象是官方 `K1_22dof.xml` 经现有 `build_scene` 组合后的完整自由基座场景。脚本通过临时 XML 文件路径编译模型，加载原有 `k1_walk` 控制器初始化，再运行 2 秒零速度命令。没有发射球，也没有机器人受球冲击。静止球可以与地面接触。

**结论：原报告的 QVEL 警告来自故意制造不稳定状态的测试；官方 K1 已有足、腿、躯干、手臂和头的有效碰撞几何，可继续官方模型的冲击测试。几何只是局部代理，不能据此声称视觉表面都有连续碰撞覆盖。**

## 1. 警告来源：三条独立证据

原 `reports/goalkeeper_validation/tests.txt` 记录 `161 passed` 后出现：

```text
WARNING: Nan, Inf or huge value in QVEL at DOF 0. The simulation is unstable. Time = 0.0000.
```

1. 单独运行 `tests/test_goal_detection.py::test_mujoco_silent_unstable_reset_is_error`，得到 **1 passed** 和完全相同的一条原生警告。该测试使用 **机器人 absent 场景**，显式给球设置 `(1e100, 0, 0)` m/s；它验证 `run_shot` 抛出含 `mjWARN_BADQVEL` 的 `RuntimeError`。这里 DOF 0 是球的自由关节速度地址，不是 K1 躯干。
2. 在新增 E1 测试之前，排除该故意输入用例运行原套件，得到 **160 passed, 1 deselected**，原生警告行数 **0**。这是原 161 项的完整剩余集合，没有删改测试或屏蔽 MuJoCo 警告。
3. 独立直接调用 `mj_step`，绕过 `run_shot` 的检查层：`mj_forward` 后 BADQVEL 计数仍为 0；首个 `mj_step` 后 `data.warning[mjWARN_BADQVEL].number` 为 **1**，时间为 `0.001` s。MuJoCo 已把球速度改为 `(0, 0, -0.00981)`，`qpos/qvel` 都重新有限。这说明只检查有限性会误把自动重置后的状态当成功；随后单独调用原 `run_shot`，仍得到 `MuJoCo unstable state: mjWARN_BADQVEL` 异常。

证据日志：

- `reports/collision_audit/isolated_warning_test.txt`
- `reports/collision_audit/baseline_without_intentional_warning.txt`
- `reports/collision_audit/native_audit.txt`

正常自由基座全场景的 2 秒零命令验证完成 **2000 个物理步 / 100 个策略步**。8 项原生 warning 初始及最终计数均为 0，状态有限，机器人与球接触样本数为 0。因此历史单条故意警告不能作为正常基线不稳定的证据。

## 2. 以编译模型为准的碰撞资格

全场景共有 **103 个 geom**。K1 子树有 **44 个**，其中 **20 个**可与球碰撞，**24 个** `contype=0, conaffinity=0`，仅用于外观。球和全部 20 个有效 K1 geom 的掩码均为 **1/1**。有效掩码规则是：

```text
(robot.contype & ball.conaffinity) != 0
OR (ball.contype & robot.conaffinity) != 0
```

编译模型的 `npair=0`、`nexclude=0`，没有显式接触对覆盖或 body 排除。球拥有独立自由关节，不与 K1 焊接，不属于 K1 的父子 body；默认同 body/父子过滤不阻止这些球–机器人接触。透明度和显示 group 不决定碰撞资格：透明的躯干箱体及足箱体仍有效；可见的官方 STL 外观网格中许多已通过 0/0 掩码禁用。

机器清单 `reports/collision_audit.json` 包含全场景每个 geom 的名称、ID、body、类型、编译后局部位置/姿态/尺寸、初始化世界位置/旋转、源 XML 属性、网格名、掩码、视觉禁用标识和球碰撞资格。编译后的 K1 几何逐条与独立编译的官方 XML 比对，类型、body ID、局部位置、姿态、尺寸及掩码完全一致。

## 3. 部位覆盖与缺口

| 部位 | 有效 geom | 官方形状 | 可测试范围与局限 |
| --- | ---: | --- | --- |
| 躯干 | 2 | 箱体、圆柱 | 主干箱体和下部短圆柱；不是整个躯干视觉网格 |
| 头 | 2 | `Head_1`、`Head_2` 网格 | 有效凸网格碰撞；不保证保留视觉凹面 |
| 左臂 | 3 | 肩 roll 圆柱、`Left_Arm_3/4` 网格 | 肘/远端臂可测试；肩 pitch 根部只有外观 |
| 右臂 | 3 | 肩 roll 圆柱、`Right_Arm_3/4` 网格 | 肘/远端臂可测试；肩 pitch 根部只有外观 |
| 左腿 | 4 | hip roll、hip yaw、knee 圆柱及短胫骨箱体 | hip pitch、ankle pitch 没有有效 geom；短代理不覆盖整条腿 |
| 右腿 | 4 | hip roll、hip yaw、knee 圆柱及短胫骨箱体 | 同左腿 |
| 左足 | 1 | 透明箱体 | 盒形足底代理；足外观网格禁用 |
| 右足 | 1 | 透明箱体 | 同左足 |

有效几何名称如下。位置是所属 body 的局部位置；箱体尺寸为半长，圆柱第二个尺寸为半长度。网格的完整编译尺寸和姿态在 JSON 中，它们不是可直接当作箱体使用的尺寸。

| 名称 | body | 类型 | 局部位置 m | 尺寸 m / 网格 |
| --- | --- | --- | --- | --- |
| `k1_geom_1` | trunk | box | 0, 0, 0.1 | 0.06, 0.08, 0.1 |
| `k1_geom_2` | trunk | cylinder | 0, 0, -0.06 | r=0.05, half=0.05 |
| `k1_geom_5` | aahead_yaw_link | mesh | 见 JSON 编译中心偏移 | Head_1 |
| `k1_geom_7` | aahead_pitch_link | mesh | 见 JSON 编译中心偏移 | Head_2 |
| `k1_geom_10` | left_shoulder_roll_link | cylinder | -0.007, 0, 0 | r=0.035, half=0.034 |
| `k1_geom_12` | left_elbow_pitch_link | mesh | 见 JSON 编译中心偏移 | Left_Arm_3 |
| `k1_geom_14` | left_elbow_yaw_link | mesh | 见 JSON 编译中心偏移 | Left_Arm_4 |
| `k1_geom_17` | right_shoulder_roll_link | cylinder | -0.007, 0, 0 | r=0.035, half=0.034 |
| `k1_geom_19` | right_elbow_pitch_link | mesh | 见 JSON 编译中心偏移 | Right_Arm_3 |
| `k1_geom_21` | right_elbow_yaw_link | mesh | 见 JSON 编译中心偏移 | Right_Arm_4 |
| `k1_geom_24` | left_hip_roll_link | cylinder | 0, 0, 0 | r=0.036, half=0.047 |
| `k1_geom_26` | left_hip_yaw_link | cylinder | 0, 0, -0.03 | r=0.04, half=0.033 |
| `k1_geom_28` | left_knee_pitch_link | cylinder | 0, 0, 0 | r=0.045, half=0.035 |
| `k1_geom_29` | left_knee_pitch_link | box | 0, 0, -0.13 | 0.03, 0.03, 0.05 |
| `k1_geom_32` | left_ankle_roll_link | box | 0.026, 0, -0.02 | 0.09, 0.035, 0.018 |
| `k1_geom_35` | right_hip_roll_link | cylinder | 0, 0, 0 | r=0.036, half=0.047 |
| `k1_geom_37` | right_hip_yaw_link | cylinder | 0, 0, -0.03 | r=0.04, half=0.033 |
| `k1_geom_39` | right_knee_pitch_link | cylinder | 0, 0, 0 | r=0.045, half=0.035 |
| `k1_geom_40` | right_knee_pitch_link | box | 0, 0, -0.13 | 0.03, 0.03, 0.05 |
| `k1_geom_43` | right_ankle_roll_link | box | 0.026, 0, -0.02 | 0.09, 0.035, 0.018 |

可选官方冲击靶为躯干 `k1_geom_1`、胫骨 `k1_geom_29/40`、足 `k1_geom_32/43`、手臂 `k1_geom_12/19`。这些类别均已有官方代理，当前测试范围无需补充碰撞模型。若后续要求肩根、整段大腿或全部视觉表面的连续覆盖，则需要单独界定补充模型及其结论，不能与官方模型结果混为一谈。球半径为 0.11 m，接触其他邻近代理的可能性必须由后续实际接触记录判定，不能仅凭发射方向归类。

## 4. 复现与原文件保护

```sh
.venv/bin/python scripts/audit_collisions.py
```

默认重做编译审计、静态清单、无冲击 2 秒控制验证以及独立故意警告探针，并沿用已保存的测试日志。要重新运行警告来源测试和当前套件，可加 `--reproduce-tests`；新增后续阶段测试后，总通过数会相应增加，本文的 160+1 是 E1 新增前保留的原套件证据。

脚本生成 `reports/collision_audit.json`。若需把 C 层警告输出保存为同样的证据日志，可执行：

```sh
.venv/bin/python scripts/audit_collisions.py > reports/collision_audit/native_audit.txt 2>&1
```

官方 XML、官方 `k1_walk.pt`、球门 URDF 的 SHA-256 审计前后完全一致，哈希值保存在 JSON。临时组合场景在审计结束后删除，没有改动官方模型或 A–D 报告、既有测试和既有模块。
