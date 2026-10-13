# 阶段 H0：K1 上肢接口审计

本轮仅静态审计并新增本文件及 `reports/defense_validation/interface_audit.json`。没有执行策略推理、`mj_step`、动作混合或 H1。读取了全局 AGENTS.md；项目内未发现额外 AGENTS.md。

结论：当前固定提交的官方 K1 步行策略直接控制全部八个臂关节，臂关节位置、速度和上次策略动作进入观测历史。当前本地官方子集没有独立 upper-body pose、关节 mask、优先级或多策略仲裁接口。因此可以把“22维目标输出之后、PD输入之前”作为新 H 模块的实验边界，但不能把臂控制称为与步行无耦合的安全覆盖。

## 证据范围与独立核验

官方代码来源是 `https://github.com/BoosterRobotics/booster_deploy.git`，固定提交 `7bb1462e2ac742eba5c01659b815e2e99fbe617f`。本地 manifest 列出的22个文件 SHA-256 全部匹配。本地只是经清单固定的子集，没有实机控制器；本报告不推断当前上游或被省略文件的能力。

独立低层证据：只编译官方 `K1_22dof.xml`，未推进物理，得到 `nq=29/nv=28/nu=22`，22个关节和执行器顺序均与 robot 配置一致。只加载 TorchScript 检查权重，未执行推理，actor 首层权重 `[512,690]`、末层 `[20,128]`，独立对应69维观测×10帧历史、20维动作。JSON包含编译后的关节地址、限制标志、参数形状、各源哈希及受保护原测试文件哈希。

## 关节、索引和实际生效参数

所有索引从0开始；源代码行号从1开始。控制22个关节，actor20个关节，不控制头部动作，头部PD仍保持零位。robot臂索引为2–9；policy臂索引为 `[0,1,4,5,8,9,12,13]`，不能把policy前8个动作当成双臂。

下表使用 `K1WalkTaskCfg` 的实际覆盖参数。基础 `K1_CFG` 臂KP=4/KD=1，准备状态又有另一组参数；这些都不是运行步行策略时的臂KP=20/KD=2。XML motor 双臂 `forcerange=±14Nm`，与运行任务软件限幅相同。腿部XML motor范围通常宽于软件 effort_limit，不能把XML上限当成本实验允许限幅。

|robot索引|policy索引|关节|默认rad|KP|KD|扭矩上限Nm|XML角度范围rad|XML行|
|---:|---:|---|---:|---:|---:|---:|---|---:|
|0|—|aahead_yaw_joint|0|4|1|6|[-1.012, 1.012]|56|
|1|—|aahead_pitch_joint|0|4|1|6|[-0.314, 0.794]|61|
|2|0|aaleft_shoulder_pitch_joint|0.2|20|2|14|[-2.932, 1.196]|70|
|3|4|left_shoulder_roll_joint|-1.25|20|2|14|[-1.642, 1.629]|74|
|4|8|left_elbow_pitch_joint|0|20|2|14|[-1.885, 1.885]|79|
|5|12|left_elbow_yaw_joint|-0.5|20|2|14|[-2.242, 0.816]|84|
|6|1|aaright_shoulder_pitch_joint|0.2|20|2|14|[-2.932, 1.196]|93|
|7|5|right_shoulder_roll_joint|1.25|20|2|14|[-1.642, 1.629]|97|
|8|9|right_elbow_pitch_joint|0|20|2|14|[-1.885, 1.885]|102|
|9|13|right_elbow_yaw_joint|0.5|20|2|14|[-0.816, 2.242]|107|
|10|2|left_hip_pitch_joint|-0.15|100|2|30|[-2.958, 2.226]|116|
|11|6|left_hip_roll_joint|0|100|2|20|[-0.375, 1.536]|120|
|12|10|left_hip_yaw_joint|0|100|2|15|[-1.012, 1.012]|125|
|13|14|left_knee_pitch_joint|0.3|100|2|35|[0.0, 2.321]|130|
|14|16|left_ankle_pitch_joint|-0.15|65|1|24|[-0.87, 0.345]|136|
|15|18|left_ankle_roll_joint|0|65|1|15|[-0.345, 0.345]|140|
|16|3|right_hip_pitch_joint|-0.15|100|2|30|[-2.958, 2.226]|152|
|17|7|right_hip_roll_joint|0|100|2|20|[-1.536, 0.375]|156|
|18|11|right_hip_yaw_joint|0|100|2|15|[-1.012, 1.012]|161|
|19|15|right_knee_pitch_joint|0.3|100|2|35|[0.0, 2.321]|166|
|20|17|right_ankle_pitch_joint|-0.15|65|1|24|[-0.87, 0.345]|172|
|21|19|right_ankle_roll_joint|0|65|1|15|[-0.345, 0.345]|176|

关节顺序来源：[assets/booster_deploy/booster_deploy/robots/k1.py:7](/path/to/booster/assets/booster_deploy/booster_deploy/robots/k1.py:7)；policy顺序及实际KP/KD/默认姿态/effort：[assets/booster_deploy/tasks/locomotion/robots/k1/__init__.py:14](/path/to/booster/assets/booster_deploy/tasks/locomotion/robots/k1/__init__.py:14)、[assets/booster_deploy/tasks/locomotion/robots/k1/__init__.py:50](/path/to/booster/assets/booster_deploy/tasks/locomotion/robots/k1/__init__.py:50)。各XML关节、执行器精确行号与范围在JSON逐项列出。

## 观测、动作和历史

一帧69维：`[0:3]`机体角速度、`[3:6]`投影重力、`[6:9]`机体vx/vy/yaw命令、`[9:29]`policy顺序的关节q−默认q、`[29:49]`关节dq×0.1、`[49:69]`上一次裁剪后的actor输出。没有球状态，也没有上肢姿态指令。八个臂关节的q/dq/action全部包含在这三段中。

初始化历史重复第一帧10次；后续roll(-1)并把新帧写入最后一行，按从旧到新flatten成690维。reset将last_action清零、history置空、filtered target恢复默认。源码：[assets/booster_deploy/tasks/locomotion/locomotion.py:113](/path/to/booster/assets/booster_deploy/tasks/locomotion/locomotion.py:113)、[assets/booster_deploy/tasks/locomotion/locomotion.py:156](/path/to/booster/assets/booster_deploy/tasks/locomotion/locomotion.py:156)、[assets/booster_deploy/tasks/locomotion/locomotion.py:65](/path/to/booster/assets/booster_deploy/tasks/locomotion/locomotion.py:65)。

actor动作裁剪±100，经名称映射到22维，action_scale=0.25rad；右肘pitch先减0.2，等价未滤波目标−0.05rad固定补偿。目标滤波为 `filtered[k]=0.2*filtered[k−1]+0.8*raw[k]`。`last_action`保存的是裁剪后的原actor输出，在右肘补偿、缩放和滤波之前。这个固定补偿不是通用upper-body接口。源码：[assets/booster_deploy/tasks/locomotion/locomotion.py:87](/path/to/booster/assets/booster_deploy/tasks/locomotion/locomotion.py:87)、[assets/booster_deploy/tasks/locomotion/locomotion.py:173](/path/to/booster/assets/booster_deploy/tasks/locomotion/locomotion.py:173)、[assets/booster_deploy/tasks/locomotion/locomotion.py:203](/path/to/booster/assets/booster_deploy/tasks/locomotion/locomotion.py:203)。

原策略没有关节目标范围或目标速度限幅。±100动作乘0.25允许±25rad名义偏移；XML关节范围是物理求解约束，不会把目标自动裁剪到范围内。扭矩限幅也不证明动作可行。

## 控制边界及耦合风险

策略周期20ms。场景使用1ms物理步、20步一次推理，每步执行 `clip(KP*(target−q)−KD*dq,±effort)`；官方通用MuJoCo默认decimation10，对应2ms，与场景的1ms有区别。原控制器对22维状态直接切片，现有命名适配器通过joint/actuator名称和transmission核对地址，适合有球等额外关节的组合场景。

候选新H实验：在原 `compute_targets()` 返回后复制22维目标，只修改选中臂slot，再交给原 `apply_control()`；保持头腿目标与**同一次推理**原目标相同。不得修改原 `filtered_dof_target`、原返回张量、`last_action` 或 history。未来的腿目标仍可能因为臂q/dq变化和基座动态变化而改变，所以“只覆盖臂slot”不是全身独立。

原场景在每个policy tick重新写 `scene.targets`，每1ms重新计算PD。因此只写一次targets会被下个tick覆盖；直接写`data.ctrl`会被PD覆盖。`set_reference_qpos()`只在第二份ghost MjData上做FK和渲染，不会驱动机器人。接口来源：[k1_goalkeeper/walk_controller.py:131](/path/to/booster/k1_goalkeeper/walk_controller.py:131)、[k1_goalkeeper/scene_simulation.py:49](/path/to/booster/k1_goalkeeper/scene_simulation.py:49)、[assets/booster_deploy/booster_deploy/controllers/mujoco_controller.py:101](/path/to/booster/assets/booster_deploy/booster_deploy/controllers/mujoco_controller.py:101)。

后处理目标留下明确的历史不一致：last_action是名义策略动作，而真实执行臂目标已不同。最小H1建议保持官方这个名义动作历史语义，记录名义/执行目标和差值；不谎称last_action是执行动作。若把执行目标重构为last_action是另一项观测改动，需要另做消融；直接 `(target−default)/0.25`忽略滤波状态与右肘补偿。解除覆盖也必须缓慢退回，避免跳到仍沿名义策略更新的内部滤波目标。

双臂惯量、重心、足地支持与自碰撞都会耦合到自由基座。肩pitch根部仅有视觉mesh；启用碰撞的臂几何是肩roll cylinder及肘pitch/yaw mesh，并非连续实体包络。看见手臂移动或球未接触均不能独立证明挡球覆盖。来源：[assets/booster_assets/robots/K1/K1_22dof.xml:68](/path/to/booster/assets/booster_assets/robots/K1/K1_22dof.xml:68)。

## H1 受控扰动实验条件（尚未实施）

1. 新建H专用脚本/模块，不改A–G或官方代码。先用零残差wrapper与相同初值的原baseline逐物理步严格比对时间、qpos/qvel/ctrl，以及每次推理的目标、history、last_action、filter。任何差异都阻断非零实验。
2. 根任务最新方案先以0.05rad post-filter**加性残差**作门槛，2s quintic升、2s保持、2s quintic返回，再观察恢复。具备进行这个隔离扰动可行性实验的接口条件；不是独占臂覆盖，也不是官方支持动作。保持零速度、自由基座、无球/球门接触，并保留原obs/last_action/filter/PD、KP/KD/effort、重力、碰撞及1ms/20ms时序。
3. 原生FK已独立核验active task default：左roll**正残差**、右roll**负残差**使肘部向侧面展开并升高。左elbow_yaw-link原点在+0.15rad时y从0.197320809增到0.220264297m、z从1.016448087增到1.025843853m，右侧y镜像。先前仅按关节范围推断left负/right正的方向已在任何动作执行前纠正。FK只改独立预览MjData并调用mj_forward；不证明完整碰撞包络或动态稳定。

初始quintic `10u³−15u⁴+6u⁵`的峰值slew是幅度×1.875/升降时长，0.05rad/2s对应0.046875rad/s。先门槛通过后才考虑0.15rad单左roll、单右roll、双roll、双pitch；全部对应例通过后才另行考虑0.30rad。更大阶段起落时长必须执行前定稿并重算slew，不能默认沿用旧smoothstep方案。所有选中关节均以hold最后1s、相同时间baseline的有符号median Δq≥残差幅度50%作响应门槛，不能用瞬时peak替代。首个不稳停止更大幅度，响应不足阻断升级。每物理步检查`nominal target+residual`落在XML范围内缩0.05rad区域，不能只检查默认姿态。此次可行性实验不增加走位、球碰撞或更大幅度。
4. 同时记录原actor20维输出、名义与执行22维目标、mask/残差、history10×69、每关节q/dq、raw PD/control/`qfrc_actuator`、自由基座姿态/高度/COM、足部/自碰撞、告警和stop原因。检查实际PD输入、实际执行器力矩和真实q的匹配baseline差值，不能只看wrapper getter。

成功标准是零残差严格等价，残差只改指定slot并真正到达PD/力矩/实际关节，头腿同次推理目标不变，稳定且平缓恢复，原文件哈希和原297tests保护不变。报告实际角度误差、恢复、漂移、最大倾角、最低高度及饱和；不以这个受控扰动宣称扑救或独立控制已验证。

## 停止条件

立即停止并保留证据：源/测试哈希改变；零残差不等价；mask/名称/shape错误或未选slot改变；任何非有限动作、目标、状态、力矩或时间；新增BADQPOS/QVEL/QACC/CTRL告警；意外时间重置；官方policy stop。不得调高KP或力矩上限后继续。

沿用物理失败条件：base z<0.3m，或trunk upright z<0.5（倾角>60°）。官方投影重力z>−0.5调用stop，但仍计算本次推理；命名适配器在返回目标前检查stop并抛错。`stop()`只改运行标志；H专用loop必须中止后续PD/物理推进，不能认为stop自动清零扭矩。来源：[assets/booster_deploy/tasks/locomotion/locomotion.py:123](/path/to/booster/assets/booster_deploy/tasks/locomotion/locomotion.py:123)、[k1_goalkeeper/walk_controller.py:137](/path/to/booster/k1_goalkeeper/walk_controller.py:137)、[k1_goalkeeper/velocity_validation.py:300](/path/to/booster/k1_goalkeeper/velocity_validation.py:300)。

根H1方案的保守早停阈值（不是官方认证限值）：settle后倾角>15°、base z<0.42m或新增self-contact立即停止该例与更大幅度升级。另检查目标超XML内缩0.05rad范围、实际q超XML范围；残差超过当前批准阶段幅度，或slew超过由起落函数/时长计算的上限（初始2s quintic门槛为0.046875rad/s）；任一臂raw PD连续>0.10s达到±14Nm，或实际control/torque超过允许上限+1e−5Nm；出现非足部地面接触，或无接触实验中的球/球门接触。每物理步检查；响应不足50%阻断升级，不记作有效动作通过。

本审计没有重跑原297tests，297是父任务给出的受保护基线计数。未修改原tests/A–G/官方文件；manifest验证、模型编译及参数检查属于只读静态证据，H1动态结论仍待实验。
