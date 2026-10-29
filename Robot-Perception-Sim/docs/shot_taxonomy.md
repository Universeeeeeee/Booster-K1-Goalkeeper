# G4 轨迹与事件测试库

30 个确定性 golden cases 已在原生 Isaac Sim 4.5 / PhysX 执行，每例两次；60 份最终记录同例逐字节一致。终局为 **20 Goal、4 Miss、6 Timeout**。复用 G1 场景、原生步进器 `scripts/run_g1_physx.py` 和父工程未修改的 `GoalDetector`，不宣称解决了 G1 的跨引擎接触差异。

三个 catalog 都是已运行的固定输入，而名称仅是检索键。标签由 native collider 接触、实际球速度和原 `GoalDetector` 计算；例如早期名为 `bounce_one` 的尝试实际有三次反弹，保留于失败/调参归档，未按名称强行标一弹。

| Catalog | 数量 | 实际证据 |
|---|---:|---|
| `config/scenarios/g4_golden.json` | 26 | 低/中/高×左/中/右九条无地面接触直接进球；真实1/2/3次地面反弹后进球；左右柱和横梁接触后Timeout；左右射偏、射高、边线Miss；力度不足Timeout；四组角速度及反向球门斜射 |
| `config/scenarios/g4_frame_glance.json` | 1 | 内侧擦柱，0.243s原生门柱接触，后续整球越线Goal |
| `config/scenarios/g4_robot_contacts.json` | 3 | 两例静态K1身体接触后Timeout，一例0.169s擦右肘/手臂 `k1_geom_21` 后Goal；接触并非扑救成功，`robot_contact_is_save=null` |

轻量逐例事件/哈希见 [主验收](g4_acceptance.json)、[擦柱验收](g4_frame_glance_acceptance.json)、[机器人验收](g4_robot_acceptance.json)。初始位置、角速度、前10ms独立解析重力检查的最大速度偏差和实际数据目录也保存在这些文件。固定seed=0；没有用随机搜索结果替换已冻结的初态。

## 来源与数据契约

- `physx_physical_shot`：动态球与原G1地面/球门/静态K1碰撞。G1原生接触报告保存位置、法线、冲量、分离距离、FOUND/PERSISTS/LOST、步前/步后时间及球速度；没有合成接触响应。
- `procedural_kinematic`：按显式重力抛物线生成 commanded poses，只用于画面/感知诊断；不处理地板或门框碰撞，可能穿过物体。输出 `physical_collision_response=false`，不得解释为物理进球验证。
- `measured_csv_replay`：读取严格单调、有限值的世界系球心样本。CSV列为 `time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s`；保留文件SHA和来源真实性未独立验证标记。不填补时间缺口，不推断碰撞。当前只有明确命名的 synthetic fixture 测试，没有收到真实实球CSV。

坐标单位米、秒、m/s、rad/s，world +Z向上，球半径0.11m，净门口3×2m，门线x=±11m。`GoalDetector` 对整球越线和净口条件插值，首次Miss保持锁存，碰框本身不产生Goal。

`trajectory/events.py` 是原检测器的 adapter。`contact_begin` 保留来源是FOUND还是初始化后已有PERSISTS，时间为步前下界；raw数据同时保留两端。`contact_end` 来自native LOST，并保存步前/后速度；LOST通常没有接触点，位置为null，不能补造。`ground_contact_count` 计算独立接触onset，`ground_bounce_count` 要求实际地板高度附近的竖直速度由负转正，且该步有native ground contact证据，两者独立。终局仍只为Goal/Miss/Timeout；robot_contact作为独立布尔量和事件序列，不覆盖后续结局。所有混合事件按 `time_s` 排序，超时也有显式事件。

## 几何与可复现性

`load_catalog`拒绝非有限状态、地面初始穿透及超出保守无门框初始区域的发射位置。全部发射点在|x|≤10、|y|≤6，远离x=±11的正面框；没有使用严格t=0窄相位查询的说法。

静态机器人由 `build_g4_static_robot.py` 从G1完整USD另存：保留23个link初始世界姿态/几何/材料，只删除机器人RigidBody/Articulation API和关节，球保持动态。没有固定自由基座来冒充站立策略。对三个实际发射点，独立USD world AABB球-机器人保守距离前两例分别1.332718m、1.334097m，第三例详见审计JSON，均大于球半径。见 [静态几何审计](g4_static_robot_audit.json)。审计USD SHA与实际运行scene SHA一致。

导出器对所有正面框中心平面穿越做保守swept检查：若球半径膨胀后的正面柱/横梁区域被穿越，附近3.1ms内必须有**对应球门及对应正面部件**的FOUND/PERSISTS证据，LOST或其他部件不能代替。30例未发现无报告穿框。该检查不等同任意USD几何的连续碰撞证明，也不覆盖门后支撑的全窄相位。

原生runner每步验证1ms physics callback，导出器再以实际初始速度和重力独立检查首10ms的vz，避免同一API写入/读回自洽而推进时钟错误。`native_manifest.stepping` 继承G1通用文案；本批没有运行名为free_fall的用例，独立证据是逐例 `first_10ms_max_ballistic_vz_error_m_s`，不能把通用文案当新增自由落体实验。

机器人raw记录的 `robot_mode=absent`、`robot_contact=false` 是复用G1 runner的历史固定字段，在静态机器人运行中并不正确。最终验收以catalog + 对应scene SHA / static audit和native K1 collider证据归一化为 `static_collision_fixture` 与真实robot_contact；原始记录保持原样以便审计。

## 执行证据与复现

远端独立目录 `/home/lizhengran/isaacsim-workspace/g4_20261029`，仅GPU0，PhysX tensor device为CPU，未做GPU物理/渲染性能声明。旧G1/G2/G3文件未改；所有输出新建。日志退出码均为0，实际runner SHA为 `02162110bc9b19d77baa7f5e0affb523b58b8ff7b140c46ccb66c606e2bdbff5`，原GoalDetector SHA为 `b3556b8e6d654007da9e610e61988010f73911adf0b1c121166ea08b5f054830`。

最终原始数据本地位于 `results/g4/{physical_v2_20261029,bounce_two_20261029,frame_glance_20261029,robot_contacts_20261029,robot_glance_edge_20261029}`，远端对应 `physical_v2,bounce_two,frame_glance,robot_contacts,robot_glance_edge`。`bounce_two` 是对主catalog唯一 `bounce_repeated` 的最终替代运行；导出时显式指定override，逐例校验raw shot与最终catalog完全相同，保留两套native manifest。机器人额外保留三组16条确定性侧向偏移诊断（各重复两次）；仅 `body_glance_y630` 被冻结到最终catalog，其原生K1接触后Goal，不把其他0-contact Goal当成碰身后进球。用robot_glance_edge显式override导出。初次 `physical`、主运行中未达二弹的旧case及其protocol/log完整归档，不混成最终证据。

在独立远端目录执行新的physical数据目录（GPU0需先空闲）：

```sh
CUDA_VISIBLE_DEVICES=0 /home/lizhengran/software/isaac-sim-4.5.0/python.sh /home/lizhengran/isaacsim-workspace/g1_migration/run_g1_physx.py --scene /home/lizhengran/isaacsim-workspace/g1_migration/absent_v6.usda --protocol g4_golden.json --source-root /home/lizhengran/isaacsim-workspace/g1_migration --output-dir physical_new --gpu 0
```

擦柱把protocol改为 `g4_frame_glance.json`。机器人将scene改为独立生成的 `static_robot.usda`、protocol改为 `g4_robot_contacts_final.json`（本地最终catalog的远端副本）。USD-only builder需要G1文档中的USD Python环境，调用示例：

```sh
python.sh build_g4_static_robot.py --source /home/lizhengran/isaacsim-workspace/g1_migration/robot_v6.usda --catalog g4_robot_contacts_final.json --output static_robot_new.usda
```

本地从子工程目录执行，所有输出必须为新文件：

```sh
../.venv/bin/python -m pytest -q -c pytest.ini tests/test_g4_trajectories.py
../.venv/bin/python scripts/export_g4_trajectories.py --source-root .. --physical-dir results/g4/physical_v2_20261029 --override-dir results/g4/bounce_two_20261029 --output results/g4/acceptance_new.json
../.venv/bin/python scripts/export_g4_trajectories.py --source-root .. --catalog config/scenarios/g4_robot_contacts.json --physical-dir results/g4/robot_contacts_20261029 --override-dir results/g4/robot_glance_edge_20261029 --output results/g4/robot_acceptance_new.json
../.venv/bin/python scripts/export_g4_trajectories.py --source-root .. --procedural-case direct_middle_center --output results/g4/procedural_new.json
../.venv/bin/python scripts/export_g4_trajectories.py --source-root .. --measured-csv /absolute/path/real_samples.csv --output results/g4/measured_new.json
```

本轮10个针对性测试通过，覆盖不合法初态、未整球越线、初始出界锁存、接触≠进球/扑救、接触与反弹独立、接触结束/超时事件、时钟非法CSV及穿框证据误用；procedural/CSV两个CLI也实际执行。

## Gate边界

30例确定性物理库、事件序列、终局独立性、三种来源接口及正面穿框/初态保守检查已有证据。球碰身后进球和未进球均已有native物理证据；固定K1不是运动控制策略。物理录制没有相机，FOV/遮挡/丢帧标签为unobserved，视觉诊断由G5独立图像用例提供，不以物理轨迹臆测。旋转只有接触动力学效果，没有空气阻力/Magnus，不能宣称复杂旋转自由飞行真实性。真实CSV内容与完整任意几何窄相位尚未验证。因此不把这组有限样本声明为G4所有推荐矩阵或真实足球物理完全通过。
