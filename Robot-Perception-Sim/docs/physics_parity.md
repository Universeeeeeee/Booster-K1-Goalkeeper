# G1 USD / PhysX 迁移与物理对照

本轮完成 G1 资产迁移和限定用例验证，但**完整物理等价 Gate 不宣称通过**：完整机器人场景的材料优先级、滚动/扭转摩擦和撞框后轨迹仍存在差异。G2 保持冻结，未开发 G3、视觉检测、运控或性能优化。业务变更全部位于 `Robot-Perception-Sim/`，基线为 `ba4b57bf8ed6817a93c3080b42e34b1f1d527b8d`。

## 来源与转换

复用 `results/g1/source_contract_20261023T172323_398380Z/` 和 `results/g1/physics_reference_20261023T173052_077419Z/`。源端逐文件 SHA-256 核对无变化，没有重生成五类 MuJoCo 轨迹，没有修改 MuJoCo 参数。

选择从冻结 MJCF 的编译模型直接生成 USD，避免 URDF 再导入引入另一套几何、碰撞和惯量默认值。导出器只执行 `mj_forward`，不推进物理；使用编译网格顶点及对应 geom 变换，避免重复应用网格重心偏移。USD 自包含，不依赖原 STL 的绝对路径。

完整场景为 `results/g1/usd_migration_20261024/robot_v6.usda`，无机器人对照为同目录 `absent_v6.usda`。保留 103 个几何、自由基座、22 个具名关节、关节 armature、23 个机器人刚体共约 19.666 kg。两个无质量头部坐标系随头部运动，没有增加相机负载。球半径 0.11 m、质量 0.43 kg、薄壳主惯量各 0.00346866666667 kg·m²；球门净口为 3×2 m。

保存后重新读取 USD，逐项检查世界位姿、关节两端锚点与轴、限位、质量、COM、主惯量及惯性轴、几何尺寸、网格顶点与拓扑、碰撞标志和过滤关系。故意改变躯干质量和头部关节轴的负向检查均被拒绝。摘要与文件哈希见 [资产验收记录](g1_asset_acceptance.json)。这是序列化资产检查，实际引擎接受情况另由 PhysX tensor 读回核对。

## 接触模型的边界

恢复系数候选为 0.3314737256574807，来自冻结落球首弹顶点：`sqrt((0.2297635655733633−0.11)/(1.2−0.11))`。它只是源数据上的首弹高度估计，不是 MuJoCo `solref/solimp` 的数学等价变换，也不是实球标定。首弹一致性属于样本内观察，撞框等其他场景单独对照。

无机器人场景统一使用滑动摩擦 0.6，对应源球 priority=1 覆盖地面/门框的规则。完整机器人场景保留各 geom 摩擦并采用 max 组合，因此球与门框/部分机器人接触可得到 1.0，而源球优先级规则为 0.6；这个完整场景的材料差异尚未解决。MuJoCo 软接触、滚动/扭转摩擦也没有精确映射。不能把无机器人球测试外推为机器人挡球等价。

使用原生无限 Plane；动态网格使用 PhysX convexHull，碰撞烘焙与 MuJoCo 的实现可能有差异。保留几何和表面颜色，未迁移 checker/skybox 纹理与反射参数。没有运行 walk 控制器或证明自由机器人能站立。

## 时间步故障与修复

首轮 `SimulationContext.step(render=False)` 实际按 1/60 秒推进，却按 1 ms 标记。自由落体在标称 0.1 秒已经落地，独立解析检查揭示了错误。后续检查安装代码发现，`PhysicsContext.set_physics_dt` 写入 USD，但 `get_physics_dt` 从未注册场景的 SimulationManager 返回 1/60 秒默认值，`_step` 又使用这个返回值。

最终使用公开原生 `simulate(dt, time)` / `fetch_results()`，每步验证物理回调返回的 dt，并累加回调时间作为轨迹时钟；另外以自由落体的位置和速度验证实际推进。未修改 Isaac 安装或私有注册表。初始化期间可能推进场景，因此正式记录前重设球的位姿、线速度和角速度；机器人冒烟也显式恢复源零关节初态及零速度。

早期无效数据保留在独立目录，不进入最终对照。运行失败会保存 `failure.json` 并返回非零，避免 Kit 快速关闭掩盖异常。

## 验证与复现前提

执行环境：`lizhengran-server`（`robotic-4090`），原生 Isaac Sim 4.5.0、Ubuntu 20.04、驱动 575.57.08；独立目录 `/home/lizhengran/isaacsim-workspace/g1_migration`。仅使用 GPU0，物理 tensor 后端为 CPU；GPU0 用于 Isaac 启动，不将此称为 GPU 物理或渲染性能验证。未写入 G2 用户空间或升级驱动、系统、SDK。

导出和离线测试需要已保存的冻结源契约/参考结果，以及父工程已有资产和 `.venv`。这些原始结果与 USD 文件保留在本地/目标机，体积较大，不提交 Git；轻量验收摘要提交。新 checkout 应先恢复这些证据，不能以重生成结果替代原冻结参考。

## 实际结果

导出、源契约、物理参考标签和离线比较的针对性测试共 **22 passed**。G2两个脚本SHA与冻结值一致；正式球测试及最终机器人冒烟的执行脚本快照均与各自manifest SHA一致。

正式数据：`results/g1/physx_final_v6/`；每例两次，十份完整记录的同例 SHA 一致，运行退出码 0。独立比较：`results/g1/physics_comparison_20261024T071819_652815Z/comparison.json`。轻量结果和实际运行参数见 [物理验收记录](g1_physics_acceptance.json)。位置误差在两条轨迹重叠时间内插值比较，不外推。

| 案例 | MuJoCo / PhysX 终局 | 最大位置差（m） | 关键观察 |
|---|---|---:|---|
| free_fall | Timeout / Timeout | 0.0000001282 | 无接触，实际加速度与1ms时间步检查通过 |
| ground_bounce | Timeout / Timeout | 0.0119215 | 首接触均约0.471s；反弹4次 / 3次 |
| center_low | Goal / Goal | 0.00130531 | 未碰门框；PhysX首次已有ground接触报告不等同新撞击 |
| post | Timeout / Timeout | 0.926145 | 首次门柱接触0.232s / 0.231s |
| crossbar | Timeout / Timeout | 0.869138 | 首次横梁接触均约0.232s |

落球首弹球心顶点为 0.229763566 / 0.229118899 m，PhysX 低约0.645 mm。接触前样本到顶点的总机械能变化为 −4.083003 / −4.085706 J；包含重力势能与薄壳转动能，以地面z=0为势能零点。整段轨迹差异不只由首弹高度决定，门框碰撞后的水平速度、后续地面接触和缺失的滚动阻尼均会积累差异。

接触时间为1ms分辨率的报告时刻。PhysX contactOffset=1mm，可在分离距离为正时报告接触。center_low首条是初始化后已有接触的 `CONTACT_PERSISTS`（t=0），源MuJoCo首次onset为0.014s，不能将两者解释成提前14ms发生实体冲击。PhysX原始数据另存found/persist/lost、点、法线、冲量和生命周期；冻结MuJoCo仅有onset，无法恢复精确接触时长，报告不编造时长配对。

最终机器人记录为 `results/g1/robot_smoke_contacts_v6/`，退出码0，替代正式球测试目录中较早的机器人初始化冒烟。显式重置后基座z=0.553690016m、22关节位置/速度及基座速度为0；100个1ms步后位移约[−0.00034523,−0.000002475,−0.00200450]m，状态有限且基座确实自由。首个1ms步没有机器人接触报告；只有球地面正分离接触，不能据此声称做过严格的t=0窄相位穿透查询。

实际PhysX tensor读回23个机器人刚体，总质量19.665999465kg；按名称配对，最大单体质量误差4.77e−7kg、主惯量特征值误差7.10e−9kg·m²、关节限位误差1.29e−7rad。运行中的DOF顺序与原22关节数组不同；20策略动作到PhysX顺序映射为 `[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]`，必须按名称映射，不能直接复用源数组索引。

旧 `physx_cases_v1` 为错误时钟数据，v2/v3为被严格检查拒绝的尝试，v4为修正步进后的诊断；均不混入验收。每个最终结果目录保存实际执行脚本快照并与manifest SHA相符，球测试和后续接触冒烟的脚本版本有所不同。大型原始数据在本地与远端同名目录保留。

## 复现命令

本地导出（新文件名，不覆盖冻结证据）：

```sh
.venv/bin/python Robot-Perception-Sim/scripts/export_g1_source.py --output Robot-Perception-Sim/results/g1/source_export_new.json
.venv/bin/python -m pytest -q -c Robot-Perception-Sim/pytest.ini Robot-Perception-Sim/tests/test_g1_source_export.py Robot-Perception-Sim/tests/test_g1_physics_comparison.py Robot-Perception-Sim/tests/test_physics_reference.py Robot-Perception-Sim/tests/test_isaac_source_contract.py
```

将导出JSON、builder/runner、冻结protocol和原样 `k1_goalkeeper/detection.py` 放入独立G1目录。目标机CPU构建（free/absent各一次，输出须不存在）：

```sh
cd /home/lizhengran/software/isaac-sim-4.5.0
export PYTHONPATH="$PWD/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310:$PWD/extsPhysics/omni.usd.schema.physx"
export LD_LIBRARY_PATH="$PWD/extsPhysics/omni.usd.schema.physx/bin:$PWD/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin:$LD_LIBRARY_PATH"
export PXR_PLUGINPATH_NAME="$PWD/extsPhysics/omni.usd.schema.physx/plugins/PhysxSchema/resources"
./python.sh /home/lizhengran/isaacsim-workspace/g1_migration/build_g1_usd.py --source /home/lizhengran/isaacsim-workspace/g1_migration/source_export_new.json --output /home/lizhengran/isaacsim-workspace/g1_migration/robot_new.usda --robot free --restitution 0.3314737256574807
./python.sh /home/lizhengran/isaacsim-workspace/g1_migration/build_g1_usd.py --source /home/lizhengran/isaacsim-workspace/g1_migration/source_export_new.json --output /home/lizhengran/isaacsim-workspace/g1_migration/absent_new.usda --robot absent --restitution 0.3314737256574807
```

在新的目标机shell运行，避免继承上面的CPU工具路径；GPU0空闲后执行（GPU1也被授权，但本轮未使用）：

```sh
cd /home/lizhengran/isaacsim-workspace/g1_migration
CUDA_VISIBLE_DEVICES=0 /home/lizhengran/software/isaac-sim-4.5.0/python.sh run_g1_physx.py --scene absent_new.usda --robot-scene robot_new.usda --source-export source_export_new.json --protocol protocol.json --source-root . --output-dir validation_new --gpu 0
```

原生启动会先初始化CUDA驱动，使用单GPU配置；不依赖G2扩展。新结果复制回子工程 `results/g1/validation_new/` 后：

```sh
.venv/bin/python Robot-Perception-Sim/scripts/compare_g1_physics.py --target-dir Robot-Perception-Sim/results/g1/validation_new
```

## 未通过项与下一步

完整接触等价仍未验证：材料优先级映射有明确差异，反弹次数不一致，撞框后最大轨迹偏差接近1m。没有通过调整源MuJoCo参数来缩小这些差异，也没有把相同终局当作完整等价。若下一阶段需要定量挡球，应先明确允许的接触误差，再解决完整场景材料规则及球滚动/扭转模型。

G1+G2集成建议仅作为后续计划：在新的组合场景中挂载已验证ZED资产，复用G2发送/接收与像素身份验证，建立DOF按名映射和世界/头部/相机变换检查。保留本次独立球基准和G2平面fixture作回归；不得直接修改已冻结G2环境。本轮没有执行该集成，也不开始G3。
