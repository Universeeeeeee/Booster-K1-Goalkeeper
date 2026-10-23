# G1 MuJoCo 源资产契约（迁移前冻结）

当前仅完成 G1 的源端契约准备，**没有生成 USD、没有运行 Isaac/PhysX、没有通过 G1 Gate**。此工作可与 G2 SDK 最小场景独立推进，不加载 GPU、不运行控制器、不执行物理步。

实际证据：`results/g1/source_contract_20261023T172323_398380Z/asset_manifest.json` 与同目录 `source_scene.xml`。结果目录被 Git 忽略，保留在本地；本文记录复现命令和实测摘要。

```sh
.venv/bin/python Robot-Perception-Sim/scripts/audit_isaac_source.py
.venv/bin/python -m pytest -q Robot-Perception-Sim/tests/test_isaac_source_contract.py
```

默认创建带 UTC 时间戳的全新 `Robot-Perception-Sim/results/g1/source_contract_*/`，复用父工程 `k1_goalkeeper.scene.build_scene(..., robot_mode="free")` 生成引用原资产的场景。指定 `--output-dir` 也必须是本子工程 `results/` 下尚不存在的目录；已存在目录会报错，避免覆盖证据。可通过 `--scene /absolute/path/to/complete_scene.xml` 读取仓库内现有的完整合成场景，并仍将 manifest 写入全新目录。输入应为已有完整 MJCF，不是带 include 的装配入口；该脚本的来源哈希覆盖场景、官方 K1 MJCF/URDF/网格、配置、装配器与静态解析函数。

实际加载 MuJoCo 3.3.7 的编译模型，`mj_forward` 后读回数据。测试 **3 passed**：22/20 映射、从独立再次加载的模型核对每个 body 的质量惯量及球门/球尺寸、具名关节轴/限位和 actuator 目标/力范围。

## 已冻结字段与实测结果

- 22 个具名 hinge，顺序直接读取官方 `booster_deploy/robots/k1.py`；20 个 policy 动作名称直接静态读取官方 walk 配置，未执行 policy 或复制其控制逻辑。
- policy→22关节索引：`[2,6,10,16,3,7,11,17,4,8,12,18,5,9,13,19,14,20,15,21]`。头部 yaw/pitch 不属于这20维动作。不要将20维策略顺序当作22电机顺序。
- 官方资产 checkout 与冻结版本一致：`12a97516f57469078707afb587a5f84c1cb110d1`。manifest 含来源逐文件 SHA-256，独立列出实际 checkout SHA；不只依赖版本字符串。
- K1 自由基座 `world_joint`，世界初态 `[10.4,0,0.55369]` m，四元数约 `[0,0,0,1]`（wxyz），初始速度为0。世界为右手系、Z向上，场地长轴X/宽轴Y。重力和1ms步长从编译模型读取。
- 场景零关节初态与官方 walk home q 分别保存。导出的 `walk_home_q_rad` 与 `walk_effort_limit_nm` 来自既有 `official_config()` 解析 walk override；不会把原始场景零姿态描述成控制器站立初态。
- 29个 body 的父级、质量、主惯量、惯性坐标系位置/四元数、body相对位姿均由编译模型读出。K1 body 质量合计19.666kg；静态球门不能据其 body 质量反推原 URDF 的动态质量。
- 每个关节包含局部/初始世界轴、锚点、范围、qpos/DOF地址、对应电机、gear、控制与力限幅开关/范围，以及官方 walk effort 限制。模型电机范围和控制器 effort 限制分别记录，不能混用。
- 球半径0.11m、质量0.43kg、各主惯量约0.00346866666667kg·m²，初态 `[7,0,0.111]` m；这是原场景的薄壳近似，未在本次校准。
- 两侧球门各为3m×2m净口，由编译后的门柱与横梁几何位置/半径计算；门线分别X=−11m、+11m。没有网，后支撑仅视觉，参考门不是赛事官方资产。
- 每个 geom 的形状/尺寸/局部位姿、碰撞 contype/conaffinity、group、condim、priority、摩擦、solref/solimp，加上显式pair和exclude签名均保存。group是可视分组，不能当作碰撞过滤层。原MuJoCo的body/父子过滤也不能在迁移时丢失。当前初始接触数为0。

## 后续 G1 验收仍需完成

用该契约逐项核对 USD/PhysX 的自由基座、22关节轴方向/limits、20动作映射、质量/惯量和惯性坐标、视觉与碰撞几何、球门净口及碰撞过滤。额外相机外壳应保持无动态负载，真实相机质量另做实验。源端哈希不能证明转换后模型等价。

之后才运行自由球、落体、弹跳、入门、撞框用例，在两个引擎以同一初态记录位置、速度、接触与结局。MuJoCo solref/solimp 不能直接当作 PhysX restitution；首弹高度、能量损失、接触时长与事件时刻都尚未配对验证。
