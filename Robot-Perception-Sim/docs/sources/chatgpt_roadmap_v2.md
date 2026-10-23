# Booster K1 / Robot-Perception-Sim 中长期执行路线图 v2.0

- 编制日期：2026-10-24
- 目标仓库：`Universeeeeeee/Booster-K1-Goalkeeper`
- 工作分支：`feat/perception-p0-p1`（Codex 开工时重新确认远端 SHA；勿盲目覆盖）
- 唯一业务开发目录：`Robot-Perception-Sim/`
- 目标硬件：Linux 工作站，2 × RTX 4090；**尚未在该工作站实测**
- 目标：建立可更换相机、可评测真实 SDK、可重复进行机器人本体感知实验的 Isaac Sim 平台；不研究新的运控算法
- 本文件为可执行**规划和验收约定**，不是完成报告；里程碑周期是排期参考，不代表已经完成或承诺时长。

## A. 远端现状、对齐原则及开始条件

**通过 GitHub 连接器于 2026-10-24 核实：**`feat/perception-p0-p1` 可见 HEAD 为 `fae88488803a04f930a1b5e68712534f712d0c60`（2026-10-20）。相对 `main@23c248f` 领先一条提交，只能确认相机 P0–P1 相关文件：`camera.py`、`stereo.py`、`scene.py`、`zedx_wide.json`、`preview_stereo.py`、`mac_validation.md`、`docs/plan.md`。用户所述“曝光模拟”“运动双目采集”与新增计划文档的提交**未在当前远端可见结果中得到确认**，不能擅自当作远端已验证成果。本地可能已经完成，下一步先核实本地状态、提交及文件，再沿用。

上一版计划：`Booster_K1_Isaac_ZED_感知迁移_Codex实施计划.md`（2026-10-23）。当前 v2 是其 **G0–G8 的中长期延展与精化**；不废弃已存在的 MuJoCo P0–P1 与早期曝光/运动采集实现。

### A1. 核心约束

1. **只开发感知仿真 Infra。** 允许加载官方 K1 `k1_walk`、轨迹回放和必要的接入胶水；禁止本轮新训练 PPO/SAC 或重构父仓库运控系统。
2. 保持单仓库、业务变更限制在 `Robot-Perception-Sim/`。复用父仓库的 K1 模型、场景、射门事件、walk 权重和源代码；不另建独立远端。
3. MuJoCo 是参考与回归基线；Isaac Sim 是目标平台。不要求跨物理引擎逐帧完全一致，但要求参数有据、事件一致性可检查、差异可量化。
4. **SDK 深度（从双目图计算）与仿真渲染真值深度绝不混淆。** 正式感知不得读 GT 球位/GT 轨迹/SDK 外的隐藏标签；Oracle 只进入诊断和离线报告。
5. 保留相机、本体、世界三组坐标和各自时间戳；世界坐标适合轨迹分析，机器人坐标适合可部署接口。
6. 先完成单卡单相机端到端，再谈双卡加速、多场景、高保真和数据扩容。不要把两张 24GB 当成一个 48GB 显存池。
7. 每阶段独立最小代码变更、测试和提交。没到验收 Gate 不进入依赖它的下一阶段。

### A2. 远端核查命令（请在实际部署机器/本地仓库执行）

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git ls-remote --heads origin feat/perception-p0-p1
git log --oneline --decorate -12
git fetch origin feat/perception-p0-p1
git log --oneline --left-right HEAD...origin/feat/perception-p0-p1
find Robot-Perception-Sim -maxdepth 4 -type f | sort
```

列出预期两条曝光/运动采集的 Commit SHA、变更路径和测试结果。若远端缺失，先审查是否推错分支、推送失败或本地未 commit；不自动 `reset --hard`、不 `git push --force`。已有未提交文件先以安全方式备份和登记。之后再冻结阶段基线 SHA。

## B. 关键版本、资料与硬件事实

### B1. 版本配对（以实际安装与官方 release 为准）

| 使用情景 | Isaac Sim | `stereolabs/zed-isaac-sim` | 用途 | 决策 |
|---|---|---|---|---|
| 旧版快速 PoC | 已部署的 4.5（如确实存在） | 对应 3.x | 检查可否让虚拟 ZED 被 SDK 正式打开 | 不破坏旧环境；成功后仍评估新平台 |
| 长期主线 | 6.1.0（官方预编译下载页面所列稳定版，2026-09） | 5.3+ 的兼容版本 | ZED Sim2Real、SDK/IMU、Isaac Lab 张量可选 | **建议**独立安装、锁定版本，先做兼容性冒烟 |
| 其他 | 6.0 | 对应 5.x | 遇到 6.1 的兼容性阻断时备选 | 不因数字更大盲目升级 |

新的官方扩展文档要求接收端 **ZED SDK 5.5.0 或以上**。SDK 安装于接收端（可在同机或另一台机器）；官方扩展经 `./build.sh` 构建，再将仓库 `exts/` 路径加入 Isaac Sim 扩展管理器。要检查构建日志：如果 Sim2Real 库未下载成功，编译可以成功但该功能不可用。官方扩展提供 SDK IPC/RTP stream、SDK-free renderer depth、Isaac Lab tensor 三条路线；三者不可互换其误差含义。

官方来源：
- https://docs.stereolabs.com/docs/integrations/isaac-sim
- https://docs.stereolabs.com/docs/integrations/isaac-sim/get-started-with-isaac-sim
- https://docs.stereolabs.com/docs/integrations/isaac-sim/setting-up-the-zed-in-isaac-sim
- https://github.com/stereolabs/zed-isaac-sim
- https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/download.html
- https://docs.isaacsim.omniverse.nvidia.com/6.1.0/reference_material/sim_performance_optimization_handbook.html

### B2. ZED X Wide 首版规格与真实性等级

- Stereolabs ZED X Wide（2.2mm），标称 **global shutter**、**120mm 双目基线**、视场角 **110° × 80°**。
- 960×600 支持 15/30/60/120 FPS；1920×1200 支持 15/30/60 FPS（具体组合按官方规格）；六轴 IMU 标称 200Hz。
- 现有 MuJoCo P1 已用 960×600、水平 110°、理想 rectified 平行双目，推导 `fx=fy≈336.10px`，不是每台实机工厂内参；已有安装偏移 `position_parent=[0.0351,0,0.04]m` 是虚拟布局假设。
- 标注参数来源：`vendor_spec`、`nominal_model`、`simulation_assumption`、`runtime_sdk`、`device_calibration`。除非采集实机标定文件，否则不得标注 `device_calibration`。
- 曝光需要保证 `0 < exposure_s <= 1/fps`（留出读出和实际约束余量）；**120FPS 帧周期约 8.33ms，不能把 10ms 曝光设为同一模式的物理有效配置**。
- 因 ZED X 是 global shutter，不应在本模型中无证据引入 rolling-shutter 行畸变。
- ZED Sim2Real 提供自动曝光、白平衡、噪声、运动模糊、暗角和锐化，但**其准确性仍需实验验证**，不可直接声称等效于逐台 ZED X 真实 ISP。

官方来源：https://stereolabs.static.ferndocs.com/docs/products/cameras/zedx/specifications

## C. 目标技术架构：三个隔离通道，一个可替换的相机接口

```text
Trajectory Catalog ────> Physics / Kinematic Replay ─────> USD scene
   ├─ PhysX shots                                      K1 + goal + ball
   ├─ procedural trajectories                                │
   └─ measured CSV replay                             head-mounted ZED X
                                                            │
                                                      Stereolabs extension
                                              ┌─────────────┼─────────────┐
                                              │             │             │
                                     RGB-L / RGB-R      Renderer GT     IMU model
                                              │            depth          │
                         SDK IPC/RTP ────────┤             │             │
                                              ▼             │             │
                                         ZED SDK client     │             │
                                              │             │             │
                                    RGB / SDK stereo depth  │             │
                                              │             │             │
            ┌─────────────────────────────────▼─────────────┘             │
            │                                                             │
         perception processor: detector → 3D ball center → tracking ← time-sync/pose
            │
       world / base / camera frame estimates → trajectory prediction
            │
            ▼
        evaluator (offline only): compare with independent GT / events / latency
```

**只设三个边界，不建立泛化框架：**

- **Simulator/recording boundary**：`FrameRecord`（左右图、时间、传感器/静态标定、运行姿态来源）+ `GroundTruthRecord`（单独文件，不进入感知函数）。Isaac、MuJoCo 和真实硬件共享文件规范。
- **Sensor provider boundary**：`zed_sdk_stream`、`sim_rgb_only`、未来 `custom_camera_sdk`；返回相同最小帧语义，但保留 provider 专有诊断字段。严禁自己伪装成 ZED SDK。
- **Estimator boundary**：`BallObservation`（2D/3D/有效性/不确定度/坐标系）→ `BallTrack`（世界与本体坐标下速度、预测、置信度、失效原因）。

推荐目录（均为建议，开工时优先复用本地已经存在的文件和命名；不应为了目录形状强行重构）：

```text
Robot-Perception-Sim/
├── config/{cameras,scenarios,benchmarks}/
├── robot_perception_sim/         # 既有 camera.py/stereo.py 等继续存在
├── isaac/                        # USD import, scene, ZED mounting / streaming
├── trajectory/                   # shot catalog, physics inputs, replay, events
├── acquisition/                  # timestamps, recording, sensor provider adapters
├── perception/                   # detection, depth, geometric center, tracking
├── evaluation/                   # offline metrics, ablations, reports
├── scripts/                      # each milestone command line entrypoints
├── tests/                        # pure math + simulation integration + SDK end-to-end
├── docs/                         # compatibility, calibration, physics, API, results
└── results/                      # generated data, gitignored; reports keep lightweight indexes
```

## D. 分阶段技术路线（每个阶段切成不超过 1–3 个 Codex 可审查 PR/commit）

下面时间为单人或小团队的**排期量级，不是承诺交付时间**；跨阶段未完成的工作不能自动标为通过。

| 阶段 | 主题 | 依赖 | 可并行 | 预计量级 | 里程碑成果 |
|---|---|---|---|---|---|
| G0 | 核对推送和冻结基线 | 无 | — | 数天 | 版本、资产、现有两条提交可复验 |
| G1 | Isaac Sim K1/球门/足球迁移 | G0 | G2 PoC 可先做 | 1–2 周 | 可运行 USD 场景和跨引擎事件对照 |
| G2 | 官方 ZED X SDK 流 | G0、G1 的最小 USD | G1 球物理调参 | 1–2 周 | `sl.Camera` 真实打开虚拟双目/SDK 深度 |
| G3 | 曝光与运动双目采集完善 | G2（SDK）；本地代码可提前验证 | G4 | 1–2 周 | 全时间戳、曝光/左右同步与录制 |
| G4 | 足球多情景轨迹/事件 | G1 | G3 | 1–2 周 | 可复现轨迹目录和全面标签 |
| G5 | 静态视觉检测及深度 | G2–G4 | G6 | 1–2 周 | 图像→足球三维位置→世界坐标、离线误差 |
| G6 | K1 横移基线迁移/回放 | G1 | G5 | 1–2 周 | 静止、左移、右移和头部转动数据 |
| G7 | 时序与运动补偿/预测 | G3、G5、G6 | G8 | 2–3 周 | 完整动态补偿对照评估与误差传播 |
| G8 | 相机光学/成像真实性 | G3、G7 | G9 | 2–3 周 | 曝光/模糊/增益/噪声/畸变消融 |
| G9 | 多环境泛化与遮挡 | G4、G7 | G10 | 2–4 周 | 细分困难场景与失败归因 |
| G10 | 双 4090 性能与数据规模 | G5–G9 | G11 | 1–2 周 | 单卡/双卡吞吐、延迟、VRAM、数据工具 |
| G11 | 替换自定义相机与文档 | G5、G7 | G10 | 2–3 周 | 参数驱动适配、独立校验、完整部署文档 |
| G12 | 实机/半实物 Sim2Real | G11 + 实物或记录 | — | 后续季度 | 真机 SDK/时间/标定输入并排评测 |

### G0：确认最新提交、两卡环境与已有实验是否真实完成

工作：
- 收集远端与本地 SHA；找到“曝光模拟”“运动双目采集”具体新增文件、CLIs、测试输出。**只补缺失内容，不再造已有能力。**
- 检查 `nvidia-smi -L`、驱动、CUDA runtime、Ubuntu、Isaac 4.5/6.1 的安装目录、SDK、extension、PyTorch/Python 兼容、Vulkan/RTX、SSH 登录方式。记录 `CUDA_VISIBLE_DEVICES`、GPU ordinal 与实际 PCI bus id 映射，避免编排写错卡。
- 短烟测：原 MuJoCo P0–P1、曝光模拟静止与匀速运动、运动双目时间同步和左右目配对、官方 walk 已存在的测试。若原代码不支持 Linux OpenGL，记录失败边界，不擅自删测试。

产物：`docs/g0_environment_baseline.md`、`results/g0/manifest.json`、对两条提交的 **已实现/缺失/跳过/失败** 矩阵。

**Gate**：所有先前已完成能力可以用明确命令复验或明确阻断条件；安全保持父目录历史不变。

### G1：MuJoCo → Isaac Sim / PhysX 的“语义等价”迁移

工作：
- 读取 MuJoCo `scene.py`、`simulation.py`、`detection.py`、`config/simulation.json` 和官方固定 SHA K1 22DoF 资产。首先冻结机器人几何、关节名称、方向、home q、世界坐标、碰撞组以及球门尺寸。
- 尽量使用原官方 URDF 导入 USD，若有 MJCF 转换路径亦须审计失配而不是无条件切换；对缺少的球、门和场地独立 USD author。K1 自由基座，不能在“站立”演示中偷改成固定机器人。
- 区分动态碰撞模型与仅作视觉资产的壳体，禁止把无质量的 ZED 外观变成实际影响 K1 平衡的附加负载；未来真实相机质量另建模型试验。
- 运行无机器人自由球、自由落体、地面弹跳、直球入门、门框碰撞、无意外初始穿透等最小用例；同一初态分别在 MuJoCo 和 PhysX 记录 `ball_world_xyz`, `v_world`, contacts, goal outcome`。
- MuJoCo `solref/solimp` 不可直接视作 PhysX restitution; 以首弹高度比、能量损失、接触时长、事件触发时刻作为可测配对指标。

产物：`isaac/scenes/goalkeeper_base.usd`、`isaac/asset_manifest.json`、`docs/physics_parity.md`、五类最小仿真录像/数据。

**Gate**：22 DoF 映射、质量与惯量、尺寸、净门开口、坐标方向可逐项核对；典型球轨迹合理且无无效状态；差异记录可复现。不可用“图看起来一样”代替几何检查。

### G2：真正的 ZED SDK 软件在环（核心里程碑）

工作：
- 使用官方 `stereolabs/zed-isaac-sim` 的可复现 release tag、ZED X Wide USD（不重复建模），通过 USD 父子层级挂到正确 head link；记录厂商标称外参与虚拟安装外参。
- 最小场景单 ZED X：Isaac SDK 扩展 → 左右图 IPC（同机器）或 RTP（跨机器）→ 官方 ZED SDK `sl.Camera` streaming client → `grab()`/`retrieve_image()`/`retrieve_measure()`，存真实 API 返回的 RGB、SDK stereo-matched depth、SDK intrinsics/timestamp/运行配置。
- 两种深度模式单独测试并标记：`zed_stereo_sdk` = 两目 RGB 经 SDK 匹配；`renderer_gt_via_sdk` = `Stream Depth` / `ingestCustomDepth` 的 Isaac 真值，不可混写或在同一误差数字里平均。
- 分辨率与 FPS 从 960×600/30 开始，验证 60，最后在吞吐允许时验证 120；不能将“设置为 120 FPS”当作“稳定输出 120 FPS”。
- 首次下载 SDK 模拟设备标定可能需要网络；记录 serial/config 关联。报告 Sim2Real 库是否真正载入，避免构建成功但功能缺失。

产物：`scripts/verify_zed_sdk_stream.py`、`docs/zed_sdk_stream.md`、`results/g2/{left,right,depth,metadata,log}`。

**Gate**：能真实打开虚拟 ZED 设备；100 连续帧中左右配对与有效深度可核验；有完整 SDK 日志、frame id、stream timestamp；明确记录 SDK 是如何计算深度的。阈值为提案，因环境限制需先测基线再冻结。

### G3：继承已做的曝光模拟 + 运动双目采集；补全端到端时间链

工作：
- **先审计两条本地新提交**。若其已实现曝光、subframe、相机运动、双目同步，直接提取测试、配置及缺项，不另外建立第二套相机采集器。
- 建立四个独立时钟/事件：`t_sim_state`、`t_exposure_start/end/mid`、`t_sensor/sdk_timestamp`、`t_receive/processing_monotonic`。要记录时钟域、时间偏移、与仿真物理/渲染时间之间映射。
- Isaac 物理 1 ms 只是对照起点，不必永远锁死；RGB 60/120Hz 与 IMU 200Hz 独立调度。处理 1ms 物理步无法整数整除 60/120 的情况，以预定采样时刻的状态插值或明确调度误差记录。
- 左右摄像头物理上 global shutter 同步，必须使用同一曝光时间窗；像素渲染可能依次执行，但不能每渲一次就推进不同的物理状态。记录左右世界位姿、帧丢失、配对失败与 rolling skew（理想情况应无）。
- 对曝光模型分别核实：现有实现、Stereolabs Sim2Real、Isaac Replicator RTX real-time 和 path-tracing subframes。避免默认将后处理拖影视作真实曝光积分、避免重复叠加两套模糊。
- 起步曝光测试建议 1 / 3 / 5 ms 在 120FPS 下；1 / 3 / 5 / 10 ms 在 60FPS 下。帧长严格大于曝光长度；参数标注仿真假设或运行参数。

产物：`acquisition/` 最小记录器、`docs/timing_contract.md`、静态和运动双目连续采集报告。

**Gate**：一份录制中所有有效左右帧有对应相同曝光时刻和标定版本；非整比采样无累计时间漂移；能按时间重放姿态与球世界状态；所有数据源有准确 provenance。

### G4：从“10类预设射门”发展到全面轨迹/事件测试库

**原则：轨迹生成、物理事件检测、最终结局分别记录。** 必须复用原球碰撞与整球越门线逻辑，不能把“touch goal”当作“goal”，不能把“碰到机器人”当作“成功扑救”。

推荐分类矩阵：

| 组别 | 子类 | 独立事件/结果 |
|---|---|---|
| 直接进门 | 低/中/高、左/中/右、斜射 | 直接 Goal；无地面反弹 |
| 地面反弹后进门 | 1 次/2 次弹跳、左右偏转、低角度 | `ground_contact_count >= 1`, Goal |
| 立柱/横梁 | 左柱、右柱、横梁；接触后入门/弹出 | `frame_contact`, 后续 Goal/Miss |
| 没进门 | 左右射偏、射高、射门力度不足、滚出边线 | Miss/Timeout，明确运动原因 |
| 复杂旋转 | 上旋、下旋、侧旋、不同速度和入射角 | contact count、spin、轨迹偏折 |
| 交互与遮挡 | 球经过身体附近、机器人挡住相机、球碰身后入门/未入门 | 可见性、接触、实际终局独立 |
| 退化/异常 | 球初始在 FOV 外、途中出 FOV、球被遮挡、帧缺失 | 感知观测是否有效 |

执行顺序：先固定 20–30 个确定性 golden cases，再扩展随机配置；以验收的 **实际物理事件** 自动归类，而不是凭初始参数猜结果。抽样层覆盖速度、初始距离、高度、角速度、入射角、球门位置。保存固定随机种子及失败归档。

至少支持三类来源：`physx_physical_shot`（动态物理）、`procedural_kinematic`（画面与感知诊断）、`measured_csv_replay`（真实轨迹回放）。后三类不能宣称其球撞框后响应真实，除非由物理引擎接管。

产物：`config/scenarios/*.json`、`trajectory/shot_catalog.py`、`trajectory/events.py`（仅在原模块不适用时新建，优先 adapter）、`docs/shot_taxonomy.md`、数据导出脚本。

**Gate**：每类 golden case 可重复；记录事件序列（时间、对象、碰撞处位置、球前后速度）、**终局**（Goal/Miss/Robot Contact/Timeout 分开），所有标签来源可核验；无不可能的初始穿透和穿框。

### G5：静止机器人下的第一条真实图像感知基线

工作：
- 先把球设计为高对比、背景简单，使用 OpenCV HSV/轮廓+质心作为可解释基线；同步评测从远端 PC 只读复用的 YOLO 权重和后处理（必须审计模型来源与许可证），输出检测器的置信度/Mask/BBox。
- ZED SDK 深度和手工 rectified triangulation 作为两种视觉深度估计，不是同一个“深度真值”。深度最初可取 bbox 中有效球前景深度分布；由于球面不是球心，采用半径+视线方向几何校正或轮廓球拟合，不能直接把球表面深度当球心距离。
- 远球应有误差敏感度分析：`σ_Z ≈ Z²/(fB) σ_d`；当前 `fx≈336 px`、`B=0.12m`，Z=6m、视差误差 0.25 px，理论 `σ_Z≈0.223m`。远距离盲目宣称厘米级精度是不成立的。
- 保存 2D 检测、SDK 深度有效率、3D 球心、`T_world_camera`（oracle 姿态实验）、跟踪 mask 和失败原因；已知真值只让 offline evaluator 访问。
- 初期位置序列估计速度可先用差分及恒速卡尔曼滤波，之后加入恒重力/弹跳混合模型；不需要深度学习轨迹网络。

产物：可命令行运行的 `run_static_perception.py`，`docs/perception_baselines.md`，静止实验 PDF/HTML 可后做，本期 JSON/CSV/PNG 足够。

**Gate**：5 个深度点 × 左/中/右视角的检测、深度误差、球心误差分项明确；有效 / invalid 返回规范；出 FOV 和被遮挡不会用 GT 伪补。

### G6：把官方 K1 walk 作为感知测试运动源

**两条并行路线，防止运动策略移植阻断感知研究：**

1. **运动学回放（优先获得感知结论）**：从已有 MuJoCo 官方 `k1_walk` 执行记录导出机器人 root 位姿 + 22 DoF 关节时间序列；Isaac 将机器人按时间驱动显示与相机跟随，球可单独回放或物理推进。仅标注“kinematic replay”，不做动态稳定性/碰撞有效性结论。
2. **PhysX 真实动态 `k1_walk` 复用（第二步）**：保留官方权重、69维观测、10帧历史、50Hz 决策、PD 和力矩边界，逐项重新核验 IMU、关节名/符号、基座方向、动作延时、驱动器 torque vs position、控制步长。因为 PhysX 与 MuJoCo 接触、地面摩擦和积分不同，MuJoCo 横移通过≠PhysX 横移通过。

固定指令测试：`body vy={0,±0.15,±0.20,±0.25}m/s`，每个执行静止→侧移→停下→反向→停下。保持固定球轨迹、seed、相机配置，对比左右平移时成像、检测、深度和位姿变换。录像里必须保存实际基座/头部角速度、姿态、线速度，不只保存 command。

产物：`scripts/collect_mobile_stereo.py`、`docs/mobile_motion_source.md`、`results/g6/`。

**Gate**：至少在 kinematic replay 中获得可重复的左右相机运动数据；PhysX walk 如不稳定必须单独报告，不能把运动学数据标成“官方 policy 在 Isaac 成功部署”。

### G7：坐标变换、时序对齐、动态补偿与短时预测

统一右手系、列向量：

```text
W world               B robot base/body
H moving head         M rigid mount
L/R optical cameras   C chosen reference optical camera
T_A_from_B transforms coordinates in frame B to frame A
```

变换：`T_WC(t) = T_WB(t) @ T_BH(t) @ T_HM @ T_MC`；`p_W(t) = R_WC(t) p_C(t) + t_WC(t)`。

**时间语义必须写死为数据契约，不是硬编码具体时钟频率。** 光学帧用曝光中点或明示的曝光积分模型；若 SDK 输出的是 frame-end timestamp，先换算再用 pose 插值。连续 IMU、robot q、root pose 应插值到真实 capture time；对旋转使用 slerp，不直接欧拉角线性平均。

速度的两种可验证方式：

- 先把所有逐帧 `p_C(t_i)` 转成 `p_W(t_i)` 后时序滤波 / 差分，得到 `v_W`；这个方法最适合作为第一版。
- 若必须使用瞬时相机系速度导数 `v_C=dp_C/dt`，则：
  `v_W = R_WC v_C + v_camera,W + omega_camera,W × (R_WC p_C)`。
  未补偿相机平移/旋转时不能将相机相对速度直接称为球世界速度。

逐级 AB 实验（同一个球轨迹和 pose 序列）：

A. 精确 2D/深度 + Oracle 位姿（几何上限）。
B. 真实图像检测 + SDK 深度 + Oracle 位姿（视觉算法误差）。
C. 真实图像检测 + SDK 深度 + 带噪 pose / VIO（部署相关误差）。
D. 真实图像检测 + SDK 深度 + **无运动补偿**（负对照）。
E. 时间偏移 -20、-10、0、+10、+20 ms 和位姿噪声消融（量化同步需求）。

预测目标：100/200/300 ms 球世界位置和球门线穿越时刻/区域（只有在未来交叉时才报告），先从恒速、重力抛物线、分段弹跳模型做 baseline。记录速度 MAE、预测 ADE/FDE、漏检率、跟踪寿命、预测时域内有效样本数。

产物：`docs/frames_and_time.md`、`evaluation/mobile_ablation.py`、真实的回归轨迹与误差图。

**Gate**：变换逆/复合单元测试，静态球、纯机器人平移、纯头部 yaw/pitch、机器人+球同时运动的解析检查全部通过；与 GT 补偿对照及预设时间偏差敏感度报告完整。不能仅用“看起来世界轨迹平滑”证明补偿正确。

### G8：光学、曝光与 Sim2Real 的系统性定量研究

**先做模块实验，不把所有退化因素一起打开。**

矩阵：

- 球速度：0 / 4 / 8 / 12 m/s；距相机：2 / 4 / 6 / 10m。
- 机身：静止 / 横移 ±0.2m/s；头部 yaw：0 / ±0.3 / ±0.6rad/s（只作为可复现实验命令，非 K1 官方能力声明）。
- 曝光：60FPS 下 1/3/5/10ms；120FPS 下 1/3/5ms。
- 成像路径：原始 clean render；已有曝光模型；官方 ZED Sim2Real；Replicator RTX realtime；仅在必要时用 path-traced 多 subframe 对照。
- 光照：强光、阴影、室内暗光、逆光；材质：球纹理、高/低对比背景。
- 噪声、gamma、gain、white balance、lens vignetting/畸变：每次只改一个因素；最后做少数联合测试。

曝光理论：点扩散是曝光时间内球和相机连续运动的结果，可近似积分 `I ≈ (1/N) Σ I(t_k)`；离线图像卷积不可代替动态 3D 遮挡变化。固定同一轨迹不应被模糊模型意外移动物理球的位置。

验收量：检测 recall、bbox 中心误差、轮廓面积/模糊长度、SDK disparity/depth invalid 比例、三维误差、处理延迟及丢帧率。为防止虚假 improvement，图像质量和几何重建误差均须报告。

**Gate**：在至少三种速度、三种曝光设置下可稳定重放图像和感知曲线；不同成像路径不会混淆成同一套真实相机标定。

### G9：扩展为感知基准集（不是做全能足球游戏）

阶段先固定 3 类环境：标准草坪、低纹理浅色地面、复杂光照户外草坪。仅在 G8 有证据表明环境差异影响性能时再加道路、泥土等大场景资产；避免初期资产成本超过研究收益。

- Domain Randomization：限定地面材质、球纹理、太阳高度/光源、曝光/增益、背景扰动物、遮挡物；**物理随机化与视觉随机化分开**。
- 控制分布：训练（若未来有）、验证、测试场景种子完全隔离；环境类别、轨迹类别、距离、球速、有效观测时长分层抽样。不得在同一采样过程中反复复用评测种子选择优化阈值。
- 困难场景：球背景同色、远球尺寸少像素、快速转头、球出视野/再次进入、机器人遮挡、镜面反射、阴影和全局自动曝光突变。
- 明确拒识与置信区间：返回 invalid 而不是错误但高置信的球位。

**Gate**：可单独加载标准 benchmark spec 一键生成并复验，在每类轨迹/环境下给出检测、深度、轨迹预测及失效类型的统计表。保存完整 dataset manifest、固定 seeds 和哈希。

### G10：两张 RTX 4090 的性能优化、数据集产出

**三个部署模式对比，先测再选：**

| 模式 | 分工 | 需排查 |
|---|---|---|
| S0 单卡 | GPU0 Isaac + SDK + YOLO | 基准真实性和单卡资源上限 |
| S1 异构双卡 | GPU0 渲染/PhysX，GPU1 运行 SDK/YOLO | GPU 间复制与同步、CPU bridge、IPC/RTP 路径 |
| S2 多 GPU 渲染 | Isaac 将左右或多路 render products 分配 GPU0/1 | 子画面时戳一致性、系统驱动、render product 显存 |

关键点：NVIDIA 文档明确，GPU PhysX 计算只利用一个 GPU；多 GPU 主要帮助渲染相机，特别是多路传感器，不保证单个 ZED X 立刻双倍。平台本身可能有多 GPU 开销；默认先做单卡稳定版。在多摄像头实验尝试 `ViewportManager.optimize_render_products()`（根据当前版本核对 API）。

统一性能指标：实测相机帧率（非配置 FPS）、sim seconds / wall second、SDK depth fps、YOLO fps、链路 p50/p95/p99、丢帧率、帧配对率、GPU 显存峰值、CUDA 核占用、CPU/RAM、磁盘写入速度，以及首次着色器编译与稳定运行分开统计。优先消除额外 viewport/回传/同步点造成的开销，再扩大数据规模。

**Gate**：同场景固定 300+ 有效帧、至少 3 次重复实验，S0/S1/S2 可公平比较。若双卡无收益则保留单卡方案，不为使用两张 GPU 牺牲可复现性。

### G11：自定义相机替换与技术文档产品化

不要写一个巨大通用传感器框架；只保障 **新的相机无需修改检测、三维估计、轨迹与评测核心**：

- 单目：必须显式声明没有 stereo 深度；可用其他深度源/多帧结构恢复，但不得输出伪 stereo。
- 双目：支持左右 `K,D,T_left_from_right`、frame/timestamp/exposure、rectified or raw、可用深度算法来源和 invalid mask。
- 可选 IMU：坐标变换、时间偏移、rate、噪声、bias；有无 IMU 分开报告。
- Provider：`ZedSDKProvider`（仿真虚拟流与未来真机使用同一 SDK consumer）、`SimRGBProvider`（纯仿真帧）、后续 `CustomCameraProvider`，只作为明确需求出现后的轻量实现。
- 配置审计：合法曝光/帧率、畸变模型、光学坐标、标定文件版本、校准日期/来源、IMU↔相机外参、同步约束。

要求至少用一个不依赖 ZED 官方 USD 的新 pinhole/stereo 传感器配置运行同一测试集，证明没有把 120mm、960×600、ZED X 名称等硬编码进核心算法。

建议文档模块：
1. `docs/architecture_and_contracts.md`：组件与输入输出数据契约。
2. `docs/isaac_physics_parity.md`：K1/球/门资产转换及差异。
3. `docs/zed_sdk_integration.md`：SDK stream、IPC、depth 源、版本问题。
4. `docs/camera_calibration_exposure.md`：内外参、曝光、畸变、噪声与来源。
5. `docs/timing_and_frames.md`：多时钟、坐标链、同步。
6. `docs/shot_scenarios.md`：足球轨迹事件与 Ground Truth。
7. `docs/mobile_perception_evaluation.md`：机器人横移 A/B、误差归因。
8. `docs/performance_and_gpu.md`：单卡双卡资源与延迟。
9. `docs/custom_camera_port.md`：新相机最短适配步骤。
10. `docs/reproduction.md`：环境 pin、命令、固定 seeds 和结果目录。

**Gate**：独立使用者只根据 README/文档在干净环境上重建一个 10 秒 Demo，生成左右 RGB、SDK 深度、世界轨迹、预测图及误差报告；从 ZED 配置切换到 generic 双目不改算法核心。

### G12：长远研究与真正半实物（在新硬件/数据到位后）

阶段 I（数据驱动）：标定实际 ZED X、采集实球轨迹与 IMU/曝光/标定、实测运动模糊与深度性能，反向校准仿真；关注长期稳定性和 SDK 算法更新。

阶段 II（半实物/硬件在环）：由真实 ZED 或自研摄像头的 SDK 输出输入同一 estimator；或将外部 MoCap 足球轨迹按实验时间驱动仿真球，机器人与相机仍虚拟。明确定义“真实输入”究竟是原始图像、SDK 图像/深度还是 3D 真值；不能把“3D 真值喂给 detector”称为真实视觉感知。

阶段 III（本体定位与主动感知研究）：在不开发新 locomotion 的前提下，研究头部主动转动策略对 FOV、有效观测时窗、深度精度的影响；引入 VIO/IMU 噪声、曝光与时间同步误差；形成可移植到自定义双目/鱼眼的基准。

## E. 验收指标：硬门槛 vs 研究指标

### E1. 不可放宽的工程正确性 Gate

- 左右帧必须明确使用同一帧采样/曝光时刻，若不同步必须记录且判定无效。
- 轨迹/pose/IMU 时间戳单调，传感器时间与 sim 时间关系可复现。
- `world/base/optical` 变换逆和复合、静止/平移/旋转解析测试正确。
- GT 只给离线评估读取，正式输出中的深度来源不能假冒 SDK stereo。
- SDK 测试必须真实运行官方 API，不能只 import 包。
- 场景生成不得偷偷调整官方 walk 观测、动作或力矩限幅以让验收通过。
- 固定 seed 的场景能重复产生同一事件标签，记录不可复现原因。
- `results/` 大文件不提交；记录小型报告、哈希、命令和版本。

### E2. 建议起始性能观察目标（并非已有结论，G5 后可冻结）

| 指标 | 早期目标建议 | 备注 |
|---|---|---|
| 静态理想几何投影 | <1px | 沿用 MuJoCo P1 的数量级，不等同实际 SDK 检测误差 |
| 视觉检测召回 | ≥95%（2–5m 高对比 clean 子集） | 不外推到自然场景 |
| 帧配对与事件记录 | 100% 可追溯，丢帧须计数 | 不等于 0 丢帧 |
| SDK depth 可用率 | 实测曲线优先，不预设绝对值 | 远距离/边缘/模糊物体可能失效 |
| 轨迹预测 | 分 100/200/300ms 的 ADE/FDE | 与恒速/重力/弹跳基线比较 |
| 移动补偿 | 在 Oracle pose 下，优于无补偿基线 | 以场景配对差异检验 |
| 运行速度 | 先稳定 30 FPS，再测试 60/120 | 基于真正 SDK 输出帧，不基于配置字段 |

### E3. 必须做的系统研究/失效归因

- 相机基线+宽 FOV 对远距离深度的约束：`fB≈40.33px·m`。在 6m 处若视差误差 0.25px，局部线性深度误差约 `0.223m`；在 10m 约 `0.620m`。这是理论灵敏度，不是 SDK 实测。
- 足球真实世界中心 vs 摄像头深度给出的可见球面：这是**模型系统偏差**，不能只靠平滑解决。
- 有限曝光引起的球像素位置积分和局部追踪偏差；接近相机时可能出现视野突然消失。帧率再高也不能替代足够可见时间。
- SDK 虚拟流与真实 ZED 的同步/标定/噪声不同，必须记录有效性范围。
- Isaac 和 MuJoCo 动力学差异不能归因到感知模型；应以同一轨迹回放先隔离图像/估计误差。

## F. 数据目录与最小正式 schema

第一版文件格式无需数据库：

```text
results/experiments/<experiment_id>/
  config.resolved.json      # effective config + hash
  manifest.json             # git sha, asset versions, fps, SDK, GPU ids, paths
  frames.jsonl              # frame_id, t_sim, exposure start/end/mid, sdk/recv timestamp, image paths, flags
  left/*.png
  right/*.png
  sdk_depth/*.npy           # optional, with invalid mask / native units metadata
  imu.jsonl                 # optional
  base_pose.jsonl           # pose source, timestamp, T_WB, cov/quality
  gt/ball_state.npz        # strictly offline, no import into estimator
  gt/events.jsonl          # contact and goal
  predictions.jsonl        # detector, 3D position, velocity, covariance, prediction horizon
  metrics.json
  plots/
```

明确保存 `source_kind` / `depth_source` / `pose_source`，例如：
- `depth_source`: `zed_stereo_sdk`, `renderer_gt_via_sdk`, `renderer_gt_direct`, `triangulation_from_uv`, `missing`。
- `pose_source`: `sim_oracle`, `kinematic_replay`, `estimated_odom`, `zed_vio`（仅功能真的启用时标注）。
- `trajectory_source`: `physx_dynamic`, `mujoco_replay`, `procedural`, `external_mocap`。

在早期只实现实际使用到的字段； schema version 变更需有迁移说明，避免为未来可能性添加空字段。数据文件应在脱离 Isaac、MuJoCo 和 ZED SDK 的环境中用标准 Python/NumPy 独立加载并评测。

## G. Codex 工作流：每次只交付一个有验收点的最小批次

**通用工作规则：**

- 每次先列依赖、当前 SHA、触及的文件、预期失败点和验收命令，再编辑。
- 每一批只处理一个阶段的一个目标；不会因为一个能力已经存在而二次实现。
- 测试分纯 Python / MuJoCo / Isaac headless / SDK真实流；环境缺失时必须明确 SKIP 原因，不得标完整 PASS。
- 报告至少包含 commit、运行环境、命令、测试通过/失败数、实际 screenshot/data、性能、偏差和下一步风险。
- 所有现场执行前检查正在运行的 Isaac/SDK/其他 GPU 作业；不能无授权中断进程、升级驱动、重启服务器或覆盖其他用户结果。
- 改动严格控制在子目录；如必须修改父工程，先给出最小理由和影响，再由用户决定。

### 下一批最建议执行：G0a（核对提交）＋G0b（兼容矩阵）＋G2-sandbox（官方 SDK 独立 PoC）

不要求一次性迁移整个 K1 才能验证官方 SDK；在 G0 后可以**并行**运行一个官方示例空场景 PoC，提前排除 ZED SDK/驱动/扩展阻断。随后完成 G1 USD 迁移，再将已验证的 ZED 摄像头挂到 K1。

**G0a 可复制指令：**

```text
在 Booster-K1-Goalkeeper 的 feat/perception-p0-p1 分支，先检查 GitHub remote HEAD 和本地 HEAD 是否一致，审计最近两条“曝光模拟”“运动双目采集”提交及新增计划文档是否真实存在于远端；逐项检查测试与 CLI。保留所有本地差异，不要 reset、clean、force push，也不要重写已有模块。提交 docs/g0_remote_and_local_audit.md，给出 SHA 与变更文件对照、测试真实结果和缺失项。仅完成审计，停止。
```

**G0b 可复制指令：**

```text
只审计当前双 4090 Linux 工作站的环境与官方 Isaac Sim / Stereolabs ZED extension / ZED SDK 版本可行性。检查系统、GPU 驱动、显存、CUDA 运行环境、Isaac 4.5 与 6.1 是否存在、SDK 版本、ZED 扩展和可运行的官方例子。给出兼容矩阵和两套安装方案，不修改驱动或删除旧版本；如果不能判断，请报告 blocker。写入 docs/g0_host_compatibility.md 并提交。
```

**G2 sandbox 可复制指令：**

```text
在不修改原 K1 场景的前提下，使用当前已确认兼容的 Isaac Sim 与 stereolabs/zed-isaac-sim 固定版本，运行官方虚拟 ZED X Wide SDK 连接最小 PoC。验证实际 SDK API 返回左右图、timestamp、有效深度，明确 stereo-matched SDK 深度 vs renderer GT depth 的差别。保存截图、版本、源代码、SDK 日志、连续 100 帧结果；若需安装新环境，只在独立目录进行，不覆盖既有 4.5；不实现球、YOLO、运控。通过后停止等待检查。
```

## H. 路线图的成功判据（面向半年+）

**第一层（原型跑通）**：K1 USD + ZED X 官方流 + 真正 SDK 深度 + 有标签物理射门 + 静止视觉球心。

**第二层（可做研究）**：左/右横移、相机变化、曝光时戳与 IMU、时序插值、世界轨迹估计，以及“无补偿/理想 pose/估计 pose”的可重复对照结果。

**第三层（可作为 Infra 交付）**：多种场景/相机参数可配置、统一数据格式、自动生成数据集与回归报告、双卡性能经测量后部署、从 ZED X 切换自定义双目的最小接口与指导文档。

**第四层（可发表/Sim-to-Real）**：真实硬件标定、实测轨迹与 ZED 数据反馈仿真、半实物输入、定量确认模型差距，并探索如何通过主动感知延长有效可观测时窗。

完成标准是**有可复现实验、可解释的误差归因和可更换相机能力**，而不是在高质量仿真视频里“看起来检测正确”。
