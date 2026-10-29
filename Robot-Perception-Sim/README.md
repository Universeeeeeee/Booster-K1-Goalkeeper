# K1 场景双目相机扩展

在现有 `Booster-K1-Goalkeeper` 仿真环境中增加 ZED X 相机资产和 MuJoCo 双目 RGB 采集接口。**能参考、复用的，一律参考、复用；尽量不要增加额外负担。** 新增开发仅在 `Robot-Perception-Sim/`，唯一提交仓库为 `Universeeeeeee/Booster-K1-Goalkeeper`。

直接复用根目录 `.venv`、K1 资产、`k1_goalkeeper.scene.build_scene`、`config/simulation.json`、球场、球门、足球及已有头部双目坐标系。没有新增 Python 依赖、机器人模型副本或独立安装流程。

在仓库根目录执行：

```bash
.venv/bin/python Robot-Perception-Sim/scripts/preview_stereo.py
.venv/bin/python Robot-Perception-Sim/scripts/preview_stereo.py --head-yaw 0.4 --head-pitch 0.3 --output-dir Robot-Perception-Sim/results/head_turned
.venv/bin/python -m pytest -q -c Robot-Perception-Sim/pytest.ini Robot-Perception-Sim/tests
```

输出位于 `Robot-Perception-Sim/results/stereo/`：`scene.xml`、`left.png`、`right.png`、`overview.png`、`mount.png` 和 `metadata.json`。支持 `--camera-config` 和原格式的 `--scene-config`，完整参数见 `--help`。生成场景仅写入本子目录。

仅在全新安装、根目录环境或 K1 资产缺失时，按父项目 README 安装原有环境。当前已有环境无需安装或下载机器人资产。

官方 ZED X 外观网格原样复用 [zed-ros2-description](https://github.com/stereolabs/zed-ros2-description/tree/22468247c1653d08a1e213809f3e9ce76291487d)，通过 [models/zedx_stereo.xml](models/zedx_stereo.xml) 添加无质量、无碰撞的相机组件。参数见 [config/cameras/zedx_wide.json](config/cameras/zedx_wide.json)。官方 SDK 的[系统要求](https://www.stereolabs.com/docs/development/zed-sdk/specifications)列出 Windows、Ubuntu、Jetson Linux，当前 macOS Apple Silicon 使用 MuJoCo 接口。未来支持的平台优先评估官方 SDK；当前接口不提供 SDK 深度、跟踪或 IMU 功能。

采集接口直接接收已有 `MjData`，默认以 **10 ms 全局曝光**、10 个时间子样本进行左右同步累积，不推进调用者的物理时间。曝光期间以当前 `qvel` 不变近似运动，通过 MuJoCo 的 `mj_integratePos` 更新状态副本；静止场景画面不变，有速度的物体或头部会产生运动模糊。`timestamp_s` 是曝光开始时刻，帧中还返回 `exposure_start_s`、`exposure_end_s` 和 `exposure_samples`：

```python
from robot_perception_sim.stereo import StereoCamera

# model 是加载相机扩展后的场景，data 来自现有仿真循环。
with StereoCamera(model, camera_config) as camera:
    frame = camera.capture(data)
    left_rgb, right_rgb = frame["left"], frame["right"]
    timestamp_s = frame["timestamp_s"]
```

命令入口会设置本目录和父项目的导入路径。若自行编写入口，需将这两个目录加入 Python 模块搜索路径。保持同一线程更新物理和采集图像；本轮预览只使用 `mj_forward`，不验证机器人动态站立。

首版为理想针孔模型：960×600、120 mm 基线、水平 FOV 110°、方形像素，推导垂直 FOV 约 83.50°。挂载偏移、外观网格放置和零畸变均为仿真假设，不是实机标定。10 ms 超过 120 fps 的 8.33 ms 帧周期，因此参考模式改为 **60 fps**；这是P1静态预览的参考模式；后续录制及G3的调度、实测速率见下文。

曝光参数为 `exposure_time_ms: 10.0`、`exposure_samples: 10`；设为 0 ms 可恢复瞬时成像。当前为等权 RGB 时间平均的运动模糊近似，不模拟曝光期间的加速度、接触响应、自动曝光或真实传感器亮度/噪声。默认静态预览速度为零，因此即使曝光 10 ms 也不会出现拖影。运动足球的 0/10 ms 对照图由测试生成在 `results/tests/exposure/`。

验证结果与坐标定义见 [docs/mac_validation.md](docs/mac_validation.md)，来源见 [docs/sources.md](docs/sources.md)，后续路线见 [docs/plan.md](docs/plan.md)。

## 运动足球双目视频

2026-10-21 增加 P2 物理射门的最小采集闭环。激活已有 `robot` 环境后，在仓库根目录执行：

```bash
python Robot-Perception-Sim/scripts/record_stereo.py --preset right --duration 1 --output-dir Robot-Perception-Sim/results/moving_stereo
python Robot-Perception-Sim/scripts/record_stereo.py --preset post --duration 1 --output-dir Robot-Perception-Sim/results/moving_stereo_post
```

默认 60 fps，先复用原有控制器站立 2 秒，再执行已有射门预设。只模拟一次物理轨迹，同一状态分别生成 0 ms 和配置曝光（默认 10 ms）的左右图像。支持已有射门预设、`--duration`、`--seed`、`--fps`、`--camera-config`。

视频编码使用现有 FFmpeg，从 PATH 或 Python 同目录查找。当前 `robot` 环境已包含 FFmpeg；使用根目录 `.venv` 时，可通过 `--ffmpeg` 指向现有可执行文件，无需安装新 Python 依赖。输出目录必须为空，已有记录不会被覆盖。

每次运行保存四个 MP4（`left_0ms.mp4`、`right_0ms.mp4`、`left_exposed.mp4`、`right_exposed.mp4`），以及原始 PNG、`frames.jsonl`、`ground_truth.npz`、`metadata.json`、`simulation.json` 和 `diagnostics.json`。PNG 为算法输入，H.264 视频用于浏览。帧时间与实际物理步时间的偏差单独记录，视频按标称帧率播放。

完整说明与本机结果见 [docs/moving_stereo_validation.md](docs/moving_stereo_validation.md)。仍采用匀速曝光近似，接触期间的误差单独报告；没有实现规则轨迹、CSV 回放或足球检测算法。

## Isaac / ZED SDK 独立 PoC

G2 已在目标 Linux 主机完成两轮各100帧真实 SDK 验收，包括正常/异常退出、同端口重启、实际两目图像身份与渲染参考时间关联。固定环境、运行命令及适用边界见 [G2 验收报告](docs/zed_sdk_stream.md)。当前为单相机2m纹理平面、瞬时成像；不代表完整 K1 迁移、真实曝光或30fps实时性能已完成。

## Isaac / PhysX G1 迁移

已生成自包含K1/球场/球门/足球USD，完成五类无机器人PhysX各两次对照及自由基座22DoF运行检查。终局一致，但撞框后轨迹和完整场景材料规则仍有差异，不能宣称完整物理等价。资产、实际数据、时间步修复和复现命令见 [G1报告](docs/physics_parity.md)。G1提交时G2保持冻结、尚未开发G3；后续G3进展见下文。

## G3 时间链与曝光

G3已完成本轮固定版本仿真验收：SDK5.1.0下三种零曝光场景各100帧，以及三种10ms有限曝光场景各12帧，336帧均通过实际两目图像、时间、标定和来源验证。曝光采用真实渲染子采样的编码RGBA平均，再经官方SDK读取；属于离线仿真近似，不代表原生ISP或60/120fps实时性能。另完成63组MuJoCo曝光对照及K1录制回放验证。见[时间与曝光契约](docs/timing_contract.md)和[验收摘要](docs/g3_acceptance.json)。后续用户已授权继续G4/G5。

## G4 足球轨迹与事件

30个固定案例、60次正式PhysX运行，同例两次记录一致。覆盖直接进球、1/2/3次地面反弹、左右柱/横梁、偏出、角速度以及静态K1接触后进球/未进球。接触事件与Goal/Miss/Timeout分开，保留原始前后速度和实际对象；不把碰球或Timeout当作成功扑救。提供物理、程序运动学和CSV三种来源接口。见[轨迹与事件库](docs/shot_taxonomy.md)。

## G5 静态图像感知

在独立简单场景中采集真实ZED SDK左右图像、深度和置信度，评测5个距离×左/中/右视角及出视野/遮挡负例。图像HSV检测、双目三角化和SDK球面拟合分别输出有效性与失败原因；GT仅用于离线误差和明确标记的oracle世界坐标变换。推荐NEURAL模式的最终185帧中，173个可见球帧全部检出、三角化球心MAE约2.45cm；SDK球面拟合164/173有效，有效样本MAE2.72cm、P95 8.10cm，4m右侧9帧明确invalid。12个负例没有误检。PERFORMANCE失败与ULTRA/纹理对照保留在[感知基线](docs/perception_baselines.md)，复现见[采集证据](docs/g5_static_capture.md)，汇总见[G5验收](docs/g5_acceptance.json)。参考YOLO已审计，因参考机离线按用户要求暂缓实测，见[来源审计](docs/g5_yolo_source_audit.md)。

G4物理场景和G5简单图像实验分别验收；完整固定K1场景的相机集成见下文G6-A。

## G6-A 完整固定 K1 场景中的头部相机

已将官方ZED X挂载在完整G1 K1头部，在两种头部姿态、两条冻结G4轨迹和0/10ms曝光下完成八组SDK5.1 NEURAL采集。368帧SDK数据包含128个目标帧、384个目标曝光子帧，逐目编号、时间戳、源状态、曝光平均和上传内容均完成独立核验。回放使用最近1ms冻结样本，未重新执行碰撞；机器人为固定视觉回放对象。

原球外观保持白色。白色阈值和灰度边缘圆两种简单检测器分别在56个有可见支持的目标帧中得到2和4个有效SDK球心；边缘圆有效样本世界误差MAE4.97cm、P95 5.54cm（仅4个样本）。直接进门及10ms条件保留检测失败，未填补GT或放宽检测/三维有效性阈值。因此相机与数据链路通过，连续足球感知能力尚未通过。

G6视觉层修正了源文件中零透明度碰撞几何的可见性；历史G1–G5数据未改。一次反弹实际发生前，球已离开两种头部姿态的视野，不能称为相机观测到了反弹。详见[G6-A验收](docs/g6a_acceptance.json)、[相机与采集复现](docs/g6a_acquisition.md)、[逐条件感知和曝光报告](docs/g6a_continuous_perception.md)。本轮不开展walk迁移、训练、实时优化或G6-B/G7。
