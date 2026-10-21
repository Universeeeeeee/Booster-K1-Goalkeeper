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

首版为理想针孔模型：960×600、120 mm 基线、水平 FOV 110°、方形像素，推导垂直 FOV 约 83.50°。挂载偏移、外观网格放置和零畸变均为仿真假设，不是实机标定。10 ms 超过 120 fps 的 8.33 ms 帧周期，因此参考模式改为 **60 fps**；采样帧率调度尚未实现，不代表实际渲染速度。

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
