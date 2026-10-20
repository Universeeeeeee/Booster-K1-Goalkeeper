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

采集接口直接接收已有 `MjData`，不推进物理时间；左右目在调用之间不修改状态的单线程仿真循环中采集，得到相同采样时刻：

```python
from robot_perception_sim.stereo import StereoCamera

# model 是加载相机扩展后的场景，data 来自现有仿真循环。
with StereoCamera(model, camera_config) as camera:
    frame = camera.capture(data)
    left_rgb, right_rgb = frame["left"], frame["right"]
    timestamp_s = frame["timestamp_s"]
```

命令入口会设置本目录和父项目的导入路径。若自行编写入口，需将这两个目录加入 Python 模块搜索路径。保持同一线程更新物理和采集图像；本轮预览只使用 `mj_forward`，不验证机器人动态站立。

首版为理想针孔模型：960×600、120 mm 基线、水平 FOV 110°、方形像素，推导垂直 FOV 约 83.50°。挂载偏移、外观网格放置和零畸变均为仿真假设，不是实机标定。120 fps 只是参考模式，不代表实际渲染速度。

验证结果与坐标定义见 [docs/mac_validation.md](docs/mac_validation.md)，来源见 [docs/sources.md](docs/sources.md)，后续路线见 [docs/plan.md](docs/plan.md)。本轮范围止于 P0–P1。
