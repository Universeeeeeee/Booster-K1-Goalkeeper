# 来源与复用

唯一开发目录为 `Booster-K1-Goalkeeper/Robot-Perception-Sim/`；提交目标为 `Universeeeeeee/Booster-K1-Goalkeeper`。独立远端 `Universeeeeeee/Robot-Perception-Sim` 已弃用。

**原则：能参考、复用的，一律参考、复用；尽量不要增加额外负担。**

| 内容 | 来源及版本 | 复用方式 |
|---|---|---|
| 仿真环境、依赖 | 根目录 `.venv`、`requirements.txt` | 直接使用，无新增依赖或独立环境 |
| 球场、球门、足球、灯光、机器人、场景配置 | 父项目 `k1_goalkeeper/scene.py`、`config/simulation.json`；参考提交 `23c248f0ef2b81a528a22e44f29402098523b4f6` | 直接调用已有 `build_scene`，将输出写到本目录 |
| K1 MJCF 与网格 | 根目录已有 `assets/booster_assets`，官方固定提交 `12a97516f57469078707afb587a5f84c1cb110d1` | 直接引用，不下载或复制 K1；复用 `head_booster_stereo_rgb_link` 安装坐标系 |
| ZED X 外观 | [stereolabs/zed-ros2-description](https://github.com/stereolabs/zed-ros2-description/tree/22468247c1653d08a1e213809f3e9ce76291487d)，提交 `22468247c1653d08a1e213809f3e9ce76291487d` 的 `meshes/zedx.stl` | 原样纳入单个 574,984 字节网格，无 ROS 依赖；SHA-256 见 `models/sources.json` |
| ZED X 外观坐标参考 | 同一提交 `urdf/zed_macro.urdf.xacro` | 参考 ROS 的 X 前、Y 左、Z 上，以及外观高度；转换到光学坐标。碰撞和质量不复用 |
| 名义相机规格 | [官方产品页面](https://www.stereolabs.com/en-us/products/zed-x) | 已核对 120 mm 基线、960×600 / 120 fps 模式；镜头与 110°/80° FOV 参考来自用户计划，不作为实机标定 |
| SDK 平台 | [官方 SDK 要求](https://www.stereolabs.com/docs/development/zed-sdk/specifications)、[安装入口](https://www.stereolabs.com/docs/installation) | 2026-10-20 核实支持 Windows、Ubuntu、Jetson Linux 和 NVIDIA GPU；当前 Mac 采用 MuJoCo 双目接口 |

新增逻辑仅包含 MJCF 相机组件、相机参数与坐标变换、原场景扩展、双目采集、预览和验证。没有迁移或复制控制器、训练环境、射门和碰撞逻辑。

上游 K1 的 BSD-3-Clause 许可证见 [licenses/booster_assets-BSD-3-Clause.txt](licenses/booster_assets-BSD-3-Clause.txt)（原资产目录中也保留）。ZED 资产采用 Apache-2.0，上游 LICENSE 完整保存在 [licenses/zed-ros2-description-Apache-2.0.txt](licenses/zed-ros2-description-Apache-2.0.txt)，未删改其中附带的其他声明。

相机外观网格未修改；新增 MJCF 包装移除了质量与碰撞作用，使用仿真安装偏移和视觉避让偏移，不代表实物安装或镜头标定。
