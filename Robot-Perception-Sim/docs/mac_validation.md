# P0–P1 macOS 验证报告

日期：2026-10-20（Asia/Shanghai）。范围为在既有 K1 守门仿真中增加相机资产和采集接口。开发文件全部位于 `Robot-Perception-Sim/`，提交仓库为 `Universeeeeeee/Booster-K1-Goalkeeper`。本提交 SHA 可通过仓库根目录 `git log -1 --format=%H -- Robot-Perception-Sim` 获取。

## 复用与新增

直接复用原场景组装函数、配置、球场、球门、足球、光照、K1 模型和头部双目光学坐标系，以及根目录 `.venv`。未新增 Python 依赖。机器人资产没有副本或重复下载流程。

新增文件职责：

| 文件 | 作用 |
|---|---|
| `models/zedx.stl`、`models/sources.json` | 官方相机外观网格、来源与哈希 |
| `models/zedx_stereo.xml` | 无质量、无碰撞的外观和双目 MJCF 组件 |
| `config/cameras/zedx_wide.json` | 名义参数、挂载偏移与参数来源分类 |
| `robot_perception_sim/camera.py` | 内参、光学坐标变换、理想投影与几何诊断 |
| `robot_perception_sim/scene.py` | 调用原场景构建器后添加相机，不复制场景逻辑 |
| `robot_perception_sim/stereo.py` | 对现有 MjData 采集左右 RGB 和同一状态时间戳 |
| `scripts/preview_stereo.py` | 静态截图、头部姿态设置、诊断元数据 |
| `tests/`、`pytest.ini` | 相机专项数学、集成、OpenGL 渲染验证 |
| `README.md`、`docs/`、`.gitignore` | 复现、计划、验证、来源和许可证；忽略生成结果 |

来源详见 [sources.md](sources.md)。K1 固定版本为 `12a97516f57469078707afb587a5f84c1cb110d1`。相机网格来自 Stereolabs `zed-ros2-description` 提交 `22468247c1653d08a1e213809f3e9ce76291487d`，原样保留单个 574,984 字节 STL 和上游许可证。

## SDK 选择与环境

[官方系统要求](https://www.stereolabs.com/docs/development/zed-sdk/specifications)列出的系统为 Windows、Ubuntu、Jetson Linux，PC GPU 为 NVIDIA；[安装入口](https://www.stereolabs.com/docs/installation)也未提供 macOS 安装。当前 Apple Silicon Mac 采用 MuJoCo 双目采集接口，未安装不受支持的官方 SDK。支持平台后续优先评估官方 SDK，且须验证仿真图像接入路径。

本机验证：Darwin 25.4.0 / arm64，Python 3.13.13、MuJoCo 3.3.7、NumPy 2.3.3、Pillow 11.3.0、pytest 8.4.2，均复用根目录环境。普通 Python 即可进行本轮离屏渲染，不需要 `mjpython`。

## 复现

在仓库根目录执行，无需安装或获取新机器人资产：

```bash
.venv/bin/python Robot-Perception-Sim/scripts/preview_stereo.py
.venv/bin/python Robot-Perception-Sim/scripts/preview_stereo.py --head-yaw 0.4 --head-pitch 0.3 --output-dir Robot-Perception-Sim/results/head_turned
.venv/bin/python -m pytest -q -c Robot-Perception-Sim/pytest.ini Robot-Perception-Sim/tests --junitxml=Robot-Perception-Sim/results/tests/junit.xml
```

已额外检查 `--head-yaw 99` 被明确拒绝。所有预览使用 `mj_forward`，没有推进物理、删除关节或添加站立控制器。

## 参数与坐标定义

| 参数 | 实际使用值 |
|---|---|
| 图像 | 左右 RGB，各 960×600 |
| 基线 | 配置 0.120 m，实测 0.12000000000000001 m |
| 水平视场角主约束 | 110° |
| 推导垂直视场角 | 83.50372346447085° |
| 内参 | fx=fy=336.0996183406607 px，cx=479.5，cy=299.5 |
| 镜头 | 理想针孔、零畸变、平行光轴、相同左右内参 |
| 时间 | 两目均为同一 MjData 状态时间；静态预览为 0 s |
| 曝光 | 10 ms 全局曝光，10 个中点子样本等权 RGB 累积；参考模式 60 fps，未测定实时帧率 |

像素坐标采用整数像素中心，左上像素中心为 (0,0)，图像边界在 -0.5 和 W/H-0.5。世界 Z 向上，K1 默认面向世界 −X。光学坐标 X 右、Y 下、Z 前。矩阵使用列向量，`T_A_from_B` 将 B 坐标变换至 A，平移单位为 m，JSON 矩阵按行存储，四元数顺序为 wxyz。

`T_world_from_camera = T_world_from_robot × T_robot_from_head × T_head_from_parent_optical × T_parent_optical_from_mount × T_mount_from_camera`。

挂载父节点复用 `head_booster_stereo_rgb_link`。其局部光学坐标中，虚拟双目中心平移为 [0.0351, 0, 0.04] m，旋转为单位旋转；左右目在 mount X 轴上分别位于 −0.06 m 与 +0.06 m。上述偏移属于仿真假设，不是 ZED 实物外参。

MuJoCo 相机局部 −Z 为前、+Y 为上；通过 Rx(pi) 转换到光学坐标。`T_left_from_right` 的旋转为单位阵，平移为 [+0.12,0,0] m，数值残差约 1e-15。

官方外观网格的 ROS 坐标转换为光学坐标，并以 [0,0.016,−0.020] m 放置在光心后方；该视觉避让为仿真假设，不承担机械安装验证。组件不具有质量、惯量或碰撞作用。

## 验证结果

**10 ms 曝光更新后：30 passed，0 failed，0 skipped，约 1.98 s。**

- 17 项纯参数、资产哈希及数学测试：内参、非法配置、非法理想对应点、网格与许可证指纹；包含负值/非有限曝光、非法子样本数和曝光超出帧周期的拒绝验证。
- 7 项场景集成测试：三组头部 yaw/pitch 的刚性跟随与前向光轴、三种深度投影/视差恢复、原场景动力学不变性。
- 6 项真实 OpenGL 渲染测试：完整场景左右 RGB 和足球可见性、三组头部姿态下的渲染标记与解析投影交叉验证、静态 0/10 ms 图像一致性、运动足球的曝光拖影。

头部姿态为 (yaw,pitch) = (0,0)、(0.4,0.3)、(−0.5,−0.2) rad。渲染标记覆盖 2、2.5、5、10 m 深度及图像中央和边缘，两目共 24 组测量。通过 RGB 阈值独立提取标记位置，最大像素误差 **0.3990405 px**，平均 **0.2829634 px**，小于 1 px 验收阈值。标记实验清空可视场景几何以排除地面遮挡，保留实际相机投影；完整场景足球可见性单独检查。

原场景与相机扩展场景的关节数、自由度数、执行器数、初始 qpos、关节范围、阻尼/armature、执行器参数、传感器类型，以及每个既有 body 的质量/惯量和每个既有 geom 的碰撞参数相符。质量矩阵与广义偏置力在相同状态下以 1e-12 绝对容差比较通过。新增 housing 的 contype=conaffinity=0，mount mass=0。

左右采集前后的 qpos 和仿真时间不变。当前采集接口用于同一线程的更新/采集循环；未实现并发采样锁或真实硬件时间同步。

默认球的解析投影约为左目 (485.622,379.423)、右目 (473.379,379.423) px，光学深度约 3.2943 m，视差约 12.2431 px。完整场景 segmentation 诊断确认足球实际出现在两路图像中；segmentation 不作为正式图像感知算法输入。

30 项测试的原始 JUnit 输出位于 `results/tests/junit.xml`；每组标记截图和误差记录位于 `results/tests/projection_*/`。`results/audit/verification.json` 记录了初版 97 个原项目文件的哈希保持一致，且子目录之外原有 Git 差异保持不变。原项目已有本地差异未纳入本轮提交。

曝光更新：`exposure_time_ms=10.0`，采样时刻为开始后的 0.5、1.5、…、9.5 ms，左右目使用相同的状态副本和采样区间。`timestamp_s` 仍指曝光开始，新增 `exposure_start_s`、`exposure_end_s`、`exposure_samples`。不调用 `mj_step`；使用当前 qvel 和 `mj_integratePos` 做短时匀速运动学近似，调用者的 qpos、qvel、qacc、ctrl 和 time 不变。

静止场景的 10 ms 图像与瞬时图像逐像素一致；20 m/s 横向运动球在两目中的红色像素覆盖宽度均增加，验证实际产生拖影，原始对照图位于 `results/tests/exposure/left_0ms.png`、`left_10ms.png` 及对应右目文件。该对照使用独立小型场景以测量曝光本身，没有增加新的轨迹或控制系统。

10 ms 与 120 fps 的 8.33 ms 帧周期不相容，参考帧率因此改为 60 fps。时间累积为等权显示 RGB 平均，不是辐射级曝光模拟；不模拟加速度、接触、自动曝光、亮度随曝光变化或传感器噪声。

## 真实 MuJoCo 截图

所有截图均由本机 MuJoCo 实际渲染，可用上面命令重新生成。生成目录默认不提交。

![K1 头部相机外观](../results/stereo/mount.png)

| 左目 | 右目 |
|---|---|
| ![左 RGB](../results/stereo/left.png) | ![右 RGB](../results/stereo/right.png) |

![原球场与相机扩展场景](../results/stereo/overview.png)

## 能力边界与后续建议

本轮完成相机接入、静态图像采集、近似运动曝光及几何验证。名义针孔模型和外观资产不能证明真实 ZED 镜头精度；未实现 SDK 深度、IMU、噪声、真实光学曝光响应、实时吞吐或端到端延迟模型。没有评估动态站立或图像足球检测性能。诊断元数据中的真值不从 `StereoCamera.capture` 返回。

按当前焦距与基线，视差 0.25 px 的局部误差在 3、6、10 m 深度约对应 0.056、0.223、0.620 m 深度误差；这只是理论敏感度，不是图像定位实验成绩。

P2 的最小建议：直接复用现有足球状态设置和射门逻辑，在本子目录增加按给定时间采集双目的入口。物理推进时优先复用已有站立/行走控制，或明确约束机器人；不要重新实现控制器。规则轨迹与 CSV 回放待确有需求时补充。本轮未实施 P2–P4。
