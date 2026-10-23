# Isaac / ZED / 远端感知调研证据

本文件配套 [执行计划](isaac_zed_execution_plan.md)。区分已读源码、已有产物、官方文档和待运行验证；没有在远端安装、升级、启动仿真/推理或操作实机。

## 1. 用户已确认的范围

- 只做仿真；ZED X 2.2 mm 广角版，不安排 Jetson/GMSL 或实机采集。
- 执行：`lizhengran@robotic-4090`，只使用物理 GPU0、GPU1。
- 参考：`pc@100.75.243.43` 的本体感知和主动感知代码。
- 交付：可由后续 Codex 执行的详细计划、技术方案与验证要求；本次尚未完成 Isaac/ZED 实施。

原 ChatGPT 对话读取到了六项用户需求；助手回复只返回内容引用占位符，未作为已达成的设计结论。当前用户补充优先于历史方案。

## 2. 正式执行机实查

VS Code 已有连接 `ssh-remote+lizhengran-server`；复用现有 SSH 别名 `lizhengran-server` 连接后 hostname 返回 `robotic-4090`。未修改 SSH 配置或主机密钥策略。

| 项目 | 只读结果 |
|---|---|
| OS | Ubuntu 20.04.6 LTS，kernel 5.15.0-136-generic |
| GPU | 共 8 张 RTX 4090，各 24564 MiB；本任务仅 GPU0/1 |
| 可用资源快照 | GPU0 23935 MiB、GPU1 24066 MiB 空闲；RAM 377 GiB，available 359 GiB；根盘可用约 537 G |
| Driver | 575.57.08 |
| CUDA | `/usr/local/cuda/bin/nvcc` = 12.4.99，登录 PATH 未找到 nvcc |
| Isaac | `/home/lizhengran/software/isaac-sim-4.5.0`；VERSION=`4.5.0-rc.36+release.19112.f59b3005.gl` |
| ZED | 常规 `/usr/local/zed/include/sl/Camera.hpp` 不存在，当前 Isaac exts 未见 ZED 名称；不代表扫描了所有自定义安装位置 |
| 容器 | Docker 和 nvidia-container-cli 1.18.1 已存在；当前用户访问 Docker socket 被拒绝，未 sudo、未改组 |

资源值是调查时快照，不是预约。仅查询了其他 GPU 的资源，没有启动/停止任务。驱动、CUDA 编译器、Python 自带 runtime 不能混为一项。

已有工作目录 `/home/lizhengran/isaacsim-workspace` 可以参考：

- `scripts/validate_rtx.py:3` 在创建 SimulationApp 前调用 `libcuda.so.1/cuInit(0)`；显式设置 active_gpu、physics_gpu 和 multi_gpu=False。此为历史成功启动路径，根因未完全建立，不能当通用修复定律。
- `outputs/isaacsim-validation-20261023.txt` 历史记录检查过 Kit Vulkan 设备表和 GPU0/1。
- 本轮独立读取 `outputs/gpu0/result.json` 与 `gpu1/result.json`，shape 均为 `[480,640,4]`，RGB 标准差约 46.584；两张 PNG 签名和 IHDR 尺寸核对为 640×480。未重新运行渲染，也未视觉打开远端图像。
- WebRTC 历史记录区分了视频可见和键鼠未可靠验证。批量仿真/SDK 测试无需等待交互串流；不把临时 UDP 转发工具作为正式架构。

## 3. 版本选择与尚待验证点

### 首选小规模兼容性试验

保留宿主 Ubuntu20.04 / driver575 / 已有 Isaac4.5，候选为 **官方扩展 v3.2.0 + SDK 5.0.7 的 Ubuntu20/CUDA12/TensorRT10 发行包**。

依据与限制：

- [v3.2.0 release](https://github.com/stereolabs/zed-isaac-sim/releases/tag/v3.2.0) 对应 Isaac4.5；[扩展 tag 配置](https://raw.githubusercontent.com/stereolabs/zed-isaac-sim/v3.2.0/exts/sl.sensor.camera/config/extension.toml) 对应 Kit106.5；[该 tag 变更记录](https://raw.githubusercontent.com/stereolabs/zed-isaac-sim/v3.2.0/exts/sl.sensor.camera/docs/CHANGELOG.md) 可追溯 SDK5.0 支持。
- [SDK5.0 官方下载页](https://www.stereolabs.com/developers/release/5.0) 有 5.0.7 Ubuntu20/CUDA12 包；[SDK5.1 页](https://www.stereolabs.com/developers/release/5.1) 桌面下载列 Ubuntu22/24。不能因参考机装了 5.4.1 就给目标 Ubuntu20 安装同一包。
- [v3.2.0 README](https://raw.githubusercontent.com/stereolabs/zed-isaac-sim/v3.2.0/README.md) 要求 Ubuntu22，**所以目标宿主上的整套组合未被证明支持**。先检查二进制 GLIBC/GLIBCXX、共享库依赖和 Kit ABI，再在隔离位置验证；不能靠 Python venv 解决系统 glibc 不兼容。
- driver575 满足 CUDA12.4 驱动层最低版本要求，但这不等于 SDK/扩展/Isaac 全栈通过。[CUDA12.4 发行说明](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html)
- 建议独立 C++ SDK 接收进程，先避开 Isaac 自带 Python 与 pyzed ABI 组合；若采用 Python，单独锁定其支持版本。当前 Isaac 的 VERSION 带 rc 标识，也须实际验证，不能只按目录名判断正式发行版兼容。

先检验依赖再安装/启动，不在共享机原地升级 OS/driver。若库依赖不满足，可准备匹配的 Ubuntu22 用户空间/容器方案；当前容器权限需由环境管理员提供，不能假设已有权限。也可选择具备官方支持组合的独立执行环境。SDK 关口受阻期间，仍可在已有 Isaac4.5 上推进几何、物理案例和数据协议，但不得把此进展报告为 SDK 目标已完成。

### 新版备选，不作为当前默认

[官方扩展 v5.3.0](https://github.com/stereolabs/zed-isaac-sim/releases/tag/v5.3.0)（release commit `f916097`）支持 Isaac6.0.1/6.1；接收 SDK 最低 5.5.0，候选 5.5.1。[SDK5.5 发行说明](https://docs.stereolabs.com/docs/development/zed-sdk/release-notes/5-x/5-5)

这不是现有宿主直接升级的建议：[Isaac6.0.1 x86_64 要求页](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/requirements.html) 列 driver595.58.03，而 [6.0.0 要求页](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html) 列580.95.05；不能混用小版本、架构或 latest 的要求。Ubuntu、驱动和权限满足后才选择新路线；容器不能替代宿主驱动。

v5.3 的 Sim2Real、helper 参数与 GT streaming 不能套用到 v3.2。该 tag build 脚本中的模拟库版本也与接收 SDK 版本不同，不人为改为同一版本号。固定 tag/commit；本次官方 main、www/docs 索引存在缓存差异，不能用 main 代替发行记录。

## 4. 真实 SDK 接入与曝光边界

[官方扩展概览](https://docs.stereolabs.com/docs/integrations/isaac-sim) 提供虚拟相机图像经 IPC/RTP/stream 进入真实 SDK 的路径；新版 [设置文档](https://docs.stereolabs.com/docs/integrations/isaac-sim/setting-up-the-zed-in-isaac-sim) 区分左右 RGB 流和 `Stream Depth`。

正式目标是左右 RGB 送 SDK 算深度；`Stream Depth=true` 的 GT 注入只做 Oracle。旧版 [extension<4.0 教程](https://docs.stereolabs.com/docs/integrations/isaac-sim/old-version-of-the-zed-extension-for-isaac-sim/setting-up-the-zed-in-isaac-sim) 已描述图像经 SDK streaming input 使用，GT streaming 是后续扩展功能。旧版 C++ 全源码读取未完成，因此最终要以接收左右图、SDK 深度及右图扰动独立验证为准。

普通 PNG/MP4 不等于 SDK 可直接接受的输入；[InputType API](https://www.stereolabs.com/docs/api/classsl_1_1InputType.html) 的公开入口不能解释为任意左右 NumPy 图像 API。SVO 可作为成功接入后的复现候选，但须一起归档标定、帧时间和 SDK 版本，并关闭模拟器后实际回放验证。[SVO 录制说明](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/recording)

参数语义：ZED X Wide 2.2 mm、120 mm 基线、全局快门是厂商规格；本项目 10 ms 是实验选择。[ZED X 规格](https://docs.stereolabs.com/docs/products/cameras/zedx/specifications)

- 当前 API 的 `EXPOSURE_TIME` 使用微秒，10 ms=10000；`ANALOG_GAIN` 使用 mdB，`DIGITAL_GAIN` 是因子。它们不等于 legacy `EXPOSURE=0..100` 或简单 RGB 亮度乘数。针对实际选择的旧 SDK，应再核对该版头文件枚举、范围和返回错误。[Video API](https://www.stereolabs.com/docs/api/group__Video__group.html)
- SDK 接收端曝光设置是否反向控制模拟器，本次未证实；旧/新扩展都要做图像输出验证。unsupported 流控制必须显式记录，不能用 getter 当闭环证据。
- 最新 Sim2Real 是拟合成像模型，不能声称运行了实机 ISP。不同 capture/stream 路径可能包含不同的 AE、AWB、噪声和模糊效果，按固定版本分别验收。
- 固定曝光若官方 helper 不支持，可采用可控渲染曝光积分作为“仿真假设模型”，与官方自动成像模型分别报告；不将实机测量列为本次必需条件。

## 5. 参考机可复用接口

`pc@100.75.243.43` 本轮查到单张4090、driver580.159.03、ZED5.4.1、Isaac4.5.0.0。这里只是代码来源，版本不决定正式执行环境。

| 远端路径 | 可参考内容 | 注意事项 |
|---|---|---|
| `/home/pc/zed_mini/zed x/camera_types.py:26` | 标定、StereoFrame、DepthFrame、BallMeasurement3D、帧号/标定ID/来源 | SDK图像时间未宣称是曝光起点或中点 |
| `/home/pc/zed_mini/zed x/zed_source.py:19` | live/SVO/stream，m 单位、IMAGE坐标、深度开关、接收时刻 | 只参考 stream 适配与数据契约，不迁移 live 设备配置 |
| `/home/pc/zed_mini/zed x/zed_sdk_adapter.py:31` | 读已打开相机的校正参数；右目相对左目投影变换 | 右光心 +B 与点坐标变换 −B 的方向不能混淆 |
| `/home/pc/zed_mini/zed x/projection.py:68` | 具名坐标变换、DLT三角化、像素扰动协方差 | 可作为独立几何对照 |
| `/home/pc/ssr/app/zean_ball_backend.py:103` | 双目 YOLO/匹配、帧去重、pair age、测量与预测分离 | 现有阈值是来源系统假设，不直接作为目标系统精度保证 |
| `/home/pc/zean.xi/stereo_ir_ball_v1/ball/detector.py:20` | Ultralytics 推理与输入处理 | 乒乓球 IR 权重，不是已验证足球 RGB 模型；先评估适用性 |
| `/home/pc/ssr/app/d455_deploy_perception.py:229` | 按 measurement_t 查历史位姿、左IR→RGB→桌心、旋转协方差 | 目标改成 ZED/机器人/世界，不复制 D455 外参 |
| `/home/pc/ssr/app/d455_switch_geometry.py:60` | Slerp/平移插值、间隔和时效检查 | 等待后一个位姿可产生额外可用延迟，不能称零延迟 |
| `/home/pc/ssr/active_rl/perception_api.py:15` | 3D measured 与 measurement_t 时效检查 | 复用边界，不引入整个 RL/实机栈 |
| `/home/pc/ssr/active_rl/head_node.py:151` | 头部关节历史插值 | 局部函数越界夹取未自带年龄上限，独立复用需加明确时效约束 |

主动感知 `ssr/active_rl/README.md` 使用冻结 RGB v5 detector、head20 actor、50 Hz smoother 和两颈关节。它不是 K1 头部策略的可直接兼容资产；本次先以固定/预设头部运动做感知评测，不搬实机发布器、A3 控制或训练流程。

## 6. 本地独立验证与研究限制

- 重读两组 `results/moving_stereo*` 的60帧 JSONL/NPZ：左右及真值索引时间一致；相机最大平移约1.284 mm，属于站立微动，不能当横移/主动转头感知验证。
- `camera.py` 的理想对应点三角化要求左右行差极小，它是几何诊断，带噪检测应用应使用标定/校正和适当匹配门槛。
- 当前 exposure 是恒 qvel 的 RGB 平均近似，无真实接触期间轨迹积分；详细改进与中点对照见执行计划。
- 当前本地 `.venv` 实查 MuJoCo3.3.7、NumPy2.3.3；旧运动采集文档另记录 conda robot 环境，两者不能混称同一版本。
- 本轮执行两条 pytest 命令，8+17 项通过、后者11项未选中；没有重新做 GPU渲染、完整物理或 SDK测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  -c Robot-Perception-Sim/pytest.ini Robot-Perception-Sim/tests/test_recording.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  -c Robot-Perception-Sim/pytest.ini Robot-Perception-Sim/tests/test_stereo_camera.py \
  -k 'nominal_calibration or invalid_calibration or invalid_ideal_correspondence or vendored_camera_asset_fingerprints'
```

实施前仍须冻结安装包完整校验、GPU UUID、目标工程路径及软件支持矩阵。官方文档证明路线存在，实际运行结果才证明目标环境完成接入；以上未验证项均已在执行计划中作为关口或失败分支。
