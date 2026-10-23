# G0 目标机与 G2 独立依赖准备

审计日期：2026-10-24，Asia/Shanghai。本文只记录环境与二进制准备，不代表 G2 的 100 连续帧验收通过。机器快照见 `results/g0/host_inventory.json`。

## 目标与保护边界

VSCode 当前连接 `ssh-remote+lizhengran-server`；现有 SSH 配置对应用户 `lizhengran`，服务端 hostname 为 `robotic-4090`。复现连接：

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 lizhengran-server
```

仅使用用户授权的 GPU 0、1。其余 GPU、系统 OS/driver、旧 Isaac Sim 4.5 安装均未改动。本轮不使用 sudo，不启动 Isaac/GPU 计算，不运行 SDK 系统安装脚本。新增远端依赖位于 `/home/lizhengran/isaacsim-workspace/g2_zed_poc/`。

| 项目 | 本轮实测 |
|---|---|
| OS / kernel | Ubuntu 20.04.6 LTS / 5.15.0-136-generic |
| glibc / GCC / CMake | 2.31 / 9.4.0 / 3.16.3 |
| NVIDIA driver | 575.57.08 |
| CUDA toolkit | `/usr/local/cuda/bin/nvcc` 12.4.99；登录 PATH 没有 nvcc |
| GPU 0 | RTX 4090，24564 MiB，PCI `00000000:4F:00.0` |
| GPU 1 | RTX 4090，24564 MiB，PCI `00000000:50:00.0` |
| RAM / 根盘 | RAM 377 GiB、available 359 GiB；根盘可用约 537 GiB（首次快照） |
| 旧 Isaac | `/home/lizhengran/software/isaac-sim-4.5.0`，VERSION `4.5.0-rc.36+release.19112.f59b3005.gl` |
| Docker | 工具存在，但当前用户访问 daemon socket 被拒绝；未提权或改组 |
| 原 ZED | `/usr/local/zed` 常规 SDK 路径不存在；旧 Isaac exts 未发现 ZED |

登录环境 `CUDA_VISIBLE_DEVICES` 未设置。启动时应显式限定 `0,1`，并检查 Kit Vulkan 设备实际 PCI bus id；不能仅以 CUDA 逻辑 ordinal 判定 RTX 使用哪张物理卡。

## 既有 RTX 结果

`/home/lizhengran/isaacsim-workspace/scripts/validate_rtx.py` 使用 `active_gpu`、`physics_gpu` 和 `multi_gpu=False`，先调用 libcuda `cuInit` 后启动 SimulationApp。旧日志记录根因尚未完全确定的 CUDA 初始化差异，不能省略已验证启动顺序。

本轮只读复核 `outputs/gpu0/result.json`、`outputs/gpu1/result.json`，均为 640×480 RGBA，RGB 标准差分别约 46.584117 与 46.584236；两张 PNG 文件签名、IHDR 尺寸和文件存在性独立核对。历史记录称两卡已通过 Kit Vulkan 表交叉核验；本轮未重新渲染或视觉打开远端 PNG。

## 固定官方依赖与下载

### Receiver：ZED SDK 5.0.7

[官方 5.0 发布页](https://www.stereolabs.com/developers/release/5.0) 给出的 Ubuntu20/CUDA12 链接：

`https://download.stereolabs.com/zedsdk/5.0/cu12/ubuntu20`

实际重定向文件为 `ZED_SDK_Ubuntu20_cuda12.8_tensorrt10.9_v5.0.7.zstd.run`，2,411,022,033 bytes。下载文件位于独立目录的 `downloads/`，SHA-256：

`c187a967991ff151dd67d287322de76b501de9aae27e40cbd034b3a22bd4560e`

Makeself 内嵌 payload MD5 `475a356bfa6be5479bf04e8532946860` 与完整下载重新计算值一致。归档头长 18,884 bytes，717 行。仅解包，没有执行 `linux_install_release.sh`。

系统无 zstd 命令且系统 Python 无 pip，使用旧 Isaac 自带的**纯 Python 解释器**将 `zstandard==0.23.0` wheel 安装到独立 `python_deps/`，以 Python 流式解包；没有更改旧解释器依赖。解包对每个路径做越界检查。

真实路径：

- SDK 根：`/home/lizhengran/isaacsim-workspace/g2_zed_poc/sdk-5.0.7-unpacked`
- 头文件：`include/sl/Camera.hpp`，版本宏明确 5.0.7。
- 主库：`lib/libsl_zed.so`，147,871,728 bytes。
- 另有：`lib/libsl_ai.so`、`dependencies/ai_dependencies/`。

对 `libsl_zed.so` 的 `ldd` 无 `not found`；ELF 要求最高 `GLIBC_2.29`、`GLIBCXX_3.4.26`，主机满足。直接依赖包括 libcuda、libnvcuvid、libnvidia-encode、libpng16、libjpeg8、libturbojpeg、libusb、libstdc++、libgomp 等，均可解析。这只能证明动态链接边界，不证明相机流或 GPU 深度运行成功。CUDA12.8 的运行行为仍须 G2 实测，不因主库无缺库就宣称 CUDA12.4 完全等价。

TensorRT10 未在系统 ldconfig 中发现，但 SDK 归档自带 TensorRT10.9，位于 `dependencies/ai_dependencies/lib`。`libsl_ai.so` 的裸 ldd 缺 nvinfer/plugin/onnxparser/sl_zed；仅对该 ldd 进程设置 `LD_LIBRARY_PATH=<SDK>/lib:<SDK>/dependencies/ai_dependencies/lib:/usr/local/cuda/lib64` 后全部解析。没有系统安装 TensorRT，也未实际运行 AI。基础 stereo 路线应从明确非 neural 模式开始，不能把渲染器深度当 SDK 双目深度。

### Sender：官方 v3.2.0 扩展

[官方 release v3.2.0](https://github.com/stereolabs/zed-isaac-sim/releases/tag/v3.2.0) 发布时间为 2025-07-23，说明对应 Isaac Sim 4.5 并增加 Linux IPC。release 的 Linux x86_64 预编译资产为 `stereolabs-zed-isaac-sim-linux-x86_64-v3.2.0.zip`，34,506,992 bytes。

v3.2.0 README 指定 Ubuntu22.04。预编译包完整下载并解压到 `/home/lizhengran/isaacsim-workspace/g2_zed_poc/extension-v3.2.0`，SHA-256 为 `e32f92650c9c38f527da5879460b26703a7d6066583305c667099d574aede160`。其 `exts/` 仅包含 `sl.sensor.camera.bridge`，没有 Python camera 扩展及正式 ZED_X.usdc，需要固定 tag 源码补齐。

**预编译 sender ABI 未通过，未启动。** `exts/sl.sensor.camera.bridge/bin/libsl.sensor.camera.bridge.plugin.so` 要求 GLIBC2.32/2.34 和 GLIBCXX3.4.29；自带 113,232,024 bytes 的 `libsl_zed.so` 要求 GLIBC2.35、GLIBCXX3.4.30、CXXABI1.3.13。主机 glibc2.31 不满足，ldd 显式报版本 not found。plugin 另需 Kit 的 boost_python310、libpython3.10、libgf，这些可检查旧 Kit 路径；但补 Kit 路径不能修复 glibc 不匹配。

因此源码重编 bridge 本身也不够：闭源发送 runtime 仍需兼容用户空间。SDK5.0.7正式接收库的导出符号中没有 `init_streamer/stream_rgb/stream_yuv`，不能拿它代替私有sender。禁止将新glibc塞入旧Isaac进程试图掩盖ABI缺口，也不升级系统。

固定tag `4a3fab88235e2815c07c1cd5e9fa7c2c17e00b7a` 的Python扩展与USD已和同版官方预编译bridge组合在 `g2_zed_poc/zed-v3.2.0-combined/`；扩展目录为 `exts/`，正式相机为 `usd/ZED_X.usdc`（18,795,469 bytes，SHA-256 `aedba222a0b7db8d909f51977f35353aac15fa13e5e4153e79225c9f2e06d79b`）。未篡改上游版本检查。

目标已有bubblewrap，用户命名空间只读 `id` 测试已通过，不需要Docker权限或宿主root。正在独立 `ubuntu22-rootfs` 内准备官方Ubuntu22用户空间，旧Isaac只读挂载、宿主NVIDIA驱动只读映射；这条路径的GPU运行、扩展加载和收流仍须G2实际验证，不能据隔离环境建立成功就标链路通过。

已从官方源码确认 sender loader 的接口版本要求 SDK≥5.1.0；这是扩展自带发送库的版本边界，**不能用 receiver 5.0.7 的 lib 覆盖 sender**。两个进程分别设置精确库路径。跨版本收流是否成功必须由 G2 真实 API 验证。

## 验证命令与未完成项

```sh
nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,memory.total,memory.free,driver_version --format=csv,noheader
ldd /home/lizhengran/isaacsim-workspace/g2_zed_poc/sdk-5.0.7-unpacked/lib/libsl_zed.so
readelf --version-info /home/lizhengran/isaacsim-workspace/g2_zed_poc/sdk-5.0.7-unpacked/lib/libsl_zed.so
```

环境准备不替代 `sl.Camera.open/grab/retrieve_image/retrieve_measure` 的实际结果；100连续帧、时间戳/配对、有效深度、真实使用 GPU、SDK 日志与 Sim2Real 功能均由独立 G2 报告验收。旧版本若不提供现代 Sim2Real 功能，应如实标记不支持，不补造能力。
