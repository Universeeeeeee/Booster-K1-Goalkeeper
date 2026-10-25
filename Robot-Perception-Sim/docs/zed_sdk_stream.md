# 官方 ZED SDK 流 PoC：G2 最终验收

日期：2026-10-24（Asia/Shanghai）。本轮仅完成 G2 修复与验收，未继续 G1/G3 或后续阶段。**固定环境下的两轮各100帧验收通过，正常退出、异常非零退出、同端口重启和图像—时间对应均有实际证据。**

这不表示完整项目已经完成，也不表示30fps实时性能、真实曝光或真实硬件标定已通过。

## 结果

| 检查 | 最终运行022 / receiver_007_final | 最终运行023 / receiver_008_final |
|---|---:|---:|
| 连续成功、唯一 SDK grab | 100 | 100 |
| SDK/源事件/左目ID/右目ID完整对应 | 100/100 | 100/100 |
| SDK参考时间与实际渲染事件参考的最大差 | 0 ns | 0 ns |
| 2m平面中心深度中位数 | 2.018883586 m | 2.018857479 m |
| 中央区域有效深度比例最低值 | 100% | 100% |
| 实测接收端host吞吐（含诊断、保存） | 9.96491 fps | 9.91366 fps |
| sender / receiver实际退出码 | 0 / 0 | 0 / 0 |

两轮复用端口30000，第一轮进程确实退出后才启动第二轮。主agent在本地重新解码两组全部PNG与float32文件，重算深度及逐帧身份/时间关联，均退出0；不是只读取采集程序自己的summary。测试代码17项通过，另有独立静态审阅，无阻止提交的实质问题。

独立读取两轮renderer GT，中央深度均约2.000000238m，与SDK深度分开。实际SDK图像已视觉检查；静态平面RGB与源图的直接通道误差明显小于红蓝互换对照，不把BGRA内部格式误当PNG通道顺序。

仅使用物理GPU0（UUID GPU-50273ff3-fae7-f61c-4197-088b4d8a0037、PCI 00000000:4F:00.0）。023的NVML记录中发送器PID2189716与接收器PID2190463均在GPU0；完成后再次检查，没有本轮发送/接收进程残留。GPU1及其余GPU未用于本轮最终实验。

## 修复内容与边界

### 1. 正常关闭与错误传播

原来的完整逐扩展关闭模式（fast_shutdown=False）在不加载ZED的最小Isaac对照中也发生GC崩溃。native GDB定位到omni.appwindow/_appwindow → carb._carb → Python模块字典析构/GC，不能将其归因于SDK深度或只靠ZED.destroy修复。

当前采用该Isaac版本官方默认的fast_shutdown=True。调用者先停止World、detach额外观察器/Writer、销毁官方annotator、清除World实例并释放函数作用域内的原生包装对象；在插件仍加载时进行GC，再通过公开IApp.post_quit(return_code)与SimulationApp.close关闭。未修改NVIDIA引擎，未加入自写的成功强退或将崩溃改报为成功。

实际负向运行010在应用已启动后使用错误USD产生真实场景异常，保留traceback并退出1；正常022/023均退出0。因此默认关闭方式不会吞掉场景异常。**该旧Kit的fast_shutdown=False底层问题仍存在，不属于本PoC支持的关闭配置。**

### 2. 不再把物理时钟冒充图像时间

原版ReadSimulationTime接线在015的100帧中有2帧时间标签领先实际像素状态一帧。失败数据保留，未放宽阈值。

适配层把SDK输入时间接到官方RationalTimeSyncGate的成对渲染参考分数，经原生Divide转成double。它是SyntheticData NewFrame的参考域，不等同于World、Kit timeline或Unix时间；原SDK原始纳秒值保留，不拟合常量偏移或擅自改写数据。

单纯在Python循环中轮询图节点仍会漏事件：020漏掉一个真实SDK参考，严格验证正确失败。最终改为官方Replicator Writer：同一write数据包中包含reference_time及两目实际RGB，逐事件写入render_observations.jsonl。source_frames.jsonl只保存作者状态更新，两者不混成一张假“发送帧表”。

画面中的23格诊断条编码源状态ID、哨兵和校验位。验证器独立读取源端Writer观察与SDK保存的左右PNG，要求每个SDK参考唯一对应一条有效的源图像身份，两目身份一致，并能查到作者World状态。这是逐帧显式关联，不是通过时钟拟合制造一致性。该方法验证本瞬时成像PoC；真实曝光时间窗与移动机器人姿态对齐仍属于G3。

### 3. 验证实际渲染配置

仅设置构造参数anti_aliasing=0并未生效，实际渲染器仍被limitedOps限制在DLSS模式，动态诊断条发生时域混合。按当前官方Replicator配置路径关闭limitedOps，再选择AA op0；读取实际设置并检查像素，未降低解码阈值掩盖混合。

最终元数据记录AA op0、limitedOps=False。相机使用官方ZED_X.usdc和运行时SDK标定：960×600、fx/fy约370.8px、主点(480,300)、基线约0.12m。不能混用旧MuJoCo名义焦距336.1px，也不把这些参数标为逐台实机标定。当前为瞬时渲染基线，未启用现代Sim2Real成像模型。

## SDK深度来源

正式链路为官方扩展左右RGB → IPC → 正式SDK5.0.7的PERFORMANCE立体深度。接收器关闭异步图像获取，每次grab后读取左右图、深度、置信度并检查IMAGE时间戳前后不变；SDK的Mat.timestamp记录矩阵最近数据操作时间，不能作为独立的逐目传感器采集时刻；本验收不把它当作逐目时间证据。

renderer distance_to_image_plane只保存到单独GT文件，不送入SDK。发送层/接收代码没有ingestCustomDepth调用。此前右目负对照仅把真实基线0.12→0.14m，SDK标定仍0.12m，物理平面仍2m；SDK测得1.7191205m、视差26px，符合错误标定预测2×0.12/0.14=1.7142857m，而不是继续回传2m真值。原文件不改，旧控制用于证明深度依赖双目；它不是对真实2m的测距通过。

最终接收参数、原始SDK日志、标定和文件格式全部保存在每次capture目录。PNG为SDK rectified U8_C4经Mat.write保存；深度与置信度为连续行float32 little-endian，单位分别为m和SDK confidence。验证器逐文件解码、检查尺寸、重算有效深度，不能仅凭metadata声明证明深度来源。

## 环境与代码

宿主Ubuntu20.04、driver575.57.08保持不变。v3.2发送库需要较新的glibc，因此在用户拥有的Ubuntu22 rootfs中，通过现有bubblewrap运行既有Isaac4.5。旧Isaac只读挂载，kit/cache映射到PoC私有缓存；仅映射宿主NVIDIA驱动库，没有替换宿主glibc、驱动或Docker权限。私有PID namespace的proc允许进程写自己的thread comm，宿主NVIDIA proc树仍只读。

调试器及所需图形库仅下载/解包到私有rootfs；最终私有libc身份以本次manifest为准，不拿最初UbuntuBase包版本代替实测。fontconfig的包配置曾因未映射组而失败，原日志保留；不宣称整套apt安装零错误。

| 文件 | 职责 |
|---|---|
| [zed_sim_poc.py](../scripts/zed_sim_poc.py) | 官方相机最小场景、参考时间接线、逐事件观察与显式关闭 |
| [zed_sdk_receiver.cpp](../isaac/zed_sdk_receiver.cpp) | 真正的SDK stream consumer、原始文件保存及连续抓取检查 |
| [CMakeLists.txt](../isaac/CMakeLists.txt) | 明确链接独立SDK，不安装系统SDK |
| [run_in_ubuntu22.sh](../isaac/run_in_ubuntu22.sh) | 此目标机的隔离用户空间与只读驱动映射 |
| [verify_zed_sdk_stream.py](../scripts/verify_zed_sdk_stream.py) | 不加载Isaac/SDK的独立文件、像素及参考时间验证 |
| [test_sdk_capture.py](../tests/test_sdk_capture.py) | 17项有效/非法数据、时钟与图像身份测试 |
| [g2_acceptance.json](g2_acceptance.json) | 轻量验收指标、源码身份与原始证据索引 |

- 发送脚本SHA-256：6167ee231c3388c363f56b66b8e0cf60360748ac9ee1edb318f0ef6e854b9278
- 验证器SHA-256：a3929b389a59c24bb8e56e5ae940d4e1c5f5d0be98a7e4676ed142a849d3821b
- 实际接收二进制SHA-256：7f00e31d3ff02bf55eb2bb724d3dd9d0835d77d8d7afe94e47e4d64309855d6c
- 最终私有libc SHA-256：5955dead1a55f545cf9cf34a40b2eb65deb84ea503ac467a266d061073315fa7

固定官方扩展tag与USD/bridge身份见[g0_host_compatibility.md](g0_host_compatibility.md)。发送私有库和接收SDK库必须分进程配置；不得用receiver 5.0.7覆盖sender的私有库。

## 复现

目标机已有独立依赖目录：
/home/lizhengran/isaacsim-workspace/g2_zed_poc

将上述Python脚本放到该目录、C++和CMake放到receiver_source，隔离wrapper同名放到根目录。使用已有官方依赖，不重装宿主系统。编译：

    task_poc=/home/lizhengran/isaacsim-workspace/g2_zed_poc
    cmake -S "$task_poc/receiver_source" -B "$task_poc/receiver_build" \
      -DZED_SDK_ROOT="$task_poc/sdk-5.0.7-unpacked" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$task_poc/receiver_build" -j2

发送端（选一个从未使用的输出目录，保留终端的真实退出状态）：

    env CUDA_VISIBLE_DEVICES=GPU-50273ff3-fae7-f61c-4197-088b4d8a0037 \
      PYTHONPATH=/poc/python_deps PYTHONUNBUFFERED=1 \
      bash "$task_poc/run_in_ubuntu22.sh" /isaac/python.sh /poc/zed_sim_poc.py \
      --extension-root /poc/zed-v3.2.0-combined/exts \
      --camera-usd /poc/zed-v3.2.0-combined/usd/ZED_X.usdc \
      --output /poc/runs/sender_NEW --gpu 0 --fps 30 --allow-root

等待ready且source_frames至少10行后，在另一终端运行接收端（同样使用新目录）：

    task_poc=/home/lizhengran/isaacsim-workspace/g2_zed_poc
    env ZED_SDK_PATH="$task_poc/sdk-5.0.7-unpacked" \
      LD_LIBRARY_PATH="$task_poc/sdk-5.0.7-unpacked/lib:$task_poc/sdk-5.0.7-unpacked/dependencies/ai_dependencies/lib:/usr/local/cuda/lib64" \
      CUDA_VISIBLE_DEVICES=GPU-50273ff3-fae7-f61c-4197-088b4d8a0037 \
      "$task_poc/receiver_build/zed_sdk_receiver" --ip 127.0.0.1 --port 30000 \
      --frames 100 --output "$task_poc/runs/receiver_NEW" --gpu 0 --timeout-seconds 120

接收结束后在自己的发送目录创建stop文件，并等待原发送进程实际退出；不要停止其他作业：

    touch "$task_poc/runs/sender_NEW/stop"

本地或具备NumPy/Pillow的Python可脱离模拟器独立验收：

    .venv/bin/python Robot-Perception-Sim/scripts/verify_zed_sdk_stream.py \
      Robot-Perception-Sim/results/g2/receiver_007_final --frames 100 \
      --source Robot-Perception-Sim/results/g2/sender_022_final \
      --report Robot-Perception-Sim/results/g2/new_readback.json

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
      -c Robot-Perception-Sim/pytest.ini Robot-Perception-Sim/tests/test_sdk_capture.py

连续运行不能共用同一输出目录。接收器120秒deadline防止无限等待；超时会记录失败并非零退出，不属于成功关闭路径。不得用成功打开设备、配置fps或进程返回值单独代替完整数据验收。

## 证据保留与限制

最终完整数据在本地results/g2以及目标机PoC的runs目录：
sender_022_final、receiver_007_final、sender_023_final、receiver_008_final。旁有实际exitcode、trial日志、SDK日志、NVML记录及两份主agent独立readback报告。大型原始数据不进入Git；提交轻量验收摘要、代码、测试和文档。

保留008无效生命周期修复、no-ZED对照/GDB、010异常退出、013设置被覆盖、015两帧错标、020轮询漏事件等原始失败证据。没有删除失败样例或改写旧metadata。早期Unix时间标签错误仅作勘误；新consumer保留未解释的原始stream时钟，由来源验证器明确其域。

配置30fps，实际落盘接收约10fps；循环的历史summary.rendered_frames字段实际计作者world.step次数，不等于Writer事件数或真实SDK帧数。两轮SDK分别未观察到193/192个中间作者状态，这不是已定位的网络丢包统计。诊断Writer有CPU读回/解码开销，不声称30fps实时或零丢帧。

本轮验证仅覆盖该固定版本、单卡、单个官方虚拟相机、瞬时成像和2m纹理平面。曝光/噪声真实性、移动机器人、本体估计、全部场景与30/60/120fps性能不在G2验收范围。按用户要求，提交后暂停。
