# G3 时间与曝光契约

G3 复用现有 `StereoCamera`、射门录制器、官方 ZED 发送器和 SDK 接收器。MuJoCo RGB 与真实 SDK 输出分别记录来源；SDK 深度仍来自双目匹配，不能用渲染真值替代。

## 四个时间域

| 时间 | 来源与含义 | 关联方法 |
|---|---|---|
| 仿真状态时间 | MuJoCo `data.time`；G3 SDK 的 `t_sim_state_s` 为作者采样时间，原生World时间单独保存 | 随保存的状态、位姿记录 |
| 曝光开始/结束/中点 | 两目共享同一个仿真时间窗 | 有限曝光保存全部中点子采样时刻；瞬时曝光三者相等 |
| SDK IMAGE 时间 | G2为成对渲染参考；G3为显式曝光中点参考的纳秒值 | 保留原值，通过真实两目像素ID关联源窗口，不拟合时钟偏移 |
| 主机单调时钟 | 采集开始/结束、SDK grab成功后、retrieve与落盘完成 | 同主机比较；不是Unix时间，不将其拟合成仿真时间 |

MuJoCo没有SDK时间，字段为null。其单调时钟只描述离线渲染成本，仿真曝光窗口不会因此变成真实传感器的墙钟曝光。SDK参考也不是World时间；只凭图节点当前值无法判定旧图像的状态。

## 调度和共享窗口

`acquisition/timing.py`以有理数计算 `origin + index/rate`，不反复累加已舍入的周期。60/120Hz RGB和200Hz IMU可以各自调用独立序号；这证明调度算法，不能据此声称已经录到120fps SDK或200Hz IMU数据。

选择1ms物理状态网格时取目标时刻之后的第一个状态，保存目标、实际状态和采样偏差。偏差必须小于1ms，不把ceil采样称为精确插值。原射门录制器继续使用固定起点加 `i/fps` 的调度及实际轨迹时间；没有增加第二套物理循环。

正曝光严格小于帧周期；零曝光是明确的瞬时基线。60Hz验证1/3/5/10ms，120Hz验证1/3/5ms。两目使用同一个状态副本及每个子采样时刻；顺序渲染两目期间不推进状态。`rolling_skew_s=0`指理想共享仿真状态，不声称两次GPU计算在主机上同时发生。

## 已有MuJoCo曝光模型

曝光计算沿用既有实现：复制调用者状态，用 `mj_integratePos` 按当前qvel外推，每个中点子采样调用 `mj_forward` 并渲染左右图，再平均编码后的RGB。保存开始/中点/结束、子采样时间和实际光学相机世界变换，不推进或修改调用者状态。

这个模型是**匀速状态外推、等权编码RGB平均**。它不积分接触/重力加速度、不等同线性辐射积分、不模拟相机ISP或自动曝光，也没有叠加RTX时域模糊。采样数量有限，细节可能发生时间混叠。静态/球运动/相机运动均通过实际PNG验证，不能仅由窗口元数据宣称图像已有模糊。

标定身份对K、图像尺寸、基线、投影和挂载参数的规范化JSON求SHA-256，排除曝光设置；标为名义模型，挂载参数标为仿真假设。录制schema2在每帧中保留每种曝光的标定身份和时间域；完整qpos/qvel/time及子采样相机姿态保存于单独GT文件，供回放与离线验证，不作为正式感知输入。

## 可复现实验

所有命令在仓库根目录运行，输出目录必须是新的。macOS渲染需要可访问CoreGraphics的终端上下文。

```sh
.venv/bin/python Robot-Perception-Sim/scripts/record_g3_exposure_matrix.py --output Robot-Perception-Sim/results/g3/exposure_matrix_new
.venv/bin/python Robot-Perception-Sim/scripts/verify_g3_exposure.py Robot-Perception-Sim/results/g3/exposure_matrix_new --verify-replay
```

矩阵使用320×200名义针孔相机及半径4cm的红色诊断球，不作为足球动力学或SDK标定实验。实际结果目录为 `results/g3/exposure_matrix_20261025_cgl/`：63组曝光记录、378张PNG、1260个子采样状态。独立验证检查静态图像完全一致，运动方向和拖影长度符合投影预测，曝光图像中心接近中点瞬时图。最大中点中心误差0.2544px，拖影长度误差1.2034px；独立加载XML和NPZ重放的相机姿态最大误差2.22e−16。

原K1录制链路实际运行命令：

```sh
/Users/vae/miniconda3/envs/robot/bin/python Robot-Perception-Sim/scripts/record_stereo.py --preset right --duration 0.1 --fps 60 --output-dir Robot-Perception-Sim/results/g3/k1_recording_new --ffmpeg /Users/vae/miniconda3/envs/robot/bin/ffmpeg
```

最终 `results/g3/k1_recording_final/` 保存6帧、四个视频和111个物理状态，6帧两目均见球，最大调度偏差0.667ms。132个保存的左右子采样姿态独立重放误差为0，代码来源哈希匹配。完整物理尾段覆盖曝光窗口；5帧窗口内有球接触，匀速预测与实际物理末端的最大偏差0.540mm，这说明近似误差确实需要保留，而非删除接触帧。

## SDK实施边界

当前Isaac4.5/官方扩展v3.2不内置已验证的有限曝光或现代Sim2Real模型。给USD设置shutter属性、声明配置fps、或给瞬时图像添加非零窗口，都不足以证明曝光实现。G3适配器以实际多次渲染的RGBA平均实现有限曝光，额外验证合成字节和SDK输出；这不等同厂商原生ISP。

历史瞬时协议的 `verify_g3_sdk_timing.py` 复用G2实际文件、像素ID和成对参考验证；最终统一协议使用 `verify_g3_sdk_composite.py`，将SDK帧关联到曝光窗口、源子采样、两目姿态、标定及接收/保存时间。丢失的作者ID单独列出，不自动称作网络丢帧；作者状态与像素ID相符仍不能代替运动几何的图像检查。

## SDK 发送与接收版本

本轮进一步核对捆绑动态库后，发现扩展发送端实际为SDK5.1.0，旧接收端为5.0.7。此前G2两轮与本轮部分基础帧确实通过，但不能据此把跨版本配对视为完整兼容。[厂商5.1发布说明](https://docs.stereolabs.com/docs/development/zed-sdk/release-notes/5-x/5-1)要求接收端版本至少等于发送端；G3采用独立的匹配版本接收端重新验收，G2历史环境和数据保留。

旧接收端出现的打开失败会记录真实非零退出码，不能用正常退出的发送端或已渲染源图像替代SDK结果。换端口、帧参考起点、同步图执行及网络路径的诊断结果均保留；网络静态图像超过预先固定的8级RGB误差上限，没有作为合格基线。

## 官方SDK中的有限曝光图像

可选G3路径仍调用官方v3.2节点和正式SDK。先在同一状态窗口内渲染左右中点子采样，用第二条像素标记独立确认每个子采样的身份与同数据包参考；再对实际RGBA值等权平均。合成PNG和送入CUDA的字节哈希必须与独立重读的子采样平均完全一致。

每个完成的左右合成图上传到独立CUDA分配，发布后不修改，在官方发送节点销毁前保持存活。这样避免官方异步工作线程只保存裸指针时，后续渲染复用图像内存造成像素与时间不一致。源端的实际像素身份、SDK原始时间以及SDK保存的两目PNG逐帧交叉核对，不能仅凭指针设置成功来判断发送正确。

G3默认帧参考为已声明曝光窗口的中点，使用从1秒开始的精确重放时钟，以满足旧桥接节点的初始化条件。它与保持不动的原生World时间分开记录。诊断中使用过起点参考，验证器只接受该次元数据明确声明的参考，不从观测结果拟合偏移。中点是图像参考时刻，最早因果可用时刻为曝光结束；主机接收与落盘时间则是另一个时间域。

当前实现关闭RTX后处理运动模糊和时域AA，不额外叠加第二套拖影。Replicator的`rt_subframes`在同一状态下重复渲染，不能直接视为运动曝光；本实现对每个曝光子采样显式改变状态并验证输出像素。现代Stereolabs Sim2Real及真实ISP不在该旧扩展中获得实现，因此本轮使用明确声明的编码RGBA平均近似。

正式观察输入仍是接收目录的左右图、SDK深度/置信度、标定和时间字段。源状态、球世界真值以及`joined_frames.jsonl`是离线验收资料，不是可部署感知算法的输入。实际运动采用有参数、可重放的诊断轨迹；不把它当作K1运控或真实球接触模型的验证。

## 独立验收标准

每次正式SDK录制需通过：源代码快照哈希、实际标定文件字节与运行内参/基线、所有已发布窗口中源子采样的左右双标记与投影几何、PNG平均与上传哈希、每个SDK帧的实际两目ID/原始时间/合成图像及接收因果界。独立验证入口为 `scripts/verify_g3_sdk_composite.py`，历史原发送路径的时间检查仍由 `verify_g3_sdk_timing.py` 读取。

静态零曝光先冻结同SDK版本、同传输方式和同标定下的编码/颜色转换基线；其RGB区域平均绝对误差不得超过8级。后续阈值为 `max(3, 1.5×静态最大误差+1)`，禁止跨版本或跨传输方式复用。原IPC/SDK5.0.7基线与新IPC/SDK5.1.0基线分别保存，不把网络路径较大的色差混入合格样本。

有限运动曝光还要求实际图像能与中点瞬时图区分：在投影拖影至少3px的样本中，源合成与瞬时图的平均差至少1级RGB；SDK图像相对合成图的误差，平均至少比相对瞬时图低0.25级，且至少75%的两目帧有改善。这些标准在正式运动曝光数据检查前固定。每个瞬时子采样的实际球像素需与保存位姿的投影相差不超过3px，SDK曝光图与实际源合成图的球中心相差不超过2px；不把多个投影坐标的算术平均当成曝光图像中心。

## 最终采集路径与运行边界

当前G3统一先采集真实的成对子采样和状态，再把完成的图像以稳定主机节奏送入官方SDK。零曝光是一点采样，10ms曝光采用3点中点求积；所有子采样、平均图和发布记录均保留。科学采样时刻仍由有理数窗口确定，主机回放配置为10Hz，两者不是同一个帧率。该路径实际输出仿真曝光图像，不是给瞬时图像添加窗口标签。

逐窗口慢渲染直接发送曾多次在SDK打开阶段失败；在匹配SDK、标定及足够显存后，预采集加稳定回放路径通过了实际SDK与图像验收。这是本轮接受的离线采集方式，不宣称定位了SDK内部全部失败原因，也不宣称60/120fps实时吞吐。产品代码没有保留失败的私有执行图、CUDA上下文改写或诊断注入。

SDK运动用例是可回放的正弦诊断轨迹：零曝光球振幅0.3m、相机振幅0.015m，频率0.5Hz；有限曝光球和相机振幅0.3m、频率2Hz。SDK诊断球半径0.09m、光学深度1.2m，纹理平面2m。这些用于时间/图像验证；实际足球动力学仍由G1单独报告。

G3帧记录分别保留曝光中点和结束时刻。`earliest_available_authored_s`是由窗口结束推导的最早因果界，不是测得的World时间或主机延迟。`Mat.timestamp`仅是矩阵操作时间；本项目以同一次grab的IMAGE时间守卫、实际两目像素和源窗口验证配对，不把它称为独立逐目传感器采集时刻。

宿主仍为Ubuntu20.04、驱动575.57.08；G3私有Ubuntu22用户空间复用只读Isaac4.5与官方扩展v3.2.0，独立解包SDK5.1.0并重新编译原C++接收器。SDK安装包来源为 `https://download.stereolabs.com/zedsdk/5.1.0/cu12/ubuntu22`，SHA-256为 `88c2d1a8a597e14c1748f552e64414e2c22773aea8a0a3c8d88c25df03797b95`。没有替换宿主驱动、SDK或G2私有环境。

后期GPU0/1被现有作业占用约20GB/卡，SDK实际出现CUDA显存不足；经用户明确授权，正式末轮发送器使用物理GPU6、接收器使用物理GPU7，各自可见的CUDA设备都是逻辑0。Vulkan渲染索引和CUDA索引分别配置，并用实际NVML进程记录确认；没有停止其他作业。

以下命令在已准备的目标机独立目录执行，输出名须从未使用，复用端口前须确认前一轮发送/接收都正常退出：

```sh
task_g3=/home/lizhengran/isaacsim-workspace/g3_acquisition
env G3_SOURCE_GPU_UUID=GPU-28d2f1b7-9192-0a8b-4efe-36018c0bd307 \
    G3_RENDER_GPU_INDEX=6 \
    G3_RECEIVER_GPU_UUID=GPU-70a53f3b-dfd4-a41f-d1be-432767f54fb2 \
    bash "$task_g3/run_g3_trial.sh" sender_NEW receiver_NEW ball_motion 12 10 96 3 30520 ipc
```

`static`和`camera_motion`同理；零曝光100帧控制组使用曝光0和足够的预备帧数，例如192。此命令依赖G3目录内的私有SDK、编译接收器、Python模块及 `calibration/SN45626933.conf`。名义标定文件及来源在仓库 `config/cameras/`，接收器使用显式文件/哈希参数和正确的设置目录分隔符；实际复制字节、完整K和0.12m基线均由独立验证器重算。

完整原始数据保留于本地 `results/g3/` 与目标机G3 `runs/`，没有提交大型SDK、PNG、浮点深度或失败核心转储。Git保留源码、配置、测试、轻量验收摘要与证据哈希。

未发布的预备窗口和未被本接收器观测到的发布ID分别计数；后者可能包含启动与停止阶段，不能直接当作网络丢包统计。验证器检查每个已发布源窗口以及每个实际SDK双目帧。60Hz是作者采样时间轴，10Hz是配置的主机回放节奏，实际交付速率另由保存的主机单调时钟计算。

## 最终结果

正式匹配版本的统一路径通过六组录制：

| 场景 | 曝光 / 子采样 | 实际SDK帧 | 最大RGB区域平均绝对误差 | 参考残差最大值 |
|---|---|---:|---:|---:|
| 静态 | 0ms / 1 | 100 | 1.77330 | 0ns |
| 球运动 | 0ms / 1 | 100 | 3.26634 | 0ns |
| 相机运动 | 0ms / 1 | 100 | 3.35271 | 0ns |
| 静态 | 10ms / 3 | 12 | 1.75343 | 0ns |
| 球运动 | 10ms / 3 | 12 | 3.24524 | 0ns |
| 相机运动 | 10ms / 3 | 12 | 3.01352 | 0ns |

新SDK5.1静态基线038最大误差1.77283828，冻结上限3.65925743；上述六组均低于该值。0ns指实际SDK纳秒值相对于已声明发布参考的序列化残差，不是硬件曝光或跨时钟绝对精度。

球/相机运动的有限曝光，SDK相对平均图比相对中点瞬时图的误差分别改善0.843/2.957级，所有高运动两目帧都更接近平均图；源平均图与瞬时图差分别1.141/3.999级。静态及运动中变化的第二条诊断标记也确实呈现平均灰度，防止把未平均的静态图当作曝光成功。三组有限曝光的SDK/实际合成球中心最大差均低于0.355px。

六组录制均使用真实SDK API，发送与接收退出码均0，端口30520连续正常复用，结束后没有本轮GPU进程。实际主机交付约9.87–10.12fps（包含保存开销），与60Hz作者时刻、10Hz回放配置分别记录。

最终控制组为 `uniform_static100_058`、`uniform_ball100_056`、`uniform_camera100_057`；有限曝光为 `fast_finite_static_054`、`fast_finite_ball_052`、`fast_finite_camera_053`，均在 `results/g3/sdk/` 下有对应接收目录与独立报告。全部已发布窗口和SDK输出的检验摘要、6036个有效载荷文件的完整哈希清单身份见 [g3_acceptance.json](g3_acceptance.json)。

针对性185项测试通过，覆盖真实渲染和集成；全子工程非渲染194项通过，两个测试集合有重叠。接收器缺失文件/错误哈希的实际CLI负向检查也通过。原K1录制132个左右子采样姿态重放误差为0，专用矩阵1260个子采样姿态最大误差2.22e−16。

本轮在这个明确的离线仿真模型和固定版本下通过G3 Gate。120Hz图像与200Hz IMU的调度逻辑已测试，但实际SDK120fps或200Hz传感器输出、真实曝光/ISP、完整K1 SDK场景、G1球接触等价及后续阶段均没有由此获得验收。

### 构建与独立SDK文件验收

在已准备的目标机目录中，接收器的实际构建参数为以下值；`cuda_headers`复制自宿主已有CUDA头文件，编译器位于G3私有Ubuntu22用户空间：

```sh
bash "$task_g3/run_g3_in_ubuntu22.sh" cmake -S /poc/receiver_source_51 -B /poc/receiver_build_51 -DZED_SDK_ROOT=/poc/sdk-5.1.0-unpacked -DCUDA_INCLUDE_DIR=/poc/cuda_headers -DCMAKE_BUILD_TYPE=Release
bash "$task_g3/run_g3_in_ubuntu22.sh" cmake --build /poc/receiver_build_51 -j2
```

将新数据目录复制回本地 `results/g3/sdk/` 后，先验证静态零曝光、冻结基线，再验证其他场景（报告名使用新的文件名）：

```sh
.venv/bin/python Robot-Perception-Sim/scripts/verify_g3_sdk_composite.py Robot-Perception-Sim/results/g3/sdk/receiver_BASE --source Robot-Perception-Sim/results/g3/sdk/sender_BASE --frames 100 --report Robot-Perception-Sim/results/g3/base_NEW.json --joined Robot-Perception-Sim/results/g3/base_joined_NEW.jsonl
.venv/bin/python Robot-Perception-Sim/scripts/verify_g3_sdk_composite.py Robot-Perception-Sim/results/g3/sdk/receiver_NEW --source Robot-Perception-Sim/results/g3/sdk/sender_NEW --frames 12 --codec-baseline Robot-Perception-Sim/results/g3/base_NEW.json --report Robot-Perception-Sim/results/g3/finite_NEW.json --joined Robot-Perception-Sim/results/g3/finite_joined_NEW.jsonl
```

当前通过验收的接收器可执行文件SHA-256为 `d0af50fe2ec31208c88c4bfdebe8f7e833e8160cfd257dd162b9ea9eade52abd`，源C++为 `471f3ee1fb9cee1e7e2ccdb636bf52a0e9ef2ac65035f6dcba34110f0d28babe`。这些身份与当前仓库源码、构建目录和启动清单独立核对；大型二进制不提交Git。

最终源码冻结后又运行完整子工程测试：**207 passed**，其中包含真实渲染与集成。JUnit保存在 `results/g3/final_all_tests_junit.xml`，其身份纳入验收摘要。
