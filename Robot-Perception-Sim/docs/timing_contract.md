# G3 时间与曝光契约

G3 复用现有 `StereoCamera`、射门录制器、官方 ZED 发送器和 SDK 接收器。MuJoCo RGB 与真实 SDK 输出分别记录来源；SDK 深度仍来自双目匹配，不能用渲染真值替代。

## 四个时间域

| 时间 | 来源与含义 | 关联方法 |
|---|---|---|
| 仿真状态时间 | MuJoCo `data.time`；Isaac 作者状态的 `World.current_time` | 随保存的状态、位姿记录 |
| 曝光开始/结束/中点 | 两目共享同一个仿真时间窗 | 有限曝光保存全部中点子采样时刻；瞬时曝光三者相等 |
| SDK IMAGE 时间 | 官方成对渲染参考域的纳秒值 | 保留原值，通过同数据包参考和实际两目像素ID关联作者状态 |
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

当前Isaac4.5/官方扩展v3.2不内置已验证的有限曝光或现代Sim2Real模型。给USD设置shutter属性、声明配置fps、或给瞬时图像添加非零窗口，都不足以证明曝光实现。SDK瞬时诊断只允许零宽共享窗口；非零曝光需要额外的实际像素与来源验证。

`verify_g3_sdk_timing.py`复用G2实际文件、像素ID和成对参考验证，将每个SDK帧关联到作者状态、两目相机姿态、球位、标定版本及接收/保存单调时间。丢失的作者ID单独列出，不自动称作网络丢帧；作者状态与像素ID相符仍不能代替运动几何的图像检查。
