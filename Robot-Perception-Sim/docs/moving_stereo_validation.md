# 运动足球双目采集验证

日期：2026-10-21（Asia/Shanghai）。新增内容仅在 `Robot-Perception-Sim/`。当前完成 P2 的最小物理射门采集，不包含规则轨迹、CSV 回放或图像感知算法。

## 实现与复用

直接调用原有 `preset_shots` 和 `run_impact`：机器人先通过已有 `SceneSimulation/k1_walk` 以零速度指令站立 2 秒，再执行已有射门。射门初始化、1 ms 物理推进、策略更新、接触与轨迹记录均由原实现负责。一次实验只模拟一份物理轨迹，两种曝光共享完全相同的起点状态。

新增 `recording.py` 按采样时刻选择已记录状态，恢复 qpos/qvel 后执行 `mj_forward`，再调用既有 `StereoCamera.capture`。每帧分别生成 0 ms 和配置曝光（当前 10 ms）的左右 RGB。曝光使用既有匀速运动学近似，未修改控制器、机器人资产或物理参数。

视频使用已有 FFmpeg H.264 编码。未新增 Python 依赖、模型副本或站立控制实现。额外保存的 segmentation 可见像素数仅用于诊断，不参与 RGB 图像生成或检测算法。

## 运行方法

激活已有 `robot` 环境，在仓库根目录执行：

```bash
python Robot-Perception-Sim/scripts/record_stereo.py --preset right --duration 1 --output-dir Robot-Perception-Sim/results/moving_stereo
python Robot-Perception-Sim/scripts/record_stereo.py --preset post --duration 1 --output-dir Robot-Perception-Sim/results/moving_stereo_post
```

实际验证使用 `/Users/vae/miniconda3/envs/robot/bin/python`，包含 MuJoCo 3.10.0、NumPy 2.5.0、Pillow 12.3.0、Torch 2.9.1，以及同目录 FFmpeg 8.0.1。FFmpeg 自动从 PATH 或 Python 同目录发现；使用其他 Python 时可通过 `--ffmpeg` 指向现有可执行文件。机器人官方策略权重直接引用原项目资产。

`--preset` 复用已有全部射门预设；支持 `--duration`、`--seed`、`--fps`、`--camera-config`。默认 60 fps，10 ms 曝光不接受 120 fps。输出必须位于本子目录的 `results/` 且为空，防止覆盖既有记录。再次执行时选择新的输出目录。

## 输出与时间定义

每组实验保存：

- `left_0ms.mp4`、`right_0ms.mp4`、`left_exposed.mp4`、`right_exposed.mp4`：用于浏览的四段视频，exposed 对应 metadata 中的配置曝光。
- `0ms/left/`、`0ms/right/`、`exposed/left/`、`exposed/right/`：无损 RGB PNG，每帧文件名为连续六位帧号。
- `frames.jsonl`：目标时间、实际物理状态时间、左右同步时间、相对射门时间、采样偏差、曝光结束时间、图像路径和真值索引。
- `ground_truth.npz`：逐帧球世界位置/速度、两目相机世界位姿，以及曝光误差验证所需的 1 ms 物理球状态序列和接触记录。普通 NumPy 即可读取，无需加载 MuJoCo。
- `metadata.json`：帧数、分辨率、K、曝光、软件版本、权重/模型指纹、坐标定义、种子和射门参数。
- `simulation.json`：原物理记录器返回的站立/跌倒/数值状态、接触事件和整份物理轨迹哈希。
- `diagnostics.json`：逐帧足球可见像素、曝光期间球接触标记、匀速预测的区间末端误差及汇总。

所有时间单位为 s；`timestamp_s` 是实际状态对应的曝光开始时刻，两目严格相同。0 ms 与 10 ms 共用起点；10 ms 的曝光区间向后延伸，因此运动物体的图像中心也会发生自然位移。

60 Hz 的目标时间为 `launch_time + frame_id/60`，取不早于目标时刻的首个物理状态。1 ms 物理步长对应交替的 16/17 ms 帧间隔；没有把每 17 步误称为 60 fps。实际误差记录在 JSONL，视频按标称 60 fps 的恒定帧率播放，浏览器视频时间轴不表达这一级采样抖动。PNG/JSONL 是后续分析依据，MP4 为有损浏览副本。

逐帧 `T_world_from_optical` 的数组形状是 N×2×4×4，第二维顺序为 left、right。矩阵作用于列向量；光学坐标 X 右/Y 下/Z 前，世界 Z 向上，单位 m。位姿对应曝光开始，不把它当作整个模糊曝光中的唯一精确位姿。

## 本机结果

两组实验均为 960×600、60 fps、射门后 1 秒。每组 60 帧，两种曝光各两目，共 240 张 PNG 和四段 MP4；两组合计 480 张 PNG、八段视频。机器人均完成物理运行，无跌倒、数值错误或策略停止，原记录器的 `balance_maintained` 为 true。

| 指标 | right：滚动射门 | post：门柱射门 |
|---|---:|---:|
| 两目均可见的起点帧数 | 20 / 60 | 12 / 60 |
| 两目均不可见的起点帧数 | 40 / 60 | 46 / 60 |
| 曝光期间存在球接触的帧数 | 59 | 28 |
| 最大采样偏差 | 0.667 ms | 0.667 ms |
| 匀速曝光末端最大位置偏差 | 0.540 mm | 50.525 mm |
| 原物理记录的球接触类型 | ground | goal、ground |

post 另有两帧只在单目可见。可见性由曝光起点的真实 segmentation 像素计算，不代表整段曝光持续可见，也不是图像检测成绩。right 的首 20 帧可观察到足球，随后自然离开视野，覆盖了正常观察和丢失视野的情况。

末端误差定义为 `||p_actual(t+10ms) - (p(t)+v(t)*10ms)||`。对齐实际物理状态时间，并记录参考结束时间。该值用于评估匀速运动近似，不是三维感知误差，也不是图像拖影像素误差。

post 的最大误差出现在帧 14 的门柱接触区间，此时足球已离开两目视野。结果说明接触时不能信赖匀速曝光近似，但不能据此宣称当前画面出现了 5 cm 的可见定位误差。需要研究接触区间的真实成像时，应使用物理历史状态做曝光积分，而非继续匀速外推；该改进本轮未实现。

## 验证

```bash
.venv/bin/python -m pytest -q -c Robot-Perception-Sim/pytest.ini Robot-Perception-Sim/tests --junitxml=Robot-Perception-Sim/results/tests/junit.xml
```

38 项专项测试通过，0 失败、0 跳过。新增八项覆盖：60 Hz 的 16/17 ms 调度无长期漂移、非法时间/帧率、超出轨迹区间、重复物理帧和缺少编码器的明确错误。根目录已验证环境执行数学和渲染回归；完整运动采集另外在用户现有 `robot` 环境执行。

独立检查已重新打开全部 480 张 PNG，并核对逐帧时间、两目同步和 NPZ 位姿/真值索引。FFprobe 对八段 MP4 均读得 H.264、960×600、60 帧、60/1 fps、1.000 s；FFmpeg 完整解码均成功。视频编码器的成功返回值之外，验证了实际交付文件。

重新执行 right 的物理过程，原记录器返回的完整轨迹 SHA-256 完全一致。子目录外源文件哈希及已有 Git 差异保持一致。原始验证记录位于 `results/audit/recording_verification.json`。

## 查看对照

[左目 0 ms 视频](../results/moving_stereo/left_0ms.mp4) · [左目 10 ms 视频](../results/moving_stereo/left_exposed.mp4) · [右目 10 ms 视频](../results/moving_stereo/right_exposed.mp4)

right 的第 15 帧，约发射后 250 ms：

| 0 ms | 10 ms |
|---|---|
| ![左目瞬时图](../results/moving_stereo/0ms/left/000015.png) | ![左目曝光图](../results/moving_stereo/exposed/left/000015.png) |

生成输出默认不进入 Git，可使用上述命令重建。本轮仍为离线渲染，不宣称实时 60 fps、真实相机端到端延迟或 ZED SDK 性能。
