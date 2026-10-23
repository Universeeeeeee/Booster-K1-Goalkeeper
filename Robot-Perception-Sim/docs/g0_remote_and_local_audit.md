# G0a：提交、本地接口与回归审计

日期：2026-10-24（Asia/Shanghai）。依据用户提供的 `Booster_K1_感知平台_中长期执行路线图_v2.md`，本文件只交付 G0a；不执行 G1、G3 及之后的开发。G2 独立 SDK PoC 由另一工作单元负责，本报告不宣称 SDK 已接通。

## 结论与基线

曝光模拟、运动双目录制和此前计划文档都已存在于当前分支。主任务完成推送后核验远端 SHA 为 `821cd1e2f3b485da14735e1eaa0537cd12ca05d7`；本审计独立读取本地 HEAD 和 `origin/feat/perception-p0-p1`，两者也均为该 SHA。远端网络核验来自主任务，本审计没有以本地 tracking ref 冒充第二次远端查询。v2 源文档中远端仅可见 `fae8848` 的记录已被本次推送证据更新。

| 内容 | 完整提交 SHA | 当前状态 |
|---|---|---|
| 10 ms 双目曝光 | `d81b04bf6bb24aa7d2edb6a9813e2bd59c9835d8` | 已在 HEAD 历史中；对应源码与测试无未提交差异 |
| 运动射门双目录制 | `a952d73acf8be260ae5677cee03a2117c73a17b7` | 已在 HEAD 历史中；对应源码与测试无未提交差异 |
| Isaac/ZED 调查与执行计划 | `821cd1e2f3b485da14735e1eaa0537cd12ca05d7` | 已推送；是计划和调查，不是 Isaac/SDK 运行成绩 |

父仓库存在大量已有未提交差异。本审计未清理、回退、提交或推送这些差异；未以 `long_term_perception_roadmap.md` 草稿作为规范。业务目录本轮仅新增本报告与未纳入 Git 的 [local_audit.json](../results/g0/local_audit.json)。该 JSON 保存原始测试输出、命令、版本、文件哈希、CLI 输出和旧数据独立读回结果。

## 两条功能提交的文件对照

以下路径相对 `Robot-Perception-Sim/`，以 `git show --name-only` 读取提交实际文件列表。

- 曝光提交（9 个文件）：`README.md`、`config/cameras/zedx_wide.json`、`docs/mac_validation.md`、`docs/plan.md`、`robot_perception_sim/camera.py`、`robot_perception_sim/stereo.py`、`scripts/preview_stereo.py`、`tests/test_exposure.py`、`tests/test_stereo_camera.py`。
- 录制提交（7 个文件）：`README.md`、`docs/moving_stereo_validation.md`、`docs/plan.md`、`docs/sources.md`、`robot_perception_sim/recording.py`、`scripts/record_stereo.py`、`tests/test_recording.py`。
- 计划提交（2 个文件）：`docs/isaac_zed_execution_plan.md`、`docs/isaac_zed_research_notes.md`。

## 本轮真实运行与隔离方式

本机环境：macOS 26.4 arm64、Python 3.13.13、MuJoCo 3.3.7、NumPy 2.3.3、Torch 2.9.0、pytest 8.4.2、Pillow 11.3.0。它不是目标 Linux 服务器，也不同于历史完整视频录制使用的 conda `robot` 环境。

原相机测试会直接写 `results/tests/scene.xml`、`original_scene.xml`、投影 PNG/JSON 和曝光 PNG。为保护历史结果，逐字节复制现有 `robot_perception_sim/`、`tests/`、`scripts/`、`config/` 与 pytest 配置到 `/private/tmp/booster_g0_b4v2bi99/Robot-Perception-Sim/`，模型/许可证/父工程依赖只通过读取链接引用；没有改生产或测试源码。复制文件的 SHA-256 均与原文件相同。关闭 pytest cache 和 Python bytecode，测试工作目录也设在临时目录，MuJoCo 日志不会写入父仓库。

| 执行 | 实际结果 | 解释 |
|---|---|---|
| 全部相机测试，沙箱内 | **32 passed、6 failed、0 skipped** | 6 项都在创建真实 OpenGL context 时报 `CGLError: invalid CoreGraphics connection`；保留失败原始输出，没有改成 skip |
| 同副本的 6 项 render，正常 macOS 图形连接 | **6 passed、32 deselected、0 failed、0 skipped** | 经工具许可在沙箱外运行原测试，真实生成 PNG；未安装或修改环境 |
| 既有 `test_walk_controller.py` + `test_scene_control.py` | **39 passed、0 failed、0 skipped** | 真实官方 TorchScript 推理、物理推进、名称映射、PD、时序、reset/命令历史和正向 body-Y 横移恢复用例 |
| 两个 CLI 的 `--help` | **2 次退出码 0** | 证明入口可加载及参数存在，不等于完整视频录制通过 |
| 既有 right/post 录像记录读回 | **各 60 帧核验通过** | 左右时间相等、严格递增、NPZ 时间逐项相同、图像路径存在；不是重新采集 |

因此，38 项现有相机用例均已在其适用执行环境覆盖并通过；不是声称首次整套执行零失败。真实渲染包括静态曝光一致、横向运动球模糊、完整场景双目可见性和三组头部角度投影；24 项独立像素测量的最大重投影误差为 **0.399041 px**。临时渲染产物路径和 SHA-256 记录在 JSON，已实际查看曝光图像。

保护检查覆盖原 `results/tests/` 全部文件及父目录 `MUJOCO_LOG.TXT`，共 **35 个文件，前后哈希全部一致**。这是一组明确列出的保护检查，不宣称对父仓库所有大文件进行了全量哈希复验。业务源码/测试/脚本/相机配置目录的 `git status --short` 为空。

实际命令（工作目录 `/private/tmp/booster_g0_b4v2bi99`；均设置 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`）：

```sh
/Users/vae/Projects/booster/.venv/bin/python -B -m pytest -q -ra -p no:cacheprovider \
  -c /private/tmp/booster_g0_b4v2bi99/Robot-Perception-Sim/pytest.ini \
  /private/tmp/booster_g0_b4v2bi99/Robot-Perception-Sim/tests

# 使用正常 macOS CoreGraphics 连接，原样复跑六项渲染测试：
/Users/vae/Projects/booster/.venv/bin/python -B -m pytest -q -ra -p no:cacheprovider \
  -c /private/tmp/booster_g0_b4v2bi99/Robot-Perception-Sim/pytest.ini \
  /private/tmp/booster_g0_b4v2bi99/Robot-Perception-Sim/tests -m render

/Users/vae/Projects/booster/.venv/bin/python -B -m pytest -q -ra -p no:cacheprovider \
  -c /Users/vae/Projects/booster/pytest.ini \
  /Users/vae/Projects/booster/tests/test_walk_controller.py \
  /Users/vae/Projects/booster/tests/test_scene_control.py
```

临时目录会被系统清理。以后复验应重建相同的源码/测试隔离副本并核对文件哈希，不能直接运行默认测试路径覆盖旧 `results/tests`。JSON 保存完整输出，报告保留关键结果。

## CLI、输出与时间语义

两个入口已实际执行 `--help`，支持以下接口：

| 入口 | 参数与行为 |
|---|---|
| `scripts/preview_stereo.py` | `--output-dir`、`--camera-config`、`--scene-config`、`--head-yaw`、`--head-pitch`；静态预览，不证明动态站立 |
| `scripts/record_stereo.py` | `--preset`、`--duration`、`--seed`、`--fps`、`--camera-config`、`--output-dir`、`--ffmpeg`；默认 right、发射后 1 s、60 fps；复用既有 10 个射门预设 |

完整录制可复验命令（从仓库根目录执行，必须使用新的空输出目录；**本审计未执行以下录制命令**）：

```sh
.venv/bin/python Robot-Perception-Sim/scripts/record_stereo.py \
  --preset right --duration 1 --fps 60 \
  --ffmpeg /Users/vae/miniconda3/envs/robot/bin/ffmpeg \
  --output-dir Robot-Perception-Sim/results/g0_recording_recheck_right
```

本轮 `.venv` 的 PATH 和 Python 同目录均未发现 FFmpeg；历史 `/Users/vae/miniconda3/envs/robot/bin/ffmpeg` 文件仍存在，但未在本审计执行。完整 MP4 编码/解码与重新录制未复跑，历史成功不能替代本轮结果。现有失败入口测试确认找不到编码器时明确报错。

实现审计：

- `StereoCamera.capture` 默认 10 ms 全局曝光、10 个等权中点子样本；复制 `MjData`，以当前 `qvel` 通过 `mj_integratePos` 外推姿态，分别渲染同一子样本的左右图，再平均 RGB。它不推进调用者物理时间，不修改其 qpos/qvel/qacc/ctrl；本轮真实渲染测试已验证静态一致和匀速运动方向/扩展量。
- `record_shot` 复用 `run_impact/SceneSimulation/k1_walk`，先零速度站立 2 s，只模拟一次物理轨迹，之后离线恢复状态，生成 0 ms/配置曝光的双目图。记录器本身没有横移指令入口。
- 60 Hz 目标采样为 `launch_time + frame_id/60`；取不早于目标时刻的首个 1 ms 状态，16/17 ms 交替，不用固定 17 ms 冒充 60 Hz。采样测试检查误差小于 1 ms、无累积漂移、无重复状态。
- `timestamp_s`、左右时间、相机位姿均对应曝光开始；10 ms 图向后覆盖曝光区间。0/10 ms 共享开始时间而非有效曝光中点，因此画面差异包含时间偏移，不全是模糊。
- 输出有四路 MP4、原始 PNG、`frames.jsonl`、`ground_truth.npz`、`metadata.json`、`simulation.json`、`diagnostics.json`。MP4 是名义恒帧率有损浏览副本，分析应使用 PNG/JSONL；segmentation 只作独立可见性诊断。
- 本轮独立读回两组历史记录，采样最大误差均 **0.666667 ms**，两目时间逐项相同，NPZ 时间匹配，相机位姿形状均为 `60×2×4×4`。这些是旧数据结构和同步证据，不是 Linux 或 SDK 运行结果。

## 已实现、缺失、未执行矩阵

| 能力 | 状态与边界 |
|---|---|
| 理想双目标定/投影/刚性挂载 | 已实现，本轮数学、模型及真实渲染检查通过 |
| 静态与匀速曝光 | 已实现，本轮真实渲染通过；采用输出 RGB 平均，不是标定的线性辐射/ISP模型 |
| 接触期间真实物理曝光积分 | 缺失；仍恒 qvel 外推，不模拟曝光内重力/加速度/碰撞响应 |
| 运动双目录制调度与数据文件 | 已实现；本轮现有单元测试和旧记录独立读回通过；完整录制/编码未重跑 |
| 官方 walk 适配与基础场景控制 | 现有 39 测试通过；本轮没有重新运行历史 A–H 全部实验或 30 秒正式矩阵 |
| 真实 SDK 双目深度、SDK 时间与实时吞吐 | 本轮 G0a 未实现/未验证，由 G2 独立 PoC 负责；不可用 MuJoCo RGB 或渲染 GT 代替 |
| 横移中的感知、足球检测、CSV 回放 | 当前记录器没有这些功能；本轮不补建 |

本地 G0a 达到“已存在能力有复验命令和明确边界”的审计目标。首次渲染失败已定位为该执行上下文的 CoreGraphics 连接限制，正常图形连接下通过；这不证明目标 Linux 图形环境可用。下一步仅按主任务执行 G0b 和独立 G2 的连续 100 帧 SDK PoC，达到或记录明确阻断后停止，不借此进入完整 K1 迁移。
