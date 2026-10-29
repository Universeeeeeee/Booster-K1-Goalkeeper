# G5 参考 YOLO 来源审计

2026-10-29 只读核查 `pc@100.75.243.43`，复用现有模型和后处理执行推理，不重新训练。

## 固定来源

- 工程：`/home/pc/zean.xi/stereo_ir_ball_v1`，Git HEAD `bc964a6bd44f82c5e97fb1a185807e6bbb60d1f0`。推理依赖工作区文件，HEAD 本身不证明工作区无改动。
- 调用链：`/home/pc/ssr/app/zean_ball_backend.py` 显式指定 `runs/full_v2/weights/best.pt`；本轮沿用该路径。`ball/deployment.py` 的另一默认发布权重并不是该调用链默认使用的文件。
- 权重 SHA-256：`9723db63ec8e2a0588030f925755419a55719e8a06c6360d95d80925a713c107`。运行时重新核验，不接受静默换权重。
- `FULL_TRAINING.md` 记录单通道 YOLO11n、来源 `full35_dataset_v1`、几何伪标签和时间隔离划分；这些是源工程记录，未重新审计原始训练数据或实机精度。
- 使用原 `ball.detector.YoloDetector` 和 `ball.boxes.detection_from_box`；置信度 0.25、NMS IoU 0.7、最多 20 个框，原实现进行 stride padding、归一化和框截断。
- 参考环境：PyTorch `2.12.0.dev20260407+cu128`、OpenCV `4.11.0`、Ultralytics `8.4.140`。本次强制 CPU，两线程，不占用参考机 GPU。

## 许可证与输入边界

已安装 Ultralytics 包元数据标明 `AGPL-3.0`。参考工程顶层没有找到针对自定义权重或代码的独立许可证；第三方 BlurBall 的许可证不能覆盖整个工程。因此本仓库不复制发布参考模型、训练数据或源码。比较脚本通过 SSH 将图像传入原环境，调用已有模型并收回检测结果。

源模型针对 D455 红外图像训练。G5 输入是 ZED 仿真 RGB 转灰度，必须作为跨域比较报告，不能称为真实红外成像或通用 RGB 检测能力。没有使用源 D455 的内参进行 ZED 三角化。检测过程只接收图像和帧编号，真值标签不发送给推理端。

复现入口：`scripts/compare_reference_yolo.py --capture <SDK接收目录> --output <新JSON路径>`。需要可访问上述参考机及既有 Python 环境。脚本保留所有候选框；不能把存在任意候选框直接算作命中足球。

## 本轮执行状态

来源、权重哈希、代码和环境查询成功后，参考机 Tailscale 离线，SSH 连续超时；没有得到本轮 YOLO 推理结果。用户明确回复“暂时连不上，先做别的”，因此暂缓该对照。匹配评测脚本 `scripts/evaluate_reference_yolo.py` 按独立解析球轮廓计算 IoU，以 0.5 为命中门槛，分别统计误检、漏检和重复候选；相关测试通过不代表真实 YOLO 精度已测量。
