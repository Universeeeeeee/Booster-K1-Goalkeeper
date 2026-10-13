# 阶段 C 基础速度控制验证

独立 K1 模型；98 个工况；每个工况重复 2 次。
跌倒 0 次；轨迹重复一致：True；坐标方向验证：True；全部运动后停止恢复：True。

下表仅列同时满足持续方向、平均跟踪和停止恢复判据的已测指令点；未证明点之间的连续范围。力矩饱和单独评估。

| 机身方向 | 2 ms 通过指令 | 1 ms 通过指令 |
|---|---|---|
| 前进 m/s | 0.4, 0.6, 0.8 | 0.4, 0.6, 0.8 |
| 后退 m/s | -0.25, -0.27, -0.28 | -0.27, -0.28 |
| 左移 m/s | 0.15, 0.2, 0.25 | 0.15, 0.2, 0.25 |
| 右移 m/s | -0.15, -0.2, -0.25 | -0.15, -0.2, -0.25 |
| 左转 rad/s | 0.2, 0.4, 0.6, 0.8 | 0.2, 0.4, 0.6, 0.8 |
| 右转 rad/s | -0.2, -0.4, -0.6, -0.8 | -0.2, -0.4, -0.6, -0.8 |

+X 前进、+Y 左移、+Z 正向角速度为俯视逆时针；机身朝向改变时平移指令随之旋转。

| 工况 | dt ms | 完成 | 方向/跟踪/恢复通过 | 停止恢复延迟 s | 最大稳态单关节饱和比例 |
|---|---:|---|---|---:|---:|
| forward_0.05 | 2 | True | False | 0.0 | 0.0% |
| forward_0.1 | 2 | True | False | 0.0 | 42.4% |
| forward_0.15 | 2 | True | False | 0.4 | 68.4% |
| forward_0.2 | 2 | True | False | 0.6 | 90.0% |
| forward_0.3 | 2 | True | False | 0.8 | 93.2% |
| forward_0.4 | 2 | True | True | 0.2 | 95.0% |
| forward_0.6 | 2 | True | True | 0.4 | 43.9% |
| forward_0.8 | 2 | True | True | 0.8 | 9.7% |
| backward_0.05 | 2 | True | False | 0.0 | 0.0% |
| backward_0.1 | 2 | True | False | 0.0 | 0.0% |
| backward_0.15 | 2 | True | False | 0.4 | 0.0% |
| backward_0.2 | 2 | True | False | 1.2 | 0.0% |
| backward_0.25 | 2 | True | True | 1.6 | 32.4% |
| left_0.05 | 2 | True | False | 0.0 | 0.0% |
| left_0.1 | 2 | True | False | 0.0 | 0.0% |
| left_0.15 | 2 | True | True | 0.2 | 69.1% |
| left_0.2 | 2 | True | True | 0.4 | 81.2% |
| left_0.25 | 2 | True | True | 1.6 | 80.8% |
| right_0.05 | 2 | True | False | 0.0 | 0.0% |
| right_0.1 | 2 | True | False | 0.0 | 0.0% |
| right_0.15 | 2 | True | True | 0.2 | 63.0% |
| right_0.2 | 2 | True | True | 0.4 | 82.2% |
| right_0.25 | 2 | True | True | 0.6 | 82.6% |
| turn_left_0.1 | 2 | True | False | 0.0 | 47.9% |
| turn_left_0.2 | 2 | True | True | 0.2 | 84.2% |
| turn_left_0.4 | 2 | True | True | 0.4 | 82.7% |
| turn_left_0.6 | 2 | True | True | 0.8 | 85.7% |
| turn_left_0.8 | 2 | True | True | 0.2 | 89.4% |
| turn_right_0.1 | 2 | True | False | 0.2 | 37.2% |
| turn_right_0.2 | 2 | True | True | 0.0 | 84.5% |
| turn_right_0.4 | 2 | True | True | 1.2 | 79.8% |
| turn_right_0.6 | 2 | True | True | 0.8 | 85.5% |
| turn_right_0.8 | 2 | True | True | 0.2 | 88.1% |
| frame_90_forward | 2 | True | True | 0.2 | 94.7% |
| frame_90_backward | 2 | True | True | 1.6 | 32.2% |
| frame_90_left | 2 | True | True | 0.4 | 82.2% |
| frame_90_right | 2 | True | True | 0.4 | 82.2% |
| frame_90_turn_left | 2 | True | True | 0.4 | 81.8% |
| frame_90_turn_right | 2 | True | True | 1.2 | 79.7% |
| frame_180_forward | 2 | True | True | 0.2 | 95.2% |
| frame_180_backward | 2 | True | False | 1.6 | 34.1% |
| frame_180_left | 2 | True | True | 0.4 | 81.8% |
| frame_180_right | 2 | True | True | 0.4 | 82.8% |
| frame_180_turn_left | 2 | True | True | 0.4 | 81.8% |
| frame_180_turn_right | 2 | True | True | 1.2 | 81.6% |
| switch_forward_back_stop | 2 | True | False | 1.2 | 92.9% |
| switch_lateral_turn_stop | 2 | True | False | 0.8 | 83.5% |
| forward_0.05 | 1 | True | False | 0.0 | 0.0% |
| forward_0.1 | 1 | True | False | 0.0 | 0.0% |
| forward_0.15 | 1 | True | False | 0.4 | 0.0% |
| forward_0.2 | 1 | True | False | 0.8 | 0.0% |
| forward_0.3 | 1 | True | False | 0.6 | 0.0% |
| forward_0.4 | 1 | True | True | 0.2 | 0.0% |
| forward_0.6 | 1 | True | True | 0.6 | 6.0% |
| forward_0.8 | 1 | True | True | 1.0 | 9.0% |
| backward_0.05 | 1 | True | False | 0.0 | 0.0% |
| backward_0.1 | 1 | True | False | 0.0 | 0.0% |
| backward_0.15 | 1 | True | False | 0.4 | 0.0% |
| backward_0.2 | 1 | True | False | 1.2 | 0.0% |
| backward_0.25 | 1 | True | False | 1.2 | 0.0% |
| left_0.05 | 1 | True | False | 0.0 | 0.0% |
| left_0.1 | 1 | True | False | 0.0 | 0.0% |
| left_0.15 | 1 | True | True | 0.6 | 0.0% |
| left_0.2 | 1 | True | True | 0.8 | 0.0% |
| left_0.25 | 1 | True | True | 0.8 | 0.2% |
| right_0.05 | 1 | True | False | 0.0 | 0.0% |
| right_0.1 | 1 | True | False | 0.0 | 0.0% |
| right_0.15 | 1 | True | True | 0.6 | 0.0% |
| right_0.2 | 1 | True | True | 0.6 | 0.0% |
| right_0.25 | 1 | True | True | 0.6 | 0.0% |
| turn_left_0.1 | 1 | True | False | 0.2 | 0.0% |
| turn_left_0.2 | 1 | True | True | 0.0 | 0.0% |
| turn_left_0.4 | 1 | True | True | 0.2 | 0.0% |
| turn_left_0.6 | 1 | True | True | 0.8 | 0.0% |
| turn_left_0.8 | 1 | True | True | 0.6 | 0.0% |
| turn_right_0.1 | 1 | True | False | 0.2 | 0.0% |
| turn_right_0.2 | 1 | True | True | 0.0 | 0.0% |
| turn_right_0.4 | 1 | True | True | 0.2 | 0.0% |
| turn_right_0.6 | 1 | True | True | 0.8 | 0.0% |
| turn_right_0.8 | 1 | True | True | 0.4 | 0.0% |
| frame_90_forward | 1 | True | True | 0.2 | 0.0% |
| frame_90_backward | 1 | True | False | 1.2 | 0.0% |
| frame_90_left | 1 | True | True | 0.8 | 0.0% |
| frame_90_right | 1 | True | True | 0.6 | 0.0% |
| frame_90_turn_left | 1 | True | True | 0.0 | 0.0% |
| frame_90_turn_right | 1 | True | True | 0.2 | 0.0% |
| frame_180_forward | 1 | True | True | 0.2 | 0.0% |
| frame_180_backward | 1 | True | False | 1.2 | 0.0% |
| frame_180_left | 1 | True | True | 0.8 | 0.0% |
| frame_180_right | 1 | True | True | 0.6 | 0.0% |
| frame_180_turn_left | 1 | True | True | 0.2 | 0.0% |
| frame_180_turn_right | 1 | True | True | 0.2 | 0.0% |
| switch_forward_back_stop | 1 | True | True | 1.2 | 5.9% |
| switch_lateral_turn_stop | 1 | True | True | 0.2 | 0.0% |
| backward_0.27 | 2 | True | True | 1.6 | 47.6% |
| backward_0.27 | 1 | True | True | 1.2 | 0.0% |
| backward_0.28 | 2 | True | True | 1.6 | 54.1% |
| backward_0.28 | 1 | True | True | 1.2 | 0.0% |

停止延迟指进入连续一秒低速区间的起点，分辨率0.2 s；确认该状态还需观察后续一秒。
位移、姿态、高度、瞬时RMSE、平均误差、原始PD需求及逐关节饱和比例见 results.json 与各工况 trajectory.npz。
能力限制及守门用途解释见项目 docs/velocity_validation.md；本轮没有执行场景接入或足球碰撞测试。
