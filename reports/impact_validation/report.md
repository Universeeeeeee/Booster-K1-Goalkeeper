# 阶段 E 足球碰撞与抗扰验证

69 个工况、138 次执行；重复一致：True；数值异常：0；自由机器人跌倒工况：1。

1ms物理、50Hz官方策略、每物理步PD；球半径0.11m、质量0.43kg。速度为实验选点，不代表FIFA规定。
发射前静止2秒；横移组再执行2秒vy=+0.2，并在发射后1秒切换零指令。发射后观察6秒；跌倒时保存并提前停止。
固定模式为原零关节姿态的刚体隔离，不能用于同姿态移除控制器的因果比较。无机器人组使用对应自由组相同初始球位置、速度、旋转。

| 工况 | 有效接触 | 轨迹改变 | 维持/恢复平衡 | 跌倒 | 任一门/右门进球 | 有效阻挡 | 法向冲量Ns | 峰值力N |
|---|---|---|---|---|---|---|---:|---:|
| front_2 | True | True | True/True | False | False/False | True | 1.152 | 282.3 |
| front_4 | True | True | True/True | False | False/False | True | 2.340 | 602.3 |
| front_6 | True | True | True/True | False | False/False | True | 3.534 | 928.6 |
| front_8 | True | True | True/True | False | False/False | True | 4.506 | 1068.5 |
| front_12 | True | True | True/True | False | False/False | True | 7.123 | 1912.2 |
| front_16 | True | True | True/True | False | False/False | True | 8.907 | 2037.0 |
| front_20 | True | True | False/False | True | None/None | None | 11.911 | 3225.9 |
| eccentric_-1_2 | True | True | True/True | False | False/False | True | 1.125 | 267.2 |
| eccentric_-1_4 | True | True | True/True | False | False/False | True | 2.296 | 575.7 |
| eccentric_-1_8 | True | True | True/True | False | False/False | True | 4.447 | 1038.2 |
| eccentric_+1_2 | True | True | True/True | False | False/False | True | 1.159 | 298.2 |
| eccentric_+1_4 | True | True | True/True | False | False/False | True | 2.337 | 613.5 |
| eccentric_+1_8 | True | True | True/True | False | False/False | True | 4.470 | 1060.5 |
| left_foot_2 | True | True | True/True | False | False/False | True | 1.657 | 194.5 |
| left_foot_4 | True | True | True/True | False | True/True | False | 2.756 | 432.8 |
| left_foot_8 | True | True | True/True | False | True/True | False | 5.504 | 916.8 |
| left_shank_2 | True | True | True/True | False | False/False | True | 1.428 | 267.7 |
| left_shank_4 | True | True | True/True | False | False/False | True | 2.258 | 490.8 |
| left_shank_8 | True | True | True/True | False | False/False | True | 4.494 | 964.8 |
| right_foot_2 | True | True | True/True | False | False/False | True | 1.681 | 195.5 |
| right_foot_4 | True | True | True/True | False | True/True | False | 2.736 | 434.3 |
| right_foot_8 | True | True | True/True | False | True/True | False | 5.528 | 918.3 |
| right_shank_2 | True | True | True/True | False | False/False | True | 1.428 | 265.8 |
| right_shank_4 | True | True | True/True | False | False/False | True | 2.257 | 488.7 |
| right_shank_8 | True | True | True/True | False | False/False | True | 4.492 | 964.1 |
| moving_2 | True | True | True/True | False | False/False | True | 1.144 | 308.6 |
| moving_4 | True | True | True/True | False | False/False | True | 2.206 | 513.8 |
| moving_8 | True | True | True/True | False | False/False | True | 4.624 | 1176.9 |
| head_2 | True | True | True/True | False | False/False | True | 1.009 | 267.1 |
| left_forearm_2 | True | True | True/True | False | False/False | True | 1.174 | 98.7 |
| right_forearm_2 | True | True | True/True | False | False/False | True | 1.188 | 95.4 |
| miss_4 | False | False | True/False | False | True/True | False | 0.000 | 0.0 |
| moving_miss_4 | False | False | True/False | False | True/True | False | 0.000 | 0.0 |
| fixed_front_2 | True | True | None/None | False | False/False | True | 1.147 | 258.4 |
| fixed_front_8 | True | True | None/None | False | False/False | True | 4.572 | 1012.7 |
| fixed_front_20 | True | True | None/None | False | True/False | True | 11.946 | 2963.2 |

None为不适用/观察被提前截断，不能当作成功阻挡。有效阻挡仅为6秒观察窗口中，相同发射的absent对照进球，而有机器人工况发生接触且右门未进球。左门进球仍由原判定器记录，不计为右门失守。
力从每次求解后的mj_contactForce读取，接触区间是该物理步开始到结束；ΣFdt积分为冲量。
每步球动量变化与全部球接触力+重力独立核对；严格1e-6Ns精度门槛在6个工况中未满足，最大2.313e-05Ns。独立底层核对确认来自原Newton求解残差，接触力重建及Euler积分相符，参数未调整。详见solver_audit.json。
详细接触部位、时刻、持续时间、速度变化、姿态、恢复和限幅见各summary.json及trajectory.npz。
覆盖和警告来源见docs/collision_audit.md；能力限制见docs/impact_validation.md。
