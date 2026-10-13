# 阶段 H1：受控上肢目标扰动验证

H1冻结版v1完整通过：3个幅度（0.05/0.15/0.30rad）×4个动作家族×2次，共24个非零trial；另有原零指令baseline2次和零残差wrapper2次。所有非零trial满足稳定、实际关节响应、返回后恢复及完整trace重复一致门槛。24trial在t≥3s的最低base z为0.541389m，最大倾角6.523937°，无新增自碰撞/非足地面/robot-ball或goal接触保护、跌倒或数值异常。

结论只支持已测试的零速度、自由基座、无球碰撞、慢升降小幅肩部残差。此结果没有证明上肢与步行独立，也没有证明侧移抬臂、实机安全、高球覆盖或守门成功。

## 冻结协议与真实控制

先落盘并冻结 [protocol.json](/path/to/booster/reports/defense_validation/protocol.json)，才开始正式运行。已有原A–G/官方/配置等840个受保护文件与root protected_manifest逐项SHA-256一致；原官方manifest亦匹配。仅新增H模块和新结果目录。

每例静置3s；quintic `10u³−15u⁴+6u⁵`在2s内升到残差、保持2s、2s返回，随后5s恢复，总计14s。0.05rad先门槛通过后才0.15，再全部通过后才0.30。三阶段峰值残差slew分别0.046875/0.140625/0.28125rad/s。

控制是在每实例原 `compute_targets()`（官方20ms推理、补偿及滤波）之后clone22维目标，仅指定肩关节加残差，再交原1ms PD。原history、last_action、内部filter、KP/KD/effort、模型碰撞及重力均未修改；未设固定基座/外力/执行时机器人qpos强制赋值。E原 `run_impact` 负责场景物理及原记录；其t2发射被指定为远处零速度球，无机器人接触。

动作符号由active任务默认姿态的原生FK交叉核验：left_roll为左肩roll正偏置，right_roll为右肩roll负偏置；both_roll组合两者；both_pitch同时给左右肩pitch负偏置。该符号说明是关节坐标，不能直接当作世界球门左右。

原零指令与零残差wrapper×2逐数组完全一致，比较项包含qpos/qvel、motor控制/实际执行器力矩、每步名义/执行目标与真实22关节状态，以及50Hz policy动作、10×69 history和filter；所有非零重复也逐数组完全一致。详细 [results.json](/path/to/booster/reports/defense_validation/results.json) 与各 `trajectory.npz` 是动态证据。

## 各幅度与家族的实际运动

响应是每个选中关节在hold最后1s、对相同物理时刻匹配零baseline的有符号median Δq；要求每关节至少为残差幅度50%。多关节行按左/右排序。比例超过100%表示官方策略和全身动力学共同响应，不表示残差独立决定姿态。表中数值采用r0，r1全trace逐项相同。

|残差rad|家族|hold末1s有符号median Δq rad|响应/残差|postsettle最低z m|最大倾角°|t9后恢复延迟s|重复|
|---:|---|---|---|---:|---:|---:|---|
|0.05|left_roll|0.066812|133.62%|0.550922|0.391921|0.000|逐数组完全一致|
|0.05|right_roll|0.065940|131.88%|0.550928|0.391921|0.000|逐数组完全一致|
|0.05|both_roll|0.058776/0.057483|117.55%/114.97%|0.550932|0.391921|0.000|逐数组完全一致|
|0.05|both_pitch|0.071298/0.071496|142.60%/142.99%|0.550897|0.868070|0.000|逐数组完全一致|
|0.15|left_roll|0.197379|131.59%|0.550899|0.391921|0.000|逐数组完全一致|
|0.15|right_roll|0.199266|132.84%|0.550858|0.391921|0.000|逐数组完全一致|
|0.15|both_roll|0.177763/0.174127|118.51%/116.08%|0.550932|0.736326|0.000|逐数组完全一致|
|0.15|both_pitch|0.228071/0.227399|152.05%/151.60%|0.548536|3.484057|0.000|逐数组完全一致|
|0.30|left_roll|0.383957|127.99%|0.550714|0.937677|0.000|逐数组完全一致|
|0.30|right_roll|0.392831|130.94%|0.550579|0.954775|0.000|逐数组完全一致|
|0.30|both_roll|0.356369/0.351665|118.79%/117.22%|0.550835|1.789417|0.000|逐数组完全一致|
|0.30|both_pitch|0.474054/0.459245|158.02%/153.08%|0.541389|6.523937|0.000|逐数组完全一致|

所有试验t≥3s的臂raw PD绝对需求最大0.952112Nm；每关节达到14Nm限幅的最大采样比例0.000000%。原software/actuator力矩限制未放宽。

## 平衡、恢复及失败分类

每物理步检查：t≥3s后base z≥0.42m、倾角≤15°；同物理时刻对照baseline的新增正法向力self/non-foot-ground接触；任何robot-ball/goal正力接触；臂raw PD连续饱和不超过0.10s；实际control/torque不超原effort+1e−5Nm。非选slot必须与同次推理名义目标一致。只有选中且非零残差臂slot检查XML内缩0.05rad目标范围，并在PD前拒绝越界目标；相应真实q检查XML角度范围。没有裁剪或改动官方腿目标。

恢复不是沿用E从t2的通用指标，而是独立调用原 `recovery_metrics` 检查返回结束t9至t14：必须有完整1s慢速0.2s bins，末2s平均vx/vy/yaw≤0.05/0.05/0.1，末2s base z≥0.45m、倾角≤10°。28runs均恢复通过。

实际响应不足独立标作authority失败；稳定但authority不足只阻断更大幅度，允许完成同幅度其它家族。真实平衡/碰撞/连续饱和或恢复失败只重复同工况核对后停止其余动作及全部升级。当前没有这类失败，H1接口门槛通过；H3/H4由根任务继续，不能把本报告的无球动作通过转写为挡球成功。

## v1证据与诊断修复版本

正式v1完成后修复两处诊断/门控表达：非有限observer stop须属于numerical失败，不能和正常保护中止一起清空numerical_error；zero parity显式要求两个wrapper summary passed。原v1的24trial没有任何guard触发，两个zero-wrapper summary确实passed，故这些修复不改变既有动态结论或阈值。

v1的protocol/results/traces原字节保留；三源码完整快照位于 [h1_v1_sources](/path/to/booster/reports/defense_validation/h1_v1_sources)，与protocol implementation SHA匹配。版本变更及哈希见 [runtime_classification_fix.json](/path/to/booster/reports/defense_validation/runtime_classification_fix.json)。当前v2已按相同prospective协议，在独立 `reports/defense_validation/h1_v2` 完成全部28runs正式复现，同样全门槛通过。独立逐数组核验：28runs、每run38个数组全部与v1严格相同，所有canonical physical trajectory SHA-256亦相同。证据见 [v1_v2_parity.json](/path/to/booster/reports/defense_validation/v1_v2_parity.json)。v1原正式artifact字节未改变，当前840个受保护原文件仍与H0清单一致。

当前源码复现实验命令（新目录，不覆盖冻结v1）：

```sh
.venv/bin/python scripts/validate_defense.py --output reports/defense_validation/h1_v2_reproduce
```

独立新增防御控制测试当前14 passed，涵盖实际零残差状态/控制/history等价、真实电机与关节变化、非法协议、原生正力自碰撞/非足地面检测、越界目标PD前拒绝，以及无物理推进的非有限分类fault回归。原297tests由根任务维护与复跑，本H1报告不以新增测试替代它们。
