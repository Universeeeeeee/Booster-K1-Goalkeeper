# 阶段 I1：训练环境接口静态审计

本审计针对 I1 新环境接入前的现有接口，目标是给未来统一的站立、移动、拦截、恢复策略建立可核验边界。审计产物为本文件及 `reports/training_interface_validation/interface_audit.json`；随后按父任务分工新增 `k1_goalkeeper/training_actions.py`。以下静态审计过程未执行 `mj_step`、策略推理、训练、新控制实验、碰撞补全或奖励调优。其后新增环境验证脚本 `scripts/validate_training_env.py` 的物理运行证据另存于验证 protocol/results，不属于本静态审计结论。A–H 的 322 tests 通过数由父任务提供，本审计没有重跑该测试集。本文件不代替 I1 新环境实现与运行验证。

结论：现有代码能提供组合场景、名称映射、官方步行控制、全场 whole-ball 事件和物理记录基础；尚不能凭接口可调用就宣称统一全身策略、直接 PD 站立能力或训练可用性已经验证。建议新增独立环境适配器，显式区分 `body_velocity` 与 `joint_target_pd` backend；严格分开 actor 当前观测、critic 当前特权与离线未来真值。

## 审计证据与模型独立核验

本地官方 deploy 子集固定提交 `7bb1462e2ac742eba5c01659b815e2e99fbe617f`，清单 22 个文件 SHA-256 全部匹配。仅编译现有 XML，不推进物理、不重新写场景：

|模型|nq|nv|nu|物理步长|
|---|---:|---:|---:|---:|
|官方 K1|29|28|22|1 ms|
|自由 K1 + 球场|36|34|22|1 ms|
|固定 K1 + 球场|7|6|0|1 ms|
|无 K1 + 球场|7|6|0|1 ms|

JSON 保存各模型源哈希、逐关节 qpos/dof 地址、执行器 transmission/gear/force 限制、实际 task 默认姿态/KP/KD/effort、源符号行号和现有测试源哈希。名称检查独立依据编译模型，而不是仅复用控制器 getter。fixed/absent 只有球的自由关节，不能当作站立策略环境。

## 物理所有权、reset 与动作接口

`scene.build_scene` 复制官方 XML，并保留 free/fixed/absent 分支；默认输出位置会写 `scenes/`。新环境构建若需要生成场景，应将 output 显式放到 I1 自有目录或临时目录，避免覆盖已有场景或历史报告。默认机器人在 `[10.4,0,0.55369]`、yaw 180°，球在 x=7，守右门。坐标与 defended goal 必须写入环境契约，不能只从 shot 名称猜测。

`SceneSimulation.step` 是单物理步入口；k1_walk 每 20 步推理一次，持有该次目标并每 1ms 更新 PD。`advance_policy_step` 仅支持步行 controller。调用方再调用 `controller.step/ctrl_step` 会引入第二个物理推进者；一次性写 `data.ctrl` 又会被场景下一次 PD 覆盖。环境 `step(action)` 应持有动作并推进约定的整数物理子步，返回真实推进时间与子步数。

现有 `SceneSimulation.reset` 清 `MjData`、物理/策略计数；walk `initialize` 用名称地址恢复 task 默认关节姿态和场景初始自由基座，清速度/控制/命令，更新状态并 start。start 清 controller 步数/时间，policy reset 清 `last_action`、history、filtered target。`set_velocity` 切换命令保留历史。测试 `test_command_switch_retains_history_and_reset_starts_new_episode` 已具体覆盖这些语义，但本审计没有重新执行它。

新环境每次 reset 还必须新建 `GoalDetector`（类本身无 reset 方法），清 contact episodes、预测缓存、coverage、上层控制状态、上次 action、done/recovery 状态与所有记录。不要在 reset 后继续使用上一回合的 observer。随机采样应由环境独立 RNG 持有，保存 seed、采样输入和完整初态；`Shot.seed` 是 metadata，`run_shot` 不据它自行随机化。

|backend|动作与控制路径|依赖及限制|
|---|---|---|
|`body_velocity`|3维 `[vx,vy,yaw_rate]`，当前机体坐标，m/s、m/s、rad/s；交给原 k1_walk|依赖 Torch 和 `k1_walk.pt`。原 bounds：vx ∈ [−0.3,1.6]、vy ±0.3、yaw ±1.8。它提供已冻结的步行能力，不是新策略直接控制22关节。|
|`joint_target_pd`|22维官方 robot 顺序的绝对目标角度 rad，经名称映射写官方 task PD|无需 pt 推理。若新实现只用 MuJoCo/NumPy，可避开 Torch 依赖；不能导入 walk_controller 后声称无 Torch 依赖。直接目标并不保证站立、移动、抗球冲击或恢复。|

对于归一化动作或默认姿态残差，应作为另一套明确动作语义记录 scale、默认位置和有效目标，不应把相同22维数组有时解释为绝对目标、有时解释为残差。非法 shape、NaN/Inf、范围越界的 reject/clip 策略也需明确；不要靠广播或 XML 物理限位修正输入。

## 名称映射、官方参数及停止边界

`K1WalkController` 核验22个命名 hinge、各自唯一同名 joint transmission motor、unit gear 和 torque motor 结构；从 `jnt_qposadr/jnt_dofadr` 读取状态、用 actuator IDs 写控制。自由基座 `world_joint` 与球 `ball_free` 必须独立解析。不要直接假定 `qpos[7:29]`、`qvel[6:28]` 或 `ctrl[:22]`，这些假设在新增关节、重排执行器后不成立。现有 walk 测试已有额外关节和重排 motor 的案例。

运行参数来自 `K1WalkTaskCfg` 对 `K1_CFG` 的覆盖，而不是基础机器人或 prepare-state 参数：头 KP/KD=4/1、臂20/2、髋/膝100/2、踝65/1。软件 effort 按官方 robot 顺序为 `[6,6,14,14,14,14,14,14,14,14,30,20,15,35,24,15,30,20,15,35,24,15]` Nm。实际公式为 `clip(KP*(target-q)-KD*dq, ±effort)`，XML motor 的更宽上限不能当作新环境允许的力矩。

walk actor20维动作覆盖全部八个臂关节与12腿关节，头保持默认PD。policy joint 顺序与22维 robot 顺序不同。它的一帧69维由角速度、投影重力、机体速度命令、20关节 q−default、dq×0.1、20维上次 actor 输出组成；10帧历史690维。官方 action_scale=.25、filter=.8、右肘 pitch 固定补偿；该内部历史不能简单从最终PD目标倒算。原策略不限制目标到 XML 角度范围；扭矩限幅不证明目标可行。

现有 walk 检查 BADQPOS/QVEL/QACC/CTRL、非有限状态和时间偏离；跌倒安全停止可能在计算观测时触发，adapter 在交出目标前再次检查 running。`stop()` 只改 running flag，不自动清 ctrl。新环境必须明确停止后不继续物理推进，并保留最后一次已交付状态、告警和原因。有限性检查单独不足以识别 MuJoCo 自动数值重置；应检查告警增量和实际时间递增。

## actor、critic 与离线真值三条通道

建议观测结构分别暴露 `actor_observation`、`critic_privileged`，离线标签由回合结束后的记录器生成；不要把整个 `info` 或历史 summary 自动拼接到 actor。访问时复制数组，避免外部修改观测反写场景或泄漏共享 buffer。

|通道|允许内容|关键限制|
|---|---|---|
|actor|时刻 t 的自身本体状态、上一次已执行动作、当前/过去的球观测及 validity/age；必要时明确标记的因果预测|全部来源时间 ≤ t。不能用未来帧平滑，也不能读尚未发射球的未来 launch 参数。|
|critic 特权|时刻 t 的精确 simulator 状态、明确声明的 ball spin/quaternion、物理参数或完整当前状态|须独立字段。默认不包含未来实际结果；部署 actor 不应误用此输入。|
|离线 future labels|实际 crossing、最终 Goal/Miss/contact/save 判断、恢复验证、无机器人对照 crossing、预测误差|在回合后生成或消费；不进入 actor。真实或反事实未来轨迹都属于此通道。|

当下精确 `qpos/qvel` 不是未来泄漏，但应称为 `simulation_state` 观测 baseline；它不能被称为部署传感器估计。F/G 从当下 simulator 获取球位置、速度、角速度，未实现相机检测、估计误差、丢失或传感器延迟。球 spin/quaternion 是否可被 actor 看到必须显式假设；若任务设定无法测量，则只给 critic。

因果预测与未来真值不同：G 的 `BallPredictor` 在独立 `MjData` 用当前球态预测 ball/floor，允许作为明确模型假设下的特征或诊断。它省略机器人及球门碰撞；`CollisionCoverage` 是冻结当前机器人姿态和候选平移的几何假设，不能视作未来步态真值。预测应带 valid、observed_time/age 和模型假设；`reference_crossings`、实测事件和 post-hoc prediction_errors 属于离线通道。不得调用完整实际场景未来 rollout 再伪装成 actor 当前观测。

## 事件、回合结束与恢复记录

`GoalDetector` 首次 whole-ball 全场退出 latch，按前后位置线性插值判定；中心 x=11 不是整球过线，右侧整球平面为 `11+radius`（默认11.11）。门口要求整球横向处于 ±1.5m、球顶≤2m且底部满足 floor tolerance。球初始完全在场外直接 Miss。它同时检测左门、右门和 touchline；环境必须明确 defended_goal，不能把任何 Goal 都当成右门失守。

机器人 contact 不等于 save，goal frame contact 不等于 Goal，Timeout 也不等于成功。`run_shot` 遇首次 exit 停止，未启动站立 controller，并在无 exit 时可能以 Robot Contact 为 outcome。E `run_impact` 在 exit latch 后继续记录，有利于观察冲击后恢复。统一任务若在 goal/miss 那一刻终止便无法评估后续恢复，需要明确 recovery tail 或另一类终止语义；应区分物理/任务终止、时间截断和数值异常，避免异常静默重置后当正常样本。

现有 E recorder 保留每1ms的 qpos/qvel、base姿态/速度、raw PD、ctrl、`qfrc_actuator`、contact force、球速度、动量残差与告警；命令在推进前读取，observer在推进后读取。launch 在命令选择之后发生，因此50Hz hook首次反应通常是2.02s，而非先看见2s发射后的状态。这一时序不能被新环境无意改变。原 recorder 基于 walk controller，直接 target backend 的新记录器需要独立保证相同物理字段和时间语义。

验证应把动作、目标、raw PD和已交付ctrl与每个物理间隔关联，再记录 post-state/applied torque/native contact。只记录 getter 宣称的 target 不够。源/配置/模型哈希、seed、初态、action/backend契约、actor采样时间、实际子步数、warning delta和终止原因应保存；最终离线标签应另存。trace应避免 reset 后残留上一回合的数据。

## I2 待解决的实际缺口

1. **统一全身动态能力未验证。** 现有 SceneSimulation 只有 zero_torque/k1_walk，H 原型是步行策略加受控臂脉冲；接上22维PD接口不证明统一站立/移动/拦截/恢复已成立。I2需要真实推进的轨迹、PD/执行力矩与关节响应证据。
2. **部署观测缺口。** 当前精确球态 baseline 无传感器估计路径；遮挡、延迟、噪声、mask/age 的任务定义和验证仍需明确。I1可以冻结真值观测假设，不能宣称已解决视觉输入。
3. **模型有效性限制。** 官方 shoulder_pitch、hip_pitch、ankle_pitch 部位有 visual-only 几何，碰撞代理不覆盖整个视觉表面；球薄壳惯量在 scene 源码标为待校准。G forecast不包含球门/机器人接触。I1如实记录限制，本轮不补全碰撞或调物理参数。
4. **训练与吞吐未建立。** 现有核心是串行验证和冻结策略推理；未建立优化器、rollout buffer、课程、奖励或向量训练吞吐证据。I1仅提供接口与小规模运行验证，不启动训练，不并行，不做奖励调优。
5. **结束与恢复目标需定稿。** whole-ball outcome、实际 contact与连续恢复是不同证据。I2需在 episode契约下用可核验轨迹判定成功和失败，避免用单次触球或未进门超时代替扑救与恢复。

审计成功只表示源文件阅读、官方清单验证和静态模型编译完成。后续 I1 实现必须另行验证非法动作、名称映射、reset隔离、真实时间/子步/PD、因果观测与终止记录；本报告不把接口静态检查升级为训练成功结论。

## I1 动作模块交接

新增 `training_actions.py` 提供 `create_action_backend(model, mode="joint_targets")`，支持 `joint_targets`（绝对22维目标）与 `body_velocity`。`JointMapping` 含名称、joint/actuator IDs、qpos/dof地址、自由基座与球地址、task默认姿态/KP/KD/effort及XML joint ranges。两backend均暴露 `model/data/mapping/action_space/default_action/reset/set_action/step_physics`。非法shape、非有限数、越过Box边界的动作直接拒绝；速度Box由task实际bounds建立。

joint backend通过AST读取与manifest校验的两份官方配置，不导入walk controller、不加载pt，每物理步按float32状态计算官方task PD，只写命名机器人motor。body backend复用原 `SceneSimulation(k1_walk)`，额外依赖仍显式保留。只在独立Python进程进行了AST语法与编译模型映射检查：22个唯一joint/motor、球与机器人地址分离、导入guard证明未导入Torch/walk；没有实例化步行策略或推进物理。模块reset、真实PD交付和环境行为由父任务的I1验证负责，不能用这些静态检查宣称已经通过。
