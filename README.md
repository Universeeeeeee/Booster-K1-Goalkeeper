# K1 足球守门 MuJoCo 仿真基线

复用 Booster K1 官方 22 DoF MJCF 和用户原始球门 URDF，提供完整球场、5 号球尺寸的足球、物理接触、可重复射门、整球越线判定与数据记录。官方 `k1_walk` 已通过独立 A/B/C 验证，并完成阶段D守门场景集成和阶段E碰撞抗扰验证；**阶段F已实现仿真真值驱动的主动横向站位基线，尚未实现下蹲、伸手或扑救动作**。固定机器人测试只用于显示和隔离接触。

## 安装与资产

Python ≥3.11；本次验证环境为 Python 3.13.13、MuJoCo 3.3.7、NumPy 2.3.3、pytest 8.4.2、Pillow 11.3.0。运行目录为项目根目录。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/audit_assets.py
.venv/bin/python scripts/build_scene.py
```

机器人来自 [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets)，固定提交 `12a97516f57469078707afb587a5f84c1cb110d1`，路径为 `assets/booster_assets/robots/K1/K1_22dof.xml`。获取脚本核对该版本；审查脚本将使用的 XML、URDF 和 24 个 STL 与固定提交中的 Git blob 逐个比较 SHA-256。官方资产的许可证保留在其仓库内。

资产工作目录通过 Git sparse-checkout 只保留 `robots/K1/`、`LICENSE` 和上游 `README.md`。不检出 T1、T2、动作数据或上游 Python 包；再次运行获取脚本也保持此范围。Git 元数据用于固定版本和源文件校验。

`models/futsal_goal_3x2_reference.urdf` 是已有参考球门，仅移动了位置，文件内容保留。场景构建使用 MuJoCo URDF 导入与 MJCF 导出，保留 visual 后支架，再把两份转换结果加入场景。没有手工重建球门，也没有用 `<include>` 引入完整 URDF。

官方行走策略的独立 A、B 验证使用额外依赖和单独入口：

```bash
.venv/bin/python -m pip install -r requirements-walk.txt
.venv/bin/python scripts/fetch_walk_policy.py
.venv/bin/python scripts/validate_walk.py
```

默认运行30秒零速度自由站立，使用官方2 ms物理步长/50 Hz策略周期。可用 `--physics-dt 0.001` 做原步长对照，或在macOS用 `mjpython scripts/validate_walk.py --viewer` 观察。官方推理、关节顺序、缩放、滤波和PD直接复用；状态/电机按名称映射。来源、结果和后续边界见 [行走策略验证](docs/walk_validation.md)。基础场景安装不要求PyTorch；未安装额外依赖/策略资产时，策略专用测试会跳过。

阶段C已完成独立模型的前后运动、双向横移/转向、速度切换和停止恢复验证：

```bash
.venv/bin/python scripts/validate_velocity.py
```

98个工况在两种步长下各重复两次，保存物理子步轨迹、位移/姿态、跟踪误差和力矩饱和情况。小指令存在死区，2 ms下有明显肘部饱和；通过的指令点和实际限制见 [基础速度控制验证](docs/velocity_validation.md)。该入口仍只使用独立K1模型。

阶段D已将策略接入完整场景，详见 [场景集成验证](docs/goalkeeper_validation.md)：

```bash
.venv/bin/python scripts/validate_goalkeeper.py
.venv/bin/mjpython scripts/visualize.py --control-mode k1_walk --camera goal_front
.venv/bin/mjpython scripts/visualize.py --control-mode k1_walk --velocity 0 0.2 0 --command-duration 2 --camera goal_front
```

14个工况各重复两次，全部无跌倒、无足球/球门意外接触，停止后恢复稳定。短时±0.15横移和后退响应不足，报告保留原C跟踪判据和误差；`k1_walk` 仅用于自由模式、1ms物理步长。查看器先站立2秒，再短时运动并自动停止；Space暂停、R重置。

阶段E已完成碰撞覆盖审计及69个工况、138次冲击/对照执行，原161项保留，阶段E时187项测试通过：

```bash
.venv/bin/python scripts/audit_collisions.py --reproduce-tests
.venv/bin/python scripts/validate_impacts.py
.venv/bin/python scripts/audit_impact_solver.py
```

正面躯干2/4/6/8/12/16m/s已测点能维持并恢复平衡，20m/s触发高度跌倒阈值；偏心、足腿和横移组的已测范围见 [足球碰撞与抗扰报告](docs/impact_validation.md)。20个有效碰撞体仅局部覆盖，6个工况保留严格动量精度未达标标记，底层复核证明来自原Newton求解残差；没有修改控制或物理参数。

阶段F正式验证37个工况、74次执行，253项测试通过，详见 [主动站位闭环报告](docs/positioning_validation.md)：

```bash
.venv/bin/python scripts/validate_positioning.py
.venv/bin/python -m pytest -q > reports/positioning_validation/tests.txt
.venv/bin/python scripts/report_positioning.py
```

入口先验证0、±0.2、±0.4、±0.6m无球目标，再执行10组三方射门对照。站位误差最大约9.5cm；相同发射下被动挡住2组、主动挡住5组，新增3组滚动球拦截，均无跌倒或意外碰门。仅使用当前真值与保守可达预算，保持官方策略和物理原样；快球、大偏角、高球及弹跳预测存在明确限制。`--phase position`可单独复查无球站位，结果保存在独立的`reports/positioning_validation/`。

阶段G保留F默认算法，增加独立的已知物理预测与碰撞覆盖判定实验，详见 [预测与可达性报告](docs/prediction_validation.md)：

```bash
.venv/bin/python scripts/validate_prediction.py
.venv/bin/python scripts/compare_prediction_models.py
.venv/bin/python scripts/analyze_prediction_results.py
.venv/bin/python scripts/ablate_prediction.py
.venv/bin/python scripts/report_prediction.py
```

11个无球目标通过，原10球F/G均挡住5球；独立8球F挡住1球、G挡住2球，无原成功丢失。57个新工况共114次执行重复一致，另只读引用37个F存档工况；无跌倒、数值异常或撞门，297项测试通过。G将站位可达、现有身体覆盖和拦截可行分别记录，仍有“接触后进球”的局限。−0.4m到位更早但修正动作增多，因此未默认替换F定位器。完整协议、原始轨迹及审计均在独立`reports/prediction_validation/`；预测采用相同已知仿真物理，不能据此声称实球精度。

## 启动与观察

macOS 交互查看使用 MuJoCo 的 `mjpython`；其他平台可用 `.venv/bin/python`：

```bash
.venv/bin/mjpython scripts/visualize.py --paused --camera goal_front
.venv/bin/mjpython scripts/visualize.py --robot-mode fixed --camera robot_side
```

默认 `free` 模式包含真实自由基座、22 个关节和 motor。取消暂停后以零力矩运行，会倒下。`fixed` 移除机器人全部关节和 actuator，成为静态刚体，用于隔离碰撞；`absent` 移除机器人，用于足球/球门测试。球始终自由运动。

无需交互窗口，输出初始姿态截图：

```bash
.venv/bin/python scripts/visualize.py --camera overview --output reports/overview.png
.venv/bin/python scripts/visualize.py --camera goal_front --output reports/goal_front.png
.venv/bin/python scripts/visualize.py --camera robot_side --output reports/robot_side.png
```

截图不推进物理，不能用它验收站立。`--width`、`--height` 控制像素尺寸；三台命名相机为全场俯视、球门正面、机器人侧面。

`config/simulation.json` 配置机器人位置/yaw、足球位置/半径/质量和球接触参数。默认机器人 `(10.4, 0, 0.55369)`、yaw=180°，足球 `(7, 0, 0.111)`。可以通过 `--config path.json` 指定配置；场景构建还支持 `--robot-mode` 与 `--output path.xml`。输出场景依赖其相对路径指向的 STL，不是独立打包资产。

## 坐标与规则尺寸

单位为 m、kg、s、rad；右手世界坐标 X 沿场长、Y 沿场宽、Z 向上，原点为场地中心地面。右门在 X=11，K1 面向 −X，足球通常向 +X 射门。

| 项目 | 尺寸/世界坐标 |
|---|---|
| 草坪 | 26×18，X∈[−13,13]、Y∈[−9,9] |
| 有效场地 | 22×14，X∈[−11,11]、Y∈[−7,7]，四周缓冲 2 |
| 白线 | 宽 0.08；所有功能区域按中心线计 |
| 大禁区 | 8×5，左 X∈[−11,−6]、右 X∈[6,11]，Y∈[−4,4] |
| 小禁区 | 5×2，左 X∈[−11,−9]、右 X∈[9,11]，Y∈[−2.5,2.5] |
| 中圈 | 圆心(0,0)，半径2 |
| 中心/罚球点十字 | 每条线段总长0.2；罚球点(±7.5,0) |
| 角球弧 | 四角(±11,±7)向场内的半径0.7四分之一圆 |
| 球门 | 原点(±11,0,0)，净宽3、净高2、门框直径0.08 |

尺寸来自本次提供的赛事要求，没有另加尺寸不明的“专属球门区”。球门局部 +Y 朝场内，因此左门绕 Z 转 −90°、右门 +90°。立柱中心距3.08不是净宽；横梁中心高2.04，下缘2.0。门线参考平面严格在 X=±11。

K1 朝向通过官方 stereo optical link 的 +Z 方向核验为机器人局部 +X，再转到世界 −X。零关节姿态足底对应基座高度0.55169；默认增加2 mm间隙以避免初始地面穿模。这是几何初始化，不是平衡姿态设计。

## 物理参数与假设

保留官方机器人关节、惯性和电机参数，timestep=0.001、Newton 求解器和默认 Euler 积分。场地有一个无限碰撞平面，26×18的草坪与所有白线只是 visual，无碰撞；场外没有实体边墙。

足球为半径0.11、质量0.43的 sphere + freejoint，周长约0.69115，落在所提供5号球周长/质量范围。外观使用程序化棋盘材质。惯量取薄球壳近似 `I=2/3·m·r²≈0.00346867 kg·m²`，**属于建模假设**。

| 足球接触参数 | 初值 |
|---|---|
| friction：滑动/扭转/滚动 | `[0.6, 0.005, 0.0002]` |
| solref | `[0.008, 0.35]` |
| solimp | `[0.9, 0.95, 0.001]` |
| condim / priority | 6 / 1 |

球的较高 priority 使球与地面、机器人、门框的接触采用上述参数，机器人其余接触保留官方设置。`solref` 定义软接触响应，不能直接称为固定恢复系数；滚动和旋转阻力由接触参数产生。以上参数未通过实球落球、滚动或撞柱实验标定，不能称为 FIFA 规定的弹性或摩擦值。没有空气阻力、马格努斯力或额外经验旋转阻尼。

球门固定于世界，没有自由度；后支架只显示，无碰撞，没有球网。官方 K1 具有20个启用碰撞的几何体，足底/躯干及部分头部、手臂、腿部可接触，但视觉表面覆盖不完整：肩 pitch、hip pitch、ankle pitch 没有直接碰撞体，其他代理几何也只覆盖局部，mesh接触使用凸包。详细清单、22电机顺序和控制接口见 [资产审查](docs/asset_audit.md)。

## 射门与进球判定

先用无机器人的场景复现球门/足球基线，再用固定姿态隔离机器人接触；自由模式显示的是无控制器动力学。

```bash
.venv/bin/python scripts/shoot_ball.py --suite --robot-mode absent --output reports/shot_suite.json
.venv/bin/python scripts/shoot_ball.py --preset center_low --robot-mode free --seed 0 --output results/free_center.json
.venv/bin/python scripts/shoot_ball.py --random 10 --seed 42 --robot-mode absent --output results/random.json
.venv/bin/python scripts/shoot_ball.py --robot-mode fixed --position 8 0 0.5 --velocity 8 0 0 --spin 0 0 0 --duration 3 --output results/custom.json
```

10个预设为 `center_low`、`left`、`right`、`high`、`near_post`、`miss_left`、`miss_right`、`over_crossbar`、`post`、`crossbar`。预设名称描述输入，结果由实际接触与所选机器人模式决定。自定义位置、线速度和角速度不能与 `--preset`、`--suite`、`--random` 混用；方向由线速度向量给出。角速度单位rad/s，在球的局部框架记录；初始球姿态为单位四元数，因此初始局部轴与世界轴一致。还可指定 `--config`；默认时长3 s、seed=0、模式free。

每次射门创建新数据并reset，恢复机器人初始状态，设置球的位置、姿态及速度，然后逐步推进。由于控制器缺失，不执行“等待有效站立”。随机输入由seed生成；预设和自定义输入不额外加入随机扰动。

右门要求整球越过 `x−r>11`，左门 `x+r<−11`，并在首次完整越线时检查 `|y|+r≤1.5`、`z+r≤2` 及球底的地面容差。判断使用相邻步之间的线性插值，越线浮点容差为1e−6 m，球底允许2 mm软接触误差。第一次完整出界事件被锁存；从门柱外侧越线后再移到球门后方开口内不能变成进球。球中心越线不等于Goal。

结果区分 `Goal`、`Miss`、`Robot Contact`、`Timeout`。进球/出界立即结束；时长结束时有机器人接触但无出界事件则标记Robot Contact，否则Timeout。`goal`、`robot_contact`、`goal_contact`是独立布尔量；Robot Contact不是成功扑救，机器人接触之后仍可能判Goal。

JSON包含环境版本、资产提交、配置/场景哈希，以及每球编号、seed、初始参数、机器人/控制模式、全状态初始化/末态、每步轨迹、接触开始事件、球撞击前后速度、越线插值位置/时间、结果与结束原因。默认保存到 `results/shots.json`，使用 `--output` 可指定路径。当前只做首次进球/完整出界检测，不实现其他裁判规则；没有球网，进球后的球不会被网留住。

## 自动化验证与输出

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_assets.py --output reports/asset_audit.json
```

验证以加载后的几何、自由度和实际推进的物理状态为依据，覆盖资产/官方参数保留、场地尺寸、球门转换、落球/滚动/旋转、门框及机器人接触、整球越线和重复射门。`reports/validation.txt` 保存本次测试输出；`reports/asset_audit.json` 保存版本、源文件哈希、碰撞覆盖、控制器缺口与零力矩倒地基线。

原场景基线测试结果：**88 passed**，策略专用测试另存于 `reports/walk_validation/tests.txt`。测试输出中的 QVEL 警告来自刻意输入极端速度的回归测试：程序必须捕获 MuJoCo 的不稳定状态/自动重置并抛出错误，不能把重置后的轨迹当作正常结果。

无机器人10球预设测试为5次Goal、3次Miss、2次Timeout；后两次分别真实接触门柱和横梁后反弹，在3秒内没有完整出界。自由模式正中射门与固定模式躯干射门均记录到了机器人接触，见 `reports/free_center.json`、`reports/fixed_robot_contact.json`。三台相机截图以及 macOS 原生查看器打开/同步均已实际验证。碰撞测试覆盖门框12 m/s和向下15 m/s的足球，不能据此推断任意极端速度下均无穿透。

审查脚本在官方自由基座模型足底2 mm间隙处以零力矩运行2 s，基座高度从0.55369降至约0.208525，躯干up轴Z分量约−0.07619。这明确显示缺少控制的模型会倒下。

## 当前限制

守门场景已接入官方行走策略并完成阶段D验证。基础速度控制仍有低速死区、短时响应不足、误差和步长敏感性；阶段E已验证选定部位及冲击条件下的平衡，存在20m/s躯干冲击跌倒和碰撞覆盖/接触标定限制；自动扑救仍未实现。原 `booster_assets` 仓库只提供模型和动作CSV，控制代码/权重来自另一个 `booster_deploy` 仓库；两个仓库的审查与验证报告分别记录。`Robot Contact`只表示接触，接触后仍可能进球，不代表扑救成功。

本阶段没有强化学习、视觉识别、真实机器人部署、完整裁判、多机器人比赛、球网或复杂空气动力学。射门用预设球初始状态驱动；同一环境、配置和种子的确定性用于基线比较，不能据此保证跨MuJoCo版本逐位一致。
