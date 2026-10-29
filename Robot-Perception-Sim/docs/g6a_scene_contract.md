# G6-A 完整K1场景、头部ZED与冻结轨迹

这是显式 **kinematic frozen replay**：完整G1 K1/场地/球门视觉资产、官方ZED X、G4已记录的球姿态逐样本重放。没有walk策略、自由站立或新增碰撞真实性结论。源G1文件和所有G4记录保持原样，新helper在独立stage上禁用源碰撞/关节并移除G1机器人及球RigidBody API；官方ZED的IMU刚体保留kinematic，避免嵌套机器人刚体。

## 真实挂载层级和来源

`acquisition/g6_scene.py::G6Scene` 引用完整G1到 `/World/G1`。质量为0的原官方光学头框架为：

```text
/World/G1/Bodies/b3_aahead_pitch_link/b5_head_booster_stereo_rgb_link
```

官方ZED引用位于其真实USD子层级 `ZEDMount/ZED`，左右相机为 `ZED/base_link/ZED_X/CameraLeft` 和 `CameraRight`。不是独立漂浮相机再拼一个名称。光学列向量为+X向右、+Y向下、+Z向前；USD相机为+Y向上、−Z向前，两者用 `diag(1,-1,-1,1)` 转换。

挂载复用已有 `config/cameras/zedx_wide.json` 的虚拟假设：stereo midpoint相对头光学框架 `[.0351,0,.04]m`，相对旋转identity。**这不是测量硬件外参。** 先读取官方USD的实际左眼相对root矩阵，再解 `T_head_from_root = T_head_from_left_desired @ inverse(T_root_from_left_optical)`；因此没有假定官方ZED root就是光学框架。左眼相对头的预期平移为 `[−.0249,0,.04]m`，右眼为 `[.0951,0,.04]m`。

头部姿态由G1两条真实USD joint localPos/localRot/axis恢复：`T_world_child = T_world_parent @ T_parent_joint @ Rx(q) @ inverse(T_child_joint)`。G1导出中的关节axis X指joint-frame轴，并非直接把head local X当yaw；localRot把yaw映射到Z、pitch映射到Y。姿态更新同时驱动独立导出的yaw/pitch刚体世界变换，其原massless光学子框架和ZED自然继承。其余K1保持原始零关节视觉姿态。

## 两个姿态的实际USD核对

已在目标机Isaac 4.5的USD-only Python环境运行 `scripts/validate_g6a_mount.py`，未启动GPU渲染或推进物理。完整矩阵、相机属性读回、源文件SHA和分时刻球状态见 [实际挂载验收](g6a_mount_acceptance.json)。第二姿态在正式采集前固定为yaw=+0.2、pitch=+0.1rad，正pitch的实际光轴朝下，未按感知效果事后调整。

| 姿态 | 左眼world中心m | 右眼world中心m | optical forward world |
|---|---|---|---|
| neutral `[0,0]` | `[10.294262,-.060000,.894364]` | `[10.294262,.060000,.894364]` | `[-1,0,0]` |
| yaw_pitch `[.2,.1]` | `[10.299591,-.080439,.883903]` | `[10.275751,.037169,.883903]` | `[-.975170,-.197677,-.099833]` |

两个姿态的 `T_left_from_right` 均为identity旋转、平移 `[.12,0,0]m`；baseline读回0.12m，左右光轴平行。实际USD相机属性为focal length1.112400055、水平aperture2.880000114、垂直aperture1.799999952；960×600时K读回fx=370.8000036、fy=370.8000281、cx=480、cy=300。这复用当前官方SDK SVGA校准，不以nominal110°FOV反推另一个K。

两个姿态还独立检查了**修正后的真实SDK左右PNG**（不是renderer预览截图）：neutral receiver01的frame24与yaw_pitch receiver02的frame26，都关联source visual ID30、物理t=0.00500000024s，双眼身份/时间验证有效。白球、绿色原turf、场地线与球门均实际可见，无相机壳体遮挡；左右球位置的视差方向和头部转向与已读回pose一致。按SDK K和已读回Twc投影的3D球心为：neutral左右约 `[497.738,296.160]` / `[462.262,296.160]` px，yaw_pitch左右约 `[580.839,255.574]` / `[544.067,255.574]` px。球体离轴时可见轮廓中心略偏离3D球心投影，不能把手工观察当亚像素标定误差。实际文件路径/SHA、双眼矩阵、几何投影及明确的人工检查边界见 [SDK姿态图像检查](g6a_sdk_pose_inspection.json)；没有生成或补填任何检测预测。

## 冻结轨迹与时间

来源清单/SHA/初态为 `config/scenarios/g6a_sources.json`：

- `direct_goal` → G4 `direct_middle_center_repeat_0.json`，初态 `[9,0,.9]m`、速度 `[8,0,1.5]m/s`，265个原生约1ms样本，末时刻0.26400001254s。
- `one_bounce` → G4 `bounce_one_repeat_0.json`，初态 `[9,0,.45]m`、速度 `[10,0,-2]m/s`，241个样本，末时刻0.24000001140s；G4实际一次地面反弹后Goal。

采集重用G3曝光时间窗/不可变左右缓存，0ms与10ms共用同一轨迹和头姿态。helper按每个曝光subtarget请求时间选择最近的**原始**约1ms样本，不插值、不外推、不重新模拟；返回requested/realized时间、量化误差、索引、来源SHA、原速度和四元数。轨迹内部最大量化约0.5ms；preroll/postroll保持端点，明确标记hold及其真实时间差，不把hold的时间差解释成采样量化精度。

轨迹没有平移、缩放或为了留在FOV而延时。两条球均从机器人前方约1.29m处开始。对全部原始1ms样本的独立几何可见性检查见 [时序验收](g6a_visibility_acceptance.json)：

| 姿态/轨迹 | 完全可见时段s | 后续几何原因 |
|---|---|---|
| neutral/direct_goal | 0–0.146 | 0.147–0.152部分near clip；0.153–0.165近裁剪/相机与球相交；≥0.166出FOV |
| yaw_pitch/direct_goal | 0–0.143 | 0.144–0.148部分near clip；0.149–0.160近裁剪/相机与球相交；≥0.161出FOV |
| neutral/one_bounce | 0–0.071 | ≥0.072出垂直FOV |
| yaw_pitch/one_bounce | 0–0.079 | ≥0.080出垂直FOV |

原始once-bounce记录的实际Vz负转正样本为 **t=0.130000006s**，球world位置 `[10.297961,0,0.109914]m`。此时neutral左眼optical球心 `[.060000,.784450,−.003700]m`，已经在眼后；down-pitch左眼optical球心 `[.079159,.771559,.062958]m`，虽在正Z一侧但远低于视锥。两者独立判定均为out_of_fov、expected_visible=false；可见性早于反弹约58ms/50ms丢失。

因此 **G6-A不在视觉上观察到实际bounce**。它完成一次反弹来源的完整冻结重放，并可检查接近过程/FOV丢失；不能把正确的出FOV失检计作检测器漏检，也不能宣称反弹后视觉跟踪成功。后续G6-B可另选更长进场距离/更完整视场覆盖的物理轨迹，但不在本次两个已冻结case中移动球或相机来改善结果。

## 视觉和可见性边界

正式球保留G1 USD原样：`g102_ball_collision`半径0.11m、displayColor `(1,1,1)`、opacity1。G1导出没有原父工程的黑白checker纹理；本轮未补造纹理、橙色球或替代几何。完整K1/场地/球门仍在渲染输入中。

`visibility_state` 独立于检测器：球体/视锥平面测试，加实际stage可见Mesh三角形的球心前表面及8个环向点射线，返回hit prim路径、近裁剪、完整/部分FOV及visible/partially_occluded/occluded/out_of_fov。近裁剪单独使用partially_near_clipped或near_clip_or_camera_sphere_intersection，不能误记成物理遮挡；时序JSON用near_clip缩写后者。有限9射线不等同所有像素的精确遮挡分割，非Mesh primitives不参与该射线测试；正式图像仍应独立核验。

初版不考虑near clip的ray模型曾把官方ZED自己的 `Mesh_030` 当遮挡。独立USD核对发现：其材质名Glass_Basic_Grey但shader opacity实际1，不能据名称宣称透明；该平面仅距eye前方0.292mm，而官方camera near clip为10mm。最终ray模型按实际near-plane剔除近裁剪之前的交点，**保留整个ZED rig参与检查**，未删除/隐藏镜头或凭材质名豁免。两姿态的0/50/100ms及末帧独立几何结果已保存；早期样本visible，末帧out_of_fov。真实渲染图像、曝光配对和感知误差由G6-A采集/评估报告另行确认，USD检查本身不等同图像成功。

## 冻结状态实际写入USD的独立读回

另用 `scripts/validate_g6a_replay.py` 在USD-only环境对两个轨迹×两个头姿态×0/10ms全部8个条件执行独立读回，不修改已冻结helper，不运行SimulationApp/renderer/physics。验证器按协议独立生成60Hz中点与10ms的5个等权midpoint-quadrature子样本，再从原JSON独立寻找最近源状态。不是只相信 `author_state` 返回的GT字段：每次之后以新的USD XformCache读取实际ball world矩阵，并独立比较归一化源四元数和旋转矩阵；同时检查相机读回及其他K1刚体。

结果见 [实际USD重放验收](g6a_replay_readback_acceptance.json)：128个目标中点及384个subsample条目共 **512次检查**（含重合中点的重复检查），覆盖直接球17个、反弹球15个目标中点/条件。21个非head-yaw/head-pitch的K1刚体逐次比较原固定世界变换。每条件另导出并重新打开USD，核对ball、双眼和全部静态K1矩阵。

| 独立读回量 | 最大差异 |
|---|---:|
| 实际USD ball translation vs独立选出的源位置 | 0m |
| 归一化四元数sign-invariant L2 | 5.1700e−7 |
| 实际旋转矩阵vs源四元数独立公式 | 1.01894e−6 |
| 其他21个K1刚体world矩阵变化 | 0 |
| 实际USD左眼矩阵vs记录的T_world_camera | 0 |
| 双眼相对矩阵vsidentity+0.12m optical X | 3.33067e−16 |
| 8次序列化/reopen后world矩阵差异 | 0 |

四元数不是逐bit相同，存在约1.03e−6rad量级表示差异，不能把它写成零误差；平移与其他固定刚体确实读回为零差异。全逐次矩阵/索引/时间记录与8份序列化stage位于 `results/g6a/usd_replay_readback_final_20261030/`，轻量JSON保存trace与stage SHA、实际载入helper/validator SHA和各条件最大值。这证明USD接受了对应姿态，不代替实际SDK图像、检测成功或动态物理验证。

复验使用与mount脚本相同的G1/camera/direct/bounce/mount-config参数，把脚本替换成 `validate_g6a_replay.py`，输出参数为 `--output-dir /absolute/path/new_readback_directory`；同样只需USD Python环境，可设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=1`，不占GPU。

## G6视觉层：遵守源alpha=0，而非重涂场景

实际SDK图像曾显示整块灰色场地。独立检查发现G1 `g0_ground` 碰撞Plane在z=0，明确 `displayOpacity=[0]`；绿色 `g1_turf` 顶面位于z=−0.001m，源RGB为 `[.12,.36,.18]`。RTX实际把本应不可见的ground当灰色遮挡面，原先只读opacity的检查没有证明渲染器真正采用它。

已在相同Isaac/Ubuntu22 renderer、固定相机/光照/白球状态下执行最小A/B/D反事实。报告与实际载入module SHA见 [视觉保真验收](g6a_visual_fidelity_acceptance.json)，图像保留于 `results/g6a/opacity_counterfactual_accepted/`：

| 版本 | 唯一视觉层变化 | 场地区域median RGB | green fraction |
|---|---|---|---:|
| A | 原始G1显示数据 | `[97,97,97]` | 0 |
| B | 仅原本alpha0的ground设invisible | `[144,190,160]` | .7184 |
| D | 全部10个原本alpha0的Gprim设invisible | `[144,190,160]` | .7204 |

没有执行颜色或材质重绑定。绿色来自原始turf。最终helper仅对**明确authored且全部displayOpacity恰为0**的10个源Gprim添加G6层 `visibility=invisible`：ground、3个机器人碰撞Mesh、6个门框碰撞Cylinder；path/源opacity/覆盖值逐项保存于provenance。球的alpha为1，原白球、其他RGB、几何、相机、轨迹及源G1/G4/G5文件均未更改。独立ray模型原本已经排除alpha0几何，这个改动让渲染结果遵守相同源语义，而非为了检测率重涂球或背景。

先前灰色G6录制保留为诊断，不能与修正后的绿色正式0/10ms配对混合。最终helper SHA为 `1aa27f52eb0493bf41a7abb5f50403c8b26d4545fdf3b8f1f0c675a3beb7be12`；capture必须保存实际执行版本。反事实脚本 `scripts/diagnose_g6a_opacity.py` 会显式恢复原visibility生成A，再生成B/D，因而在修正helper之后仍可复验；每次使用新输出目录。早期probe使用错误配置路径/namespace import的尝试及日志完整保留，不作为最终实际module验收。

## 接口与复验

```python
scene = G6Scene(stage, g1_usd, camera_usd,
                {"direct_goal": direct_json, "one_bounce": bounce_json},
                mount_config_path, head_pose="neutral")
scene.select_trajectory("one_bounce")
state = scene.author_state(requested_source_seconds)
```

`camera_root_path/left_camera_path/right_camera_path` 用于官方annotator，`camera_pose` 为G3 marker所需Gf row矩阵。`metadata()` 保存mount/scene/轨迹SHA与当前相机，`author_state()` 返回双眼world USD row矩阵、optical列矩阵、K、GT球心及独立可见性；这些GT只用于采集记录/离线评估，不能传入检测器。

USD-only复验命令（先按G1文档设置USD的PYTHONPATH/LD_LIBRARY_PATH/PXR_PLUGINPATH_NAME）：

```sh
python.sh scripts/validate_g6a_mount.py --g1-usd /absolute/path/robot_v6.usda --camera-usd /absolute/path/ZED_X.usdc --direct /absolute/path/direct_middle_center_repeat_0.json --bounce /absolute/path/bounce_one_repeat_0.json --mount-config config/cameras/zedx_wide.json --output results/g6a/mount_new.json
../.venv/bin/python -m pytest -q -c pytest.ini tests/test_g6_scene.py
```

5项CPU契约测试通过；实际两姿态USD检查全部通过。大资产、冻结轨迹和早期错误ray模型审计保留在 `results/g6a/` 与远端独立 `g4_20261029/g6a_runtime/`，不覆盖旧G1/G4/G5实验。
