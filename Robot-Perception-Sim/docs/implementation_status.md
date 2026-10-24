# v2 计划执行状态

依据：[原对话下载的 v2 路线图](sources/chatgpt_roadmap_v2.md)。业务范围保持纯仿真、仅目标robotic-4090的GPU0/1；不运行实机、不修改既有运控/训练模块。

## 当前边界

用户最新要求为“继续执行G2，修复好后提交，暂停”。本轮G2验收已完成，提交后暂停，**不继续G1/G3及后续阶段，完整项目尚未完成**。用户随后取消额度检测，监测agent已停止；没有使用重置卡。

前一轮因剩余5%重置为100%按用户条件暂停；本轮由用户明确恢复，只执行G2修复。子agent使用GPT-6.1 Sol、中等推理。

## 已完成

- G0审计提交6e765c1、G1源资产契约94cfe2d、五类MuJoCo参考8a73def已经推送到feat/perception-p0-p1。G1仅源端准备，尚未进行USD/PhysX等价验收。
- G2已用官方扩展v3.2.0、正式SDK5.0.7和独立Ubuntu22用户空间实际运行；宿主Ubuntu20、驱动和旧Isaac安装不变。
- 两轮最终运行022/023，各100次连续成功唯一SDK抓取。发送/接收四个退出码均0，同端口30000复用成功，没有自有GPU进程残留。
- 两轮全部左右PNG、depth/confidence文件已在本地独立读回；100/100帧的SDK参考、源端Writer事件和左右像素ID完整对应，参考残差0ns。2m平面SDK深度中位约2.019m，中央有效率最低100%。
- 恢复官方默认关闭方式并显式释放调用者资源，公开post_quit传播真实退出码；正常0和真实场景异常1均已验证。未把旧Kit逐扩展关闭模式的底层GC缺陷宣称已被修改引擎修复。
- 时间输入改接成对渲染参考，源观测使用官方Writer同数据包的reference_time和RGB；不再轮询当前图节点推测历史帧，不把参考时钟当Unix/World时钟。作者状态日志与观察事件日志分开。
- 关闭实际DLSS/时域AA限制并核验输出，诊断条可清晰独立解码；没有通过放宽阈值掩盖错帧。
- 新增独立验证器17项测试通过，静态审阅无阻止提交的问题。

完整说明、适用边界与复现命令见[G2验收报告](zed_sdk_stream.md)，轻量机器可读结果见[g2_acceptance.json](g2_acceptance.json)。

## 数据位置与恢复入口

本地：Robot-Perception-Sim/results/g2/。最终数据为sender_022_final、receiver_007_final、sender_023_final、receiver_008_final；旁有日志、退出码、NVML证据、原始/主agent独立验证报告。大型数据不提交Git。

远端：lizhengran-server:/home/lizhengran/isaacsim-workspace/g2_zed_poc/runs/。同名完整数据保留。运行环境和依赖位于该PoC根目录；不要重新覆盖旧安装或已有结果。

若用户以后明确继续，先核对新Git HEAD、当前工作区差异及远端是否已有作业，再按原计划选择下一阶段。G2当前约10fps接收吞吐包含诊断和落盘开销；30fps实时、真实曝光、运动机器人与真实硬件校准均未验收，不据G2通过外推这些结论。
