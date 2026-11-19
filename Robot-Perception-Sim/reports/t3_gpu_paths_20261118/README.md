# T3 中间 RGB GPU 路径

用户选择方案 A 为主，停止 B 进一步接入。仅中间RGB，不涉及左右类脑、TD/SD。

环境：Isaac6.0.1，Replicator1.13.27，Warp；GPU2。保持800×600 ROI、D=[-0.2,0.04,0,0,0]、HDR输入、DLSS重建、关闭插帧与运动模糊。单次HDR与曝光比例模型，非真实快门积分。

A：原代码已经get_annotator('HdrColor',device='cuda')，瓶颈是随后.numpy()回读HDR。新代码在Warp CUDA FP16数组上计算曝光和sRGB/RGB8、线性测光归约；回读RGB8与一个浮点统计值。没有声称texture零拷贝：Annotator内部可能有GPU侧转换/拷贝。写盘仍同步，本轮不混入异步写盘改动。

300帧初测42.308fps含fsync，全文件几何像素ID及hash通过。CPU同画质历史两轮24.077/24.360fps，非本轮紧邻随机重复对照。数值核验两张同一源HDR，RGB8逐字节一致；GPU浮点原子归约亮度与CPU误差0.000218/0.000321（最大约目标0.45的0.071%），AE轨迹不承诺完全一致。数值探针回读HDR额外开销不计为性能样本。

B：GPU Interop节点在安装环境存在，普通图测试读回texture指针/尺寸为0，尚未接入渲染器post-render上下文，不能称不可实现或性能失败。官方曝光例子读取uchar4，不能直接处理浮点HDR。停止后续B开发。SPGNode/author_spg高层API未在6.0.1找到，但底层omni.rtx.spg0.2.1已随安装存在；不能把高层API缺失误称整个SPG不可用。未升级Isaac。

官方参考：
- https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.13.30/source/extensions/omni.replicator.core/docs/API.html
- https://docs.omniverse.nvidia.com/kit/docs/omni.graph.image.nodes/latest/GeneratedNodeDocumentation/OgnGpuInteropCudaEntry.html
- https://docs.isaacsim.omniverse.nvidia.com/6.1.0/py/source/extensions/isaacsim.sensors.experimental.rtx/docs/index.html

A延长测试：6000帧、8.64GB，独立读回所有像素ID和hash通过，含fsync 125.351s，47.8658fps。60秒源时间（12秒路线循环）约125秒墙钟。不是100fps通过。read_ms含GPU转换、归约及回读同步，不能当作纯PCIe拷贝时间；本轮不与旧read_ms直接同义比较。短测含首次内核编译，长测使用缓存，加上共享负载变化，不将42→48fps宣称新优化收益。
视频park_gpu_A.mp4来自A01原始300帧，按100Hz源时间播放，裁去顶部120行诊断区域，无叠字或后期降噪。
