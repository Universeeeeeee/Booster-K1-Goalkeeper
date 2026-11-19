# RGB isolated continuation

Only files under this directory were edited; shared RGB/stereo scripts/config unchanged. Remote counterpart /data_ro_1/lizhengran_t3_rgb_isolated_20261118. GPU2 used, GPU4 other workload avoided. Separate runtime user/home/cache/tmp now used. No other processes stopped. Initial wrapper copies shared cache paths; this was recognized and changed before queue04. Do not claim original wrapper fully isolated.

P2 async03: actual pixel mismatch expected1 got0, captured failure.txt. Reject current async rendering candidate, no throughput claim. gating03 failed CUDA external memory import/semaphore; this does not validate fixed gating timing. queue04 isolated cold-cache run timed out150s before capture; no pending-queue performance result.

P0 installed orchestrator DEFAULT_SUBFRAMES_REALTIME=1; unset RTSubframes falls back1 in its code, manual world.step does not call orchestrator step. Actual GPU pass counts remain unmeasured. nsys --trace=cuda,nvtx --sample=none --cpuctxsw=none successfully generated smoke report for /bin/sleep, not Isaac CUDA trace. CPU perf sampling lacks permission; CUDA trace not proven unavailable.

P1 short disk test864MB inclusive fsync402.20MB/s, all600 blocks independently hashed. Short cache-sensitive result, not sustained capacity claim. Real pending counter implemented only in isolated script; not yet executed to valid recording.

Next: warm isolated environment successfully; collect valid queue series, sustained disk test, and actual CUDA trace. No merge/deployment into shared paths. Previous RGB7200-frame59.96fps result remains historical evidence, not new success.

## queue05有效结果
预热100帧独立校验通过，首次加载约120秒含Shader和Warp编译，不计性能。随后三组1200帧：同步47.94、异步54.41、无写盘54.66fps含各自收尾。异步真实pending峰2、p95=1、最终0；future回收等待0.0166s不等于真实满队列阻塞。无写盘仍做哈希，不做文件持久化验证。同步和异步全文件像素ID/hash通过。单次短测，未做重复稳定性统计；场景加载与渲染预热分开，但Shader编译尚未独立量化。

磁盘8.64GB、1.44MB块、同目录buffered顺序写，平均含fsync436.28MB/s，首864MB1850.60MB/s，p95普通write0.9815ms，最终fsync14.9925秒，6000块全部读回hash通过。写入前段主要落页缓存，不能把前段高速认作持续盘速；本轮对有限批次持久化预算有余量，未充分证明无限时100fps无积压。

Tracy扩展启用、start_tracy_profiling已调用，trace08正常跑完但capture未连接；没有可分析GPU区间。nsys09通过容器内部可见工具挂载捕获了Vulkan启动API，但旧Nsight2023.4.4的CUPTI注入与libtorch_cpu所需cuptiActivityEnableDriverApi冲突，初始化退出，无稳定采集GPU轨迹。nsys05/07空报告和错误路径不计通过。暂未修改系统库、升级工具或干扰双目。三个问题（实际GPU渲染时间、CPU等待GPU比例、重复RenderProduct实际更新）尚无完整证据回答。

## queue06重复测试与资源冲突
用户保留异步写盘，要求6000–12000帧重复和优先GPU Profiling。两轮6000帧请求均首帧空一维HDR导致Warp拒绝，无有效长测。增加实际尺寸/像素65535预热门槛后，关闭Viewport的两轮仍预热失败；不关闭Viewport的30帧探针ready_probe3成功，提示需检查Viewport对象与目标RenderProduct关联，尚未单变量重复确认。未将该30帧作为性能证据。

09:29UTC发现另一任务实际运行/poc/t3_abc_20261118/source/scripts/profile_t3_render.py，与GPU Profiling目标重叠。未改动或停止该任务；自身短探针结束后未启动新长测/Trace，已向用户询问分工/错峰。公共代码配置未改。隔离脚本新增预热拒绝与显式同步设置，不重试async03。Tracy准备显式绝对插件路径、capture mask、独立18086端口，但未运行正式Trace。

## queue06完成与Tracy
用户要求本聊天负责Profiling，等待另一个profile_t3_render.py批次完整结束后才启动。全过程未改共享代码或停止对方任务。

queue06_final1/final2各6000帧/8.64GB，全量像素编号/hash独立通过，57.7927/57.8102fps含fsync。pending峰2、p95=1、结束0，fsync2.515/3.193s。该结果高于前短测54fps，不能把差值全归于代码，测试环境共享负载/运行时长不同。新增有效预热检查，禁用Viewport时排除相机RenderProduct并显式开启其更新，解决本轮空缓冲启动风险。旧失败记录保留。

Tracy通过绝对路径加载carb.profiler-tracy.plugin，capture mask启用，TRACY_PORT=18086+IPv4，成功采集trace10.tracy 28.71MB，102753 GPU zones。本地构建官方v0.11.1 csvexport（附加只读GPU事件导出），未安装系统包。trace10_cpu.csv、trace10_events.csv、trace_gpu.csv与trace_summary.json保存。剔除前200 RTX区间后1419个区间，RTX Rendering均值4.574ms/p95 4.710ms；App Main loop18.171ms；waitUntilDone每次0.001826ms，两次/循环；waitForLastSubmission均值0.0874ms，四次/循环。跨线程/嵌套不能求和归因。主循环、RTX Rendering、PostRenderGraphDef::doUpdateRenderProduct各1419，未见此窗口成倍重复RenderProduct更新；不是所有内部pass逐个审计。GPU时间不足以解释主循环全部耗时，优先CPU场景同步/提交/回读阻塞进一步拆分，不能将差额直接认定纯CPU计算。该Trace为Viewport未主动关闭的独立诊断运行，不能代替最终关闭Viewport配置的精确GPU占比；日志/矩阵必须区分。Nsight CUPTI冲突仍未解决。
