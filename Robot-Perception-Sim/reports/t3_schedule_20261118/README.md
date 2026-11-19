# 调度与异步写盘

P0：实际两个RenderProduct：相机与默认Viewport；Viewport更新已关，streaming扩展为空。RTSubframes设置读回null，不据此断言1；代码未调用orchestrator.step。600帧602次world.step，7200帧7202次。无独立GPU trace，不将world.step次数当GPU渲染次数。

P1：32个有界future、单写入线程、immutable bytes，线程保存hash/完成时间，主线程最终排空/fsync。短测46.72→47.20→55.19fps；全部600帧像素编号/hash通过。queue_peak=32统计未回收future而非真实未完成任务，不能当队列积压峰值，不能据此认定长期增长。

焦距正式改1.79mm，aperture按K等比例调整至2.760003×2.070002mm，保持ROI视场；运行读回已核。配置及本地中间RGB入口已修改，服务器正式record/preview/gpu/schedule与配置已部署。未重启或覆盖用户WebRTC当前会话。

7200帧10.368GB独立全部编号/hash通过，含fsync 120.0858s、59.9571fps。不是100fps通过，也未按逐窗口门槛宣称稳定60fps。写完成相对主线程提交返回时刻最大延迟96.965ms（可能负值，线程可先完成）。

P0 gating初测触发CUDA700，不能采用；需要重新设计缓冲生命周期。P2 async启动未产出完整benchmark，退出0不能证明成功，未计提速。nsys CPU采样环境检查因perf权限失败，但CUDA trace本轮尚未运行，不可声称GPU trace不支持。

剩余：真实未完成队列时序统计、独立持续磁盘测试、CUDA时间线、异步渲染失败诊断。磁盘/home剩8.4GB，不再追加大型录制，后续须使用有充足空间卷。未删除用户数据。
