# T3 渲染/回读受控对照

每档 300 个 100 Hz 源时间样本，每样本固定 3 个渲染 tick；静态园区、既有相机内参/候选畸变、DLSS 和默认 Viewport 保持。三目包含 800×600 CenterRGB、两路 1280×720；双目不创建中心 Render Product。未声称墙钟实时 100 Hz。

时间单位 ms，三元组为平均 / P95 / 最大。GPU 时间来自 Hydra GPU profiler RTX Rendering 根节点，每渲染 tick；不是 Python 计时。利用率由 nvidia-smi 每 200ms 采样，按测量区间筛选。

|目数|档位|总时间/样本|GPU RTX 时间/tick|GPU 利用率 %|
|---|---|---|---|---|
|2|cpu|52.77 / 57.96 / 76.67|7.65 / 7.83 / 10.77|36.51 / 40.10 / 41.00|
|2|gpu|49.17 / 54.30 / 56.75|7.43 / 7.56 / 10.43|40.76 / 44.00 / 44.00|
|2|render|47.45 / 56.55 / 78.13|7.41 / 7.74 / 10.00|35.44 / 43.00 / 43.00|
|2|save|149.54 / 162.24 / 191.89|7.85 / 8.14 / 11.01|14.08 / 20.00 / 41.00|
|2|sd|124.93 / 136.06 / 159.59|7.86 / 8.12 / 11.21|17.20 / 20.00 / 43.00|
|2|sync_copy|59.72 / 67.80 / 82.52|7.78 / 8.20 / 10.70|35.02 / 40.00 / 40.00|
|3|cpu|62.54 / 67.89 / 73.23|10.10 / 10.26 / 13.60|42.11 / 45.00 / 49.00|
|3|gpu|61.54 / 79.47 / 98.45|10.18 / 10.76 / 14.30|42.60 / 47.00 / 47.00|
|3|render|97.03 / 108.53 / 161.85|12.99 / 15.57 / 17.92|33.03 / 42.00 / 45.00|
|3|render_repeat|52.50 / 56.91 / 62.04|9.87 / 10.08 / 13.34|46.74 / 48.45 / 51.00|
|3|save|178.64 / 198.33 / 226.49|10.67 / 11.11 / 14.76|14.52 / 26.00 / 47.00|
|3|sd|149.70 / 159.51 / 186.13|10.57 / 10.83 / 14.63|17.53 / 26.00 / 50.00|
|3|sync_copy|68.89 / 80.27 / 89.61|10.32 / 10.63 / 14.66|38.16 / 43.00 / 48.00|

## 分阶段耗时

|配置|阶段|平均 / P95 / 最大 ms|
|---|---|---|
|profile2_2_cpu|render_ms|48.59 / 53.83 / 71.89|
|profile2_2_cpu|get_ms|1.09 / 1.18 / 2.08|
|profile2_2_cpu|sync_ms|0.00 / 0.00 / 0.00|
|profile2_2_cpu|copy_ms|1.59 / 1.61 / 5.45|
|profile2_2_cpu|color_ms|0.00 / 0.00 / 0.00|
|profile2_2_cpu|gray_ms|0.00 / 0.00 / 0.00|
|profile2_2_cpu|sd_ms|0.00 / 0.00 / 0.00|
|profile2_2_cpu|save_ms|0.00 / 0.00 / 0.00|
|profile2_2_cpu|total_ms|52.77 / 57.96 / 76.67|
|profile2_2_gpu|render_ms|46.66 / 51.34 / 53.93|
|profile2_2_gpu|get_ms|1.06 / 1.15 / 2.33|
|profile2_2_gpu|sync_ms|0.00 / 0.00 / 0.00|
|profile2_2_gpu|copy_ms|0.00 / 0.00 / 0.00|
|profile2_2_gpu|color_ms|0.00 / 0.00 / 0.00|
|profile2_2_gpu|gray_ms|0.00 / 0.00 / 0.00|
|profile2_2_gpu|sd_ms|0.00 / 0.00 / 0.00|
|profile2_2_gpu|save_ms|0.00 / 0.00 / 0.00|
|profile2_2_gpu|total_ms|49.17 / 54.30 / 56.75|
|profile2_2_render|render_ms|45.87 / 55.19 / 76.17|
|profile2_2_render|get_ms|0.00 / 0.00 / 0.01|
|profile2_2_render|sync_ms|0.00 / 0.00 / 0.00|
|profile2_2_render|copy_ms|0.00 / 0.00 / 0.00|
|profile2_2_render|color_ms|0.00 / 0.00 / 0.00|
|profile2_2_render|gray_ms|0.00 / 0.00 / 0.00|
|profile2_2_render|sd_ms|0.00 / 0.00 / 0.00|
|profile2_2_render|save_ms|0.00 / 0.00 / 0.00|
|profile2_2_render|total_ms|47.45 / 56.55 / 78.13|
|profile2_2_save|render_ms|55.14 / 63.24 / 77.72|
|profile2_2_save|get_ms|1.10 / 1.21 / 2.21|
|profile2_2_save|sync_ms|0.00 / 0.00 / 0.00|
|profile2_2_save|copy_ms|1.60 / 1.87 / 4.57|
|profile2_2_save|color_ms|58.00 / 60.46 / 73.41|
|profile2_2_save|gray_ms|6.36 / 6.75 / 9.90|
|profile2_2_save|sd_ms|1.47 / 1.67 / 2.30|
|profile2_2_save|save_ms|22.23 / 23.20 / 59.25|
|profile2_2_save|total_ms|149.54 / 162.24 / 191.89|
|profile2_2_sd|render_ms|53.26 / 59.00 / 76.39|
|profile2_2_sd|get_ms|1.08 / 1.19 / 1.42|
|profile2_2_sd|sync_ms|0.00 / 0.00 / 0.00|
|profile2_2_sd|copy_ms|1.60 / 1.70 / 4.53|
|profile2_2_sd|color_ms|57.65 / 59.77 / 81.24|
|profile2_2_sd|gray_ms|6.32 / 6.54 / 9.66|
|profile2_2_sd|sd_ms|1.48 / 1.77 / 2.40|
|profile2_2_sd|save_ms|0.00 / 0.00 / 0.01|
|profile2_2_sd|total_ms|124.93 / 136.06 / 159.59|
|profile2_2_sync_copy|render_ms|51.28 / 58.99 / 76.70|
|profile2_2_sync_copy|get_ms|1.13 / 1.31 / 1.64|
|profile2_2_sync_copy|sync_ms|4.16 / 4.48 / 4.55|
|profile2_2_sync_copy|copy_ms|1.57 / 1.77 / 4.50|
|profile2_2_sync_copy|color_ms|0.00 / 0.00 / 0.00|
|profile2_2_sync_copy|gray_ms|0.00 / 0.00 / 0.00|
|profile2_2_sync_copy|sd_ms|0.00 / 0.00 / 0.00|
|profile2_2_sync_copy|save_ms|0.00 / 0.00 / 0.00|
|profile2_2_sync_copy|total_ms|59.72 / 67.80 / 82.52|
|profile2_3_cpu|render_ms|57.04 / 61.64 / 68.12|
|profile2_3_cpu|get_ms|1.51 / 1.56 / 3.09|
|profile2_3_cpu|sync_ms|0.00 / 0.00 / 0.00|
|profile2_3_cpu|copy_ms|1.98 / 2.03 / 5.34|
|profile2_3_cpu|color_ms|0.00 / 0.00 / 0.00|
|profile2_3_cpu|gray_ms|0.00 / 0.00 / 0.00|
|profile2_3_cpu|sd_ms|0.00 / 0.00 / 0.00|
|profile2_3_cpu|save_ms|0.00 / 0.00 / 0.00|
|profile2_3_cpu|total_ms|62.54 / 67.89 / 73.23|
|profile2_3_gpu|render_ms|58.01 / 75.50 / 93.47|
|profile2_3_gpu|get_ms|1.49 / 1.73 / 4.24|
|profile2_3_gpu|sync_ms|0.00 / 0.00 / 0.00|
|profile2_3_gpu|copy_ms|0.00 / 0.00 / 0.00|
|profile2_3_gpu|color_ms|0.00 / 0.00 / 0.00|
|profile2_3_gpu|gray_ms|0.00 / 0.00 / 0.00|
|profile2_3_gpu|sd_ms|0.00 / 0.00 / 0.00|
|profile2_3_gpu|save_ms|0.00 / 0.00 / 0.02|
|profile2_3_gpu|total_ms|61.54 / 79.47 / 98.45|
|profile2_3_render|render_ms|95.01 / 106.75 / 158.38|
|profile2_3_render|get_ms|0.00 / 0.00 / 0.02|
|profile2_3_render|sync_ms|0.00 / 0.00 / 0.00|
|profile2_3_render|copy_ms|0.00 / 0.00 / 0.00|
|profile2_3_render|color_ms|0.00 / 0.00 / 0.00|
|profile2_3_render|gray_ms|0.00 / 0.00 / 0.00|
|profile2_3_render|sd_ms|0.00 / 0.00 / 0.00|
|profile2_3_render|save_ms|0.00 / 0.00 / 0.00|
|profile2_3_render|total_ms|97.03 / 108.53 / 161.85|
|profile2_3_render_repeat|render_ms|50.51 / 54.28 / 59.52|
|profile2_3_render_repeat|get_ms|0.00 / 0.00 / 0.00|
|profile2_3_render_repeat|sync_ms|0.00 / 0.00 / 0.00|
|profile2_3_render_repeat|copy_ms|0.00 / 0.00 / 0.00|
|profile2_3_render_repeat|color_ms|0.00 / 0.00 / 0.00|
|profile2_3_render_repeat|gray_ms|0.00 / 0.00 / 0.00|
|profile2_3_render_repeat|sd_ms|0.00 / 0.00 / 0.00|
|profile2_3_render_repeat|save_ms|0.00 / 0.00 / 0.01|
|profile2_3_render_repeat|total_ms|52.50 / 56.91 / 62.04|
|profile2_3_save|render_ms|64.45 / 74.06 / 88.02|
|profile2_3_save|get_ms|1.59 / 1.70 / 3.10|
|profile2_3_save|sync_ms|0.00 / 0.00 / 0.00|
|profile2_3_save|copy_ms|1.82 / 1.95 / 5.33|
|profile2_3_save|color_ms|73.51 / 79.15 / 93.69|
|profile2_3_save|gray_ms|6.67 / 7.17 / 12.08|
|profile2_3_save|sd_ms|1.50 / 1.72 / 3.61|
|profile2_3_save|save_ms|24.81 / 25.73 / 75.82|
|profile2_3_save|total_ms|178.64 / 198.33 / 226.49|
|profile2_3_sd|render_ms|61.81 / 67.96 / 83.61|
|profile2_3_sd|get_ms|1.51 / 1.62 / 1.96|
|profile2_3_sd|sync_ms|0.00 / 0.00 / 0.00|
|profile2_3_sd|copy_ms|1.99 / 2.07 / 5.56|
|profile2_3_sd|color_ms|72.40 / 75.89 / 91.99|
|profile2_3_sd|gray_ms|6.41 / 6.65 / 10.69|
|profile2_3_sd|sd_ms|1.46 / 1.67 / 2.29|
|profile2_3_sd|save_ms|0.00 / 0.00 / 0.01|
|profile2_3_sd|total_ms|149.70 / 159.51 / 186.13|
|profile2_3_sync_copy|render_ms|57.62 / 69.31 / 78.77|
|profile2_3_sync_copy|get_ms|1.51 / 1.63 / 1.94|
|profile2_3_sync_copy|sync_ms|5.67 / 5.91 / 5.97|
|profile2_3_sync_copy|copy_ms|2.02 / 2.38 / 5.89|
|profile2_3_sync_copy|color_ms|0.00 / 0.00 / 0.00|
|profile2_3_sync_copy|gray_ms|0.00 / 0.00 / 0.00|
|profile2_3_sync_copy|sd_ms|0.00 / 0.00 / 0.00|
|profile2_3_sync_copy|save_ms|0.00 / 0.00 / 0.00|
|profile2_3_sync_copy|total_ms|68.89 / 80.27 / 89.61|

## 结论与限制

- 三目热缓存纯渲染 repeat 为有效复测；首个 render 档显著较慢，保留但不作为稳态基线。失败 smoke/warm 无有效结果，未纳入表。
- save 比 sd 同时多了数据写入与 SHA-256；另报 fsync 尾段，不能将总吞吐变化全部归因于 SD。
- sync_copy 插入设备同步，改变流水线等待；copy_ms 是 CPU 观察到的同步复制延迟，不是 CUDA event/NSight GPU DMA 时间。
- 默认更新中的 Viewport Render Product 在各档保持，因此纯渲染并非仅相机纹理工作负载。
- render/gpu 档不回读像素，不能逐帧确认像素身份；sd/save 档检查编号，save 两目/三目均独立全量通过所供 NumPy SD、文件 hash 与编号检查。
- 官方 CameraSensor CPU/CUDA 示例及 get_data 源码已核对；为保持 HDR 链路主对照未改用 CameraSensor，使用现有 Replicator CUDA HdrColor 与 .numpy()。
- CPU 颜色转换包括 FP16→FP32、帧身份检查、sRGB 与打包；gray_ms、sd_ms 另列。
- GPU 利用率是设备级粗采样，RTX Rendering timer 不包含所有 CUDA/CPU 阶段。

远端原始记录：`/var/tmp/lizhengran-t3-abc-20261118/profile2_*`；隔离运行脚本与源码：`/home/lizhengran/isaacsim-workspace/urban_street_20261112/t3_abc_20261118/`。
