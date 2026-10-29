# G5 static SDK collection and depth diagnosis

The recommended capture is **`static_sender_13` / `static_receiver_13`, SDK 5.1 NEURAL, original fine orange texture**. It contains 185 actual SDK stereo frames: 173 positive frames spanning all 15 depth/angle cells and 12 out-of-FOV/occluded negatives. Sender and receiver exited 0. Independent decoding of both delivered RGB strips, unique frame identities and SDK timestamps verified every association.

The unchanged perception evaluator detects all 173 positive frames and yields 164 valid SDK sphere estimates (94.80%). Conditional center MAE/P95/max are 0.027205/0.080996/0.118492 m. All nine invalid estimates are at 4 m/right; they remain invalid, with `sphere_fit_inconsistent`. The G5 matrix and explicit valid/invalid reporting are complete; this does not mean every SDK observation succeeds or that general deployment accuracy is established. All 12 negative frames are rejected. See `g5_acceptance.json` and `perception_baselines.md` for the complete, authoritative metric grids.

The scene uses a sphere of radius 0.11 m at optical Z = 1, 1.5, 2, 3 and 4 m, with X/Z = −0.4/0/+0.4. A textured grayscale background lies at 6 m. Each case holds for six samples, and the sequence repeats twice. The standard G3 immutable CUDA RGB publication path feeds the official ZED bridge. No renderer/custom depth enters the receiver.

The receiver directory contains SDK calibration, RGB, depth and confidence. Separate evaluator JSONL rows contain authored sphere truth and a column-vector `T_world_camera` for left optical coordinates (+X right, +Y down, +Z forward). The association script rejects duplicate source identities, duplicate or regressing receiver indices/timestamps, repeated decoded identities, missing source identities, wrong scenarios, disagreeing stereo strips and timestamps more than 2 ns from the authored publication. Ten focused association tests cover these identity checks.

## Preserved controls

| Run | Scene / SDK mode | Observation |
| --- | --- | --- |
| 07 | Fine orange, PERFORMANCE | RGB triangulation works; only 1/173 SDK sphere estimates accepted and its center error is about 0.798 m. |
| 09 | Original G3 static, PERFORMANCE | The known 2 m textured plane measures 2.015975 m; independent disparity is 22 px. Scale/calibration control passes. |
| 10 | Fine orange, ULTRA | 26/173 SDK estimates accepted, exclusively at 1 m. Larger distances still fail. |
| 11 | Coarse black/orange, ULTRA | Texture improves raw SDK correspondence in several cells, but black edge patches break the fixed HSV silhouette: only 10/173 detections. Retained as a failed diagnostic. |
| 12 | Fine orange, PERFORMANCE, raw SDK disparity | Failing sphere depth × signed SDK disparity equals −44.496002 m·px within about 4×10⁻⁶. This equals −fx×baseline: conversion/units are correct, while SDK correspondence is wrong. |
| 13 | Fine orange, NEURAL | Restores useful SDK foreground depth across the grid; 164/173 accepted without predictor threshold changes. |

Sender fx = 370.8000036 px, SDK fx = 370.7999878 px; the source and SDK baselines are 0.12 m. Fixed generic OpenCV StereoSGBM applied only to delivered grayscale image pairs independently recovers sphere surface depths where PERFORMANCE fails (for example approximately 0.917 m versus SDK 8.054 m for the 1 m sphere). This is an offline diagnostic, not a substituted SDK measurement. `scripts/diagnose_g5_sdk_depth.py` reproduces these image, background, calibration and raw SDK depth/disparity checks. Compact evidence is in `g5_sdk_depth_diagnosis.json`.

## Artifacts and environment

Remote artifacts are under `/home/lizhengran/isaacsim-workspace/g5_static/runs/`; local copies are under `results/g5/`. All receivers, evaluator labels, sender metadata/states/source snapshots, exit codes and relevant runtime manifests were copied. Large sender composites/subframes remain remote. Exact calibration, camera asset, source, C++ binary, bundled model and optimized engine hashes are recorded in `g5_static_capture.json` and the local runtime manifests. Receiver 10/11 source snapshots were reconstructed from the applied patch; their binary hashes were not recorded contemporaneously. Runs 07/09 share the unchanged read-only G3 receiver; runs 12/13 have recorded binary hashes.

The successful sender used physical GPU 0 and the receiver physical GPU 1. After teardown there were no owned GPU processes, and observed memory returned to 148/16 MiB. No unrelated jobs were stopped. An independent writable copy of the 1.3 GiB G3 rootfs restored the known working userspace/configuration; G3 and G2 dependencies remain mounted read-only. The initial read-only rootfs plus fresh private-home configuration stalled at Isaac's viewport-handle wait. A temporary viewport bypass produced no render packets and was removed. The separate effects of rootfs writability and inherited user configuration were not isolated.

Actual runtime versions: SDK 5.1.0, TensorRT `getInferLibVersion()` = 100900 (10.9.0), NVIDIA driver 575.57.08, CUDA driver API 12090 (12.9); existing compile headers report CUDA 12040 (12.4). Both the receiver and preparation utility compile against the existing SDK 5.0.7 and 5.1.0 headers/libraries. Actual NEURAL data here uses SDK 5.1. PERFORMANCE remains the receiver default; optional requested modes are validated against SDK readback. Invalid mode and disparity arguments exit 1 before output creation or SDK open.

## Reproduction

Copy the source scripts/acquisition package into the isolated G5 directory first. Prepare the rootfs once; do not overwrite an existing checkout or rootfs:

```sh
cp -a --reflink=auto /home/lizhengran/isaacsim-workspace/g3_acquisition/ubuntu22-rootfs \
  /home/lizhengran/isaacsim-workspace/g5_static/ubuntu22-rootfs
cd /home/lizhengran/isaacsim-workspace/g5_static
mkdir -p runs sdk-neural
cp -a /home/lizhengran/isaacsim-workspace/g3_acquisition/sdk-5.1.0-unpacked/resources sdk-neural/
ln -s /mnt/sdk-5.1.0-unpacked/lib sdk-neural/lib
ln -s /mnt/sdk-5.1.0-unpacked/dependencies sdk-neural/dependencies
bash run_g5_in_ubuntu22.sh cmake -S /poc/receiver_source -B /poc/receiver_build \
  -DZED_SDK_ROOT=/mnt/sdk-5.1.0-unpacked -DCUDA_INCLUDE_DIR=/mnt/cuda_headers
bash run_g5_in_ubuntu22.sh cmake --build /poc/receiver_build -j4
```

The bundled 27 MiB neural model was already present. Optimize it before launching the sender, using writable G5 resources; this utility refuses an implicit model download. TensorRT creates a roughly 16.2 MB GPU-specific engine once:

```sh
env CUDA_VISIBLE_DEVICES=GPU-3c0a37f5-27f6-5b9e-64a4-7c59aadf0cd2 \
  ZED_SDK_PATH=/poc/sdk-neural \
  LD_LIBRARY_PATH=/mnt/sdk-5.1.0-unpacked/lib:/mnt/sdk-5.1.0-unpacked/dependencies/ai_dependencies/lib \
  bash run_g5_in_ubuntu22.sh /poc/receiver_build/prepare_g5_neural

env G5_SDK_PATH=/poc/sdk-neural G5_RECEIVER_BINARY=/poc/receiver_build/zed_sdk_receiver \
  G5_DEPTH_MODE=NEURAL G5_RECEIVER_TIMEOUT_SECONDS=600 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  bash run_g5_trial.sh NEW_SENDER NEW_RECEIVER g5_static 185 0 204 1 30050 ipc
python3 associate_g5_evaluation.py --sender runs/NEW_SENDER \
  --capture runs/NEW_RECEIVER --output runs/NEW_EVALUATION.jsonl
```

For local offline diagnosis, use the repository virtual environment:

```sh
.venv/bin/python Robot-Perception-Sim/scripts/diagnose_g5_sdk_depth.py \
  --experiment Robot-Perception-Sim/results/g5/static_sender_07 \
    Robot-Perception-Sim/results/g5/static_receiver_07 Robot-Perception-Sim/results/g5/evaluation07.jsonl \
  --plane Robot-Perception-Sim/results/g5/diagnostic_plane2_sender_09 \
    Robot-Perception-Sim/results/g5/diagnostic_plane2_receiver_09 \
  --disparity-control Robot-Perception-Sim/results/g5/diagnostic_receiver_12 \
  --report Robot-Perception-Sim/results/g5/reproduced_diagnosis.json
```
