# G6-A actual SDK acquisition

G6-A renders the complete frozen G1 K1 scene with the official ZED X attached to the existing head optical frame, and sends its paired RGB through the verified G3 immutable CUDA/official bridge path into ZED SDK 5.1 NEURAL. It replays the existing G4 `direct_middle_center` and `bounce_one` records; it runs no new collision response, walking, controller, training or policy. The ball retains the G1 plain-white appearance. G1's original checker texture was not transferred into that USD asset.

The eight conditions are two frozen trajectories × two head poses (`neutral`, `yaw_pitch` = +0.2/+0.1 rad) × 0/10 ms exposure. The mount remains an explicit existing virtual assumption, not measured hardware extrinsics. Mount and visual-source audits are in `g6a_mount_acceptance.json` and `g6a_visual_fidelity_acceptance.json`.

The final corpus contains **368 actual SDK stereo frames, all 128 intended target frames, and 384 verified target subframes**. Targets are planned replay images, including independently labeled out-of-FOV/occluded states. All eight successful runs exited sender 0/receiver 0; every target ID was delivered once. Maximum verified RGB transport MAE is 0.648233/255 and maximum nearest-state time error is 333.344 microseconds. Source, calibration and publication checks are acquisition evidence, not detector accuracy.

## Time and startup contract

Both exposure conditions use the same physical midpoint list: 5 ms plus integer 1/60 s increments, followed by `duration−5 ms` if distinct. This final shorter interval includes an endpoint-adjacent window without extrapolating the frozen record. Direct goal has 17 target images and one-bounce has 15, for 128 intended target images across eight conditions.

For 0 ms, render the nearest frozen 1 ms state at each midpoint. For 10 ms, render five equal midpoint-quadrature samples across the surrounding ±5 ms window, choosing the nearest frozen state independently for each sample. No interpolation crosses a collision, and no new physical response is generated. Save every requested time, realized time, sample index, position, velocity, quaternion and quantization error. Average the rendered encoded uint8 RGBA samples; this is the existing G3 approximation, not radiometric sensor/ISP equivalence.

Native World/render-reference clocks, SDK publication-window timestamps and frozen physical shot time are separate and recorded explicitly. The final physical image interval can be shorter than 1/60 s. Transport publication is replayed at a 0.2 s wall period; these captures make no real-time or latency claim.

Every trial prepares 30 instantaneous fixed-start pre-roll frames, its complete target sequence, and 60 instantaneous fixed-end post-roll frames. Before the first target is published, require SDK receiver metadata with actual `NEURAL` readback and at least three completed receiver grabs. Request pre-roll-count + target-count SDK frames; post-roll provides delivery margin. Pre/post-roll remain in raw SDK files and evaluation labels but are excluded from target metrics. Completeness requires every intended target visual ID to appear exactly once in the delivered SDK images. The intended target sequence is archived before replay in `source_observations.jsonl`.

One-bounce/neutral/10 ms attempt01 failed during SDK open with repeated metadata-size mismatch messages. The unchanged ready guard stopped before publishing any target; receiver exited 124 and sender 1. It is retained as a failed stream-initialization attempt, excluded from successful capture counts. The retry uses a fresh stream port; remaining runs also use distinct ports. No incomplete target sequence is converted into a successful observation. A separate yaw/direct0 attempt used odd port 30065; SDK rejected it before open, and it also published no targets. Its successful retry used 30066. Failed logs, source records and exit codes are archived separately.

## Rendered diagnostic IDs

The inherited diagnostic strips at 1.99 m could be hidden by a sphere approaching the head camera. G6 therefore uses separate per-eye near-clip strips; the other eye's strip lies outside its frustum. Their fixed image rectangles are `[20,20,296,32]` and `[20,45,296,57]`. Centers are recomputed from actual USD camera transforms and calibration. Both strips are actual rendered geometry; no SDK or source pixels are painted afterward.

Use literal per-cell emissive materials to prevent moving-ball/rig shadows from changing bit contrast. Pixel-down corner order initially made the faces back-facing. Archived attempts04 and05 differ only by reversing face winding;04 rendered black strips despite accepted emissive inputs, while05 delivered valid IDs and completed. The source diff and raw failures are preserved in `results/g6a/instrumentation_diagnostics/`. The earlier connected-primvar experiment was also back-facing, so its independent support was not established.

The image-only G6 detector may exclude these fixed instrumentation rectangles, with their declared margin. This is independent of ball position/GT. Capture pixels remain intact; candidates touching that boundary are not reconstructed.

## Source visual intent

G1 authored ten fully transparent visual/collider gprims. In this renderer, the alpha-zero ground plane remained opaque and covered the intended green turf. An independent A/B test changed only that plane's visibility: the gray median approximately 97/97/97 became 144/190/160. The final G6 visual layer marks exactly the ten explicitly authored alpha-zero gprims invisible, preserving all original colors, materials, transforms, geometry and G1 file bytes. Paths and reason are recorded in scene provenance. Earlier gray-field runs remain diagnostic/source-version evidence and are excluded from the intended-scene final corpus.

## Artifacts and verification

All work is isolated under `/home/lizhengran/isaacsim-workspace/g6a_head_replay`. G1/G3/G4 and the installed Isaac/SDK dependencies are mounted read-only; the rootfs, shader cache, neural resources/engine copy and outputs are G6-owned. Sender uses physical GPU0, receiver GPU1. No other jobs are stopped.

Local `results/g6a/collection_manifest.json` identifies every successful source/receiver pair, evaluator JSONL, source sequence and integrity report. Raw SDK left/right PNGs and float32 depth/confidence are copied without filtering or GT fill. All source composites and subframe PNGs, source states, publication records, source snapshots and independent visibility evidence are copied locally. Authored truth is joined only after image prediction inputs have been kept separate.

`verify_g6_capture.py` independently checks source hashes, calibration, rendered packet/output/subframe IDs, exact encoded composite means and immutable publication hashes, nearest frozen-state selection, SDK timestamp association, raw payloads, causal host clocks, delivered RGB versus source composites and complete target delivery. `associate_g6_evaluation.py` invokes that verification before producing evaluation labels. Invalid SDK depth and invalid detections remain evidence; acquisition integrity is not a perception-accuracy pass.

Replay one condition on the prepared remote workspace with new output names and an unused port:

```sh
bash run_g6a_trial.sh NEW_SENDER NEW_RECEIVER direct_goal 10 neutral 30070
```

The runner fixes the full G1 scene, official camera, archived G4 trajectory, mount config and SDK NEURAL mode. Source/receiver file hashes are frozen in `runs/formal_source_freeze_sha256.txt`. For offline verification and label construction:

```sh
.venv/bin/python Robot-Perception-Sim/scripts/associate_g6_evaluation.py \
  --source SOURCE_DIR --capture RECEIVER_DIR \
  --trajectory SOURCE_DIR/sources/frozen_g4_trajectory.json --output NEW_EVALUATION.jsonl
```

Continuous perception and 0/10 ms photometric comparisons are reported separately. In particular, correct camera/source integrity and changed exposure pixels do not imply that either frozen white-ball detector succeeds.

After the corpus was complete, the repository wrapper gained only preflight validation requiring an even integer port 1024–65534. Five negative shell cases reject with exit 2 before any external command can spawn; seven replay/handshake tests also pass. The executed wrapper remains archived as `results/g6a/runtime/run_g6a_trial.executed.sh`, SHA256 `69c01279bf1876f9dfab372e6d9ed0e17494d22dd65c2878bb18d76b2835fae7`. The final repository wrapper has a distinct recorded hash; this validation-only change did not alter any captured camera/source/receiver pixels.

Final cleanup found no owned capture/driver or CUDA compute processes; GPU0/1 memory returned to 149/16 MiB. All pre-existing G1–G5 data and unrelated custom-stereo/ROS files remain untouched. `g6a_acquisition.json` records the final condition manifest, hashes, retries and cleanup.
