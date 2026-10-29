# G6-A continuous perception evaluation

This stage uses the G6-only `g6_white_circle_mvp` detector on the original plain-white G1 ball in the full K1 field scene. The orange G5 detector/module is untouched. G6 reuses the existing G5 rectified `triangulate` and `fit_sdk_sphere` functions with their valid/invalid rules unchanged. Continuous evaluation does not add a tracker, temporal smoothing, GT-filled detections or predicted position recovery. Every complete SDK image is processed independently, and valid/invalid observations are retained throughout the sequence.

Before inspecting any actual white-ball capture, the MVP thresholds were frozen: OpenCV HSV saturation ≤60 and value ≥180; contour area ≥6 px²; circularity ≥0.70; bounding-box aspect ratio 0.65–1.5; contour-area/bounding-box-area ratio 0.45–0.90. The extent filter rejects filled rectangular distractors, and aspect/roundness reject long field lines. Two comparable accepted contours (second area greater than half the largest) return `ambiguous_contours`; an accepted contour touching the image boundary returns invalid. Confidence is contour circularity, not a calibrated probability. No GT crop, position, visibility, robot pose or perfect object label enters selection. Thresholds, detector provenance and hashes of prediction/evaluation source files are saved in every run summary.

The known diagnostic pixel-ID strips are excluded as fixed instrumentation: image rectangles `[20,20,296,32]` and `[20,45,296,57]`, each expanded by exactly 2 px. These source-instrument coordinates are not derived from GT, pose, detector output or ball position. Exclusion happens before contours and is recorded in detector provenance. A selected candidate touching an excluded-region boundary returns `diagnostic_overlap`; no ball pixels are reconstructed inside the excluded area. If masking destroys roundness and the remaining white component touches the boundary, the absence of another candidate also returns this diagnostic failure. This authorized instrumentation exclusion changes no numerical white/shape/geometry thresholds. Synthetic tests verify that marker glyphs disappear while a separate ball remains detectable, and a ball overlapping the strip remains invalid.

This simple image detector is not a general football recognizer. A white ball merged with a white field line can return `no_ball_contour`; white rounded robot/goal components can cause ambiguity. Partial occlusion and clipping may invalidate shape support. Such failures remain measurements and will not trigger threshold tuning to improve the actual-data pass rate. Synthetic tests cover a white sphere beside larger line/rectangle distractors, comparable circular distractors, clipping, half/full occlusion, a connected-line failure, orange rejection and unchanged G5 depth invalids.

The robot remains fixed, with the ZED attached to its actual head mount. Frozen G4 direct-goal and one-bounce physical state sequences are replayed at their original physical times. The replay creates the rendered trajectory; it is not a new dynamic contact simulation or evidence that the fixed robot walks. Oracle camera poses are evaluation inputs only.

Run the CLI with one or more capture/label/source-sequence triples:

```sh
python scripts/run_continuous_perception.py \
  --capture results/g6a/direct_0/receiver \
  --evaluation results/g6a/direct_0/evaluation.jsonl \
  --source-sequence results/g6a/direct_0/source_observations.jsonl \
  --capture results/g6a/direct_10/receiver \
  --evaluation results/g6a/direct_10/evaluation.jsonl \
  --source-sequence results/g6a/direct_10/source_observations.jsonl \
  --output results/g6a/perception
```

Repeat the triple for the one-bounce conditions. Output must not exist. Receiver metadata, rectified PNGs, little-endian SDK depth/confidence arrays and frame JSONL use the unchanged G5 capture contract. Metadata must report frame rate, calibration, SDK mode and metres. `run_NNN/predictions.jsonl` and left/right mask PNGs are written before source labels are opened. `run_NNN/evaluation.jsonl` then joins predictions with source records, saving independent visibility/timing diagnostics and separately labeled oracle-world estimates. The top-level summary and CSV aggregate by trajectory, fixed head-pose condition and exposure. Exposure pairs never mix head-pose conditions.

Evaluation JSONL must contain exactly one row per delivered receiver frame, including pre/postroll tagged by `segment` or source `phase` (`target` is the default). Each target label carries:

- `frame_index`, `source_frame`, `sequence_id`, `trajectory_name`, `head_pose`, `exposure_ms`;
- `source_time_s`, the actual associated exposure midpoint, and `trajectory_time_s`, the physical replay midpoint used to pair exposure conditions;
- `ball_center_camera_m`, `ball_world_position_m`, `T_world_camera`, and `ball_radius_m`;
- independent `expected_visible`, `visibility_basis`, `visibility_reason`, and optionally exposure-level `any_visible`, `all_visible`, subframe times/visibility/evidence;
- `timing_valid`, supported separately by the both-eye pixel-identity and SDK timestamp association verifier.

`source_observations.jsonl` contains the complete intended **target** observation sequence with the same source identity, midpoint, pose, center and visibility fields. Pre/postroll is excluded from target denominators. This independent sequence establishes missing prefix and suffix deliveries, which cannot be inferred from delivered receiver frames alone. Intended-but-undelivered rows have null `frame_index`, `observation_missing: true`, invalid method outputs and no invented camera/world estimate. These are evaluation records of missing observations, not synthesized predictions. If no intended sequence is supplied, the report can detect interior source-ID/time gaps but cannot establish prefix/suffix completeness.

Midpoints must strictly increase within each run/sequence; repeated replays require distinct `sequence_id` values. `source_frame` identifies the actual image dispatch, not merely the newest simulator state available to a callback. Known source rows and delivered labels must agree on their midpoint. Camera/world GT center labels must agree under the true rigid `T_world_camera`; inconsistency is rejected.

Exposure comparisons require identical physical midpoint schedules. For a target midpoint `t`, zero exposure samples at `t`, while 10 ms spans `[t−5 ms,t+5 ms]`; a comparison shifted by 5 ms is reported as unmatched rather than attributed to blur. Subframe samples must preserve the frozen physical trajectory and true camera pose. The source report must describe interpolation/nearest-state sampling and retain individual subframe times. The source `expected_visible` flag is independent midpoint support. Finite-exposure evaluation uses source `any_visible` (any sample with visible support), while source `all_visible` records visible support at every subframe. The evaluator preserves that flag as `all_subframes_have_visible_support` and separately derives fully visible silhouette support by checking source `fully_in_fov`, physical-mesh blocked-ray counts, near-clipped-ray counts and partial-occlusion evidence. Thus every subframe having some support does not imply a completely visible silhouette. The exact source rule and occlusion evidence belong in `visibility_basis`/`visibility_evidence`. An analytic FOV test alone does not prove absence of body occlusion. Partial visibility is reported separately. Pixel-identity marker geometry must not unexpectedly cover the ball ROI.

Metrics distinguish these quantities:

- Evaluator-only ball association uses frozen bbox IoU ≥0.50. A supplied `renderer_ball_bbox_xyxy` is preferred. Zero exposure otherwise uses the independent midpoint camera-center/radius sphere projection. Finite exposure uses the union of sphere bounds projected at independently visible subframes, consistent with the any-subframe visibility denominator; midpoint and union IoU are both retained. If visible subframe GT is absent, the report explicitly labels a midpoint fallback. The IoU threshold remains 0.50. Source labels never choose the runtime contour. Raw white-candidate validity is reported separately from matched-ball detection rate. A valid white distractor elsewhere in the scene counts as a false positive and a missed ball (`wrong_object_detection`), rather than a successful detection.
- SDK/triangulation algorithm-valid rates and ball-matched 3D valid rates are separate on timing-eligible expected-visible frames; all-intended-visible rates also include missing/timing-invalid deliveries in their denominator. A matched 2D contour does not prove that a valid stereo/depth estimate is accurate.
- Camera/world center MAE, P95 and maximum for **all algorithm-valid** visible estimates, including wrong-object predictions; explicitly labeled ball-matched subsets are reported separately. Association failures are not silently removed from the all-valid error statistics. World positions use the saved oracle pose. Rigid-transform consistency means camera/world Euclidean errors should agree; this does not validate a visual pose estimator.
- Estimate-center projection versus GT-center projection, and detector centroid versus analytic GT sphere-silhouette centroid.
- Estimate-center projection versus the detected centroid, saved as a residual with `projection_to_detection_is_independent_accuracy_check: false`. Triangulation is algebraically consistent with its input centroid, so its near-zero residual cannot validate depth.
- Algorithm reasons categorized as detection, stereo, depth or sphere fit. Independently labeled FOV/occlusion visibility and association/timing failures remain diagnostics, not rewritten detector failure reasons. An invisible frame can still produce a false detection; this is counted explicitly.
- Continuous visible-observation loss intervals, separated into algorithm-invalid, wrong-object detection, timing-invalid and intended-but-undelivered streaks. A visible white distractor does not end actual-ball loss. Duration uses actual adjacent source midpoints, capped by a supplied `observation_interval_end_s` (especially for a short trajectory's final partial interval). If the last interval end is unavailable, one nominal terminal frame period is explicitly marked `terminal_support_assumed`; the exact first-to-last invalid-midpoint span is also saved separately. Unexpected source-ID/time gaps split algorithm streaks and are listed separately as unobserved intervals. Unknown gaps are not converted into fictitious algorithm-invalid frames. Loss runs reset at independently invisible frames; FOV/occlusion counts remain available separately.

These tests are not hardware/SDK evidence: synthetic checks cover oracle transform and projection distinctions, absent prefix/suffix delivery, unknown interior gaps, invalid timing, independent visibility/occlusion categories, 0/10 ms midpoint pairing, rejected inconsistent labels and byte-identical predictions after GT-only changes. Actual captures and measured condition tables must be attached separately before claiming G6-A experimental results.

The independent visibility source currently supplies analytic silhouette/frustum and nine-ray occlusion evidence, rather than renderer instance segmentation. This is an offline geometric visibility rule, not a sensor visibility estimate or a perfect pixel mask. Partial occlusion and bbox-only association retain limitations, especially where another white object overlaps the projected ball. The report must preserve the exact evidence and qualify conclusions accordingly.

Current explicit source reasons, including `partially_near_clipped`, are preserved. Only a legacy `occluded` reason whose every sampled ray is `target_behind_near_clip_plane` and has no physical mesh blockers is normalized to `near_clip_or_camera_sphere_intersection`, with the original source reason retained. This separates camera/target near-plane geometry from physical body occlusion without modifying the frozen source scene or detector.

## Pre-correction diagnostic: direct goal, neutral head, 0 ms

The initial capture is retained as a pre-correction diagnostic, not final intended-scene acceptance evidence. The final eight conditions are collected separately after source appearance/opacity verification. The diagnostic capture is `results/g6a/direct_neutral_e0_receiver05`, with its associated evaluation and complete sender source sequence. The evaluator processed all 47 delivered raw frames and retained pre/postroll predictions. The target denominator contains exactly 17 intended and delivered observations; there are no missing deliveries or invalid timing associations.

Independent source visibility labels identify nine target frames with visible support (eight fully visible and one partial), one near-clip/camera-sphere-intersection frame, and seven out-of-FOV frames. The frozen white MVP produces no candidate or matched ball on any of those nine visible frames. All 17 target image predictions return `no_ball_contour`; triangulation and SDK sphere estimates are invalid, with no GT or cross-method fill. Center and reprojection errors are unavailable, not zero.

Actual SDK image inspection shows the plain white, shaded sphere against a bright grayscale sky/field. The low-saturation/high-value mask can merge the bright sphere portion with the background and omit its shaded surface, violating the frozen contour-shape assumption. This is an image-detector domain failure; it does not prove that the camera failed to render or stream the ball. The detector thresholds remain unchanged after inspection.

The visible actual-ball loss interval runs from physical midpoint 0.0050000002 s through support endpoint 0.1550000074 s: 0.1500000071 s. Its first-to-last invalid midpoint span is 0.1330000063 s. These durations use the actual frozen source times and declared interval endpoint, with no terminal-period assumption or unobserved gaps. This condition establishes the continuous camera/SDK/association/evaluation path and exposes the MVP perception failure; it does not establish a successful 3D ball-position baseline.

Artifacts: `results/g6a/direct_neutral_e0_perception05/{summary.json,metrics.csv,run_000/predictions.jsonl,run_000/evaluation.jsonl,run_000/masks/}`. Remaining exposure/head/trajectory conditions must be reported separately before drawing comparison conclusions.

## Separate edge-circle MVP architecture

After retaining the white-HSV failure control, a separate image-only `g6_edge_circle_mvp` architecture was authorized. Select it explicitly with `--detector edge_circle`; the default `white_hsv` control and its numerical thresholds remain unchanged. This is an architecture change, not a claim that the original white mask detector succeeded.

Before evaluating actual images with this architecture, ordinary parameters were frozen: grayscale Gaussian blur 5×5, sigma 1.5; Hough gradient circles with dp=1, minimum center distance 30 px, Canny high threshold 100, accumulator threshold 30, radius range 6 px through image-height/2. Candidate scores measure Canny(50,100) edge support at 72 perimeter angles within a ±2 px radial band. At least 50% support is required. Ranking uses only this image edge support; a second candidate scoring at least 90% of the first produces `ambiguous_circles`. Clipped circles and circles intersecting the fixed instrumentation exclusions are invalid. The output circle mask then goes through the unchanged G5 triangulation/sphere-fit rules. GT never sets a circle center, radius range, crop or candidate identity.

Synthetic tests pass for a shaded white sphere against a bright background with field-line/rectangle distractors, blank/straight-line negatives, comparable-circle ambiguity, clipping and instrumentation overlap. These synthetic results do not establish actual-scene detection accuracy.

The first actual direct-goal/neutral/0 ms edge run also fails: zero of nine independently visible target frames has a supported circle, and all 17 target predictions return `no_circle_support`. No triangulation/SDK center is fabricated. The visible loss remains 150.000 ms and position-error statistics remain unavailable. An independent full-image check on SDK frame 15 finds zero raw Hough proposals with the frozen parameters, despite 4,168 Canny edge pixels elsewhere in the image. No threshold loosening or parameter search was performed after this result. The separate report is `results/g6a/direct_neutral_e0_edge_perception05`; the original HSV report remains intact.

## Separate exposure-image diagnostic

`compare_exposure_images.py` compares source RGB composites at identical realized physical trajectory midpoints, separately for each fixed head pose and both eyes. It rejects mismatched physical schedules, frozen trajectory hashes, exposure settings or camera poses. This diagnostic does not run inside either detector and does not supply image crops or positions to prediction.

```sh
python scripts/compare_exposure_images.py \
  --baseline-sender results/g6a/direct_neutral_0_sender \
  --finite-sender results/g6a/direct_neutral_10_sender \
  --output results/g6a/exposure_image_comparison.json
```

The report saves absolute 8-bit RGB-channel MAE globally, after the same fixed instrumentation exclusion. It additionally projects the known sphere geometry at all visible exposure subframes in both conditions, takes the union of their bounding-box regions and reports RGB MAE within that **GT-assisted offline ball ROI**. This is an evaluator diagnostic, not a detected mask or inferred ball position. Pixel samples are counted once in overlapping boxes. Near-camera intersections without a finite sphere silhouette have no ROI estimate.

Outside-ball-ROI MAE is also retained as a background/render-noise comparison. Aggregates are pixel weighted and per-midpoint values remain available. These measurements establish source-image differences between 0/10 ms conditions even if the detectors fail; differences alone do not establish physically accurate shutter response, improved SDK depth or successful ball perception. The exposure sampling/composite integrity verifier supplies the separate evidence that source images use the recorded subframe samples and weights.

Finite-exposure association uses only offline source subframe geometry and does not alter either frozen detector. Center/world/reprojection accuracy still compares against the quantized physical midpoint GT, including every algorithm-valid visible prediction even if association fails. The exposure union is an approximate swept bounding box, not an exact rendered blur mask; nearest frozen-state time errors and partial visibility remain in each record. A window can contain visible pixels while its midpoint sphere is offscreen, so image association and midpoint position error intentionally remain distinct measurements.

## Final corrected-scene eight-condition results

The final collection uses the corrected intended scene appearance/opacity path, original plain-white G1 ball, full fixed K1 scene and its actual head-mounted stereo rig. All eight conditions use the same frozen source versions recorded in `docs/g6a_perception_results.json`; the older gray/pre-correction attempts remain diagnostic artifacts and are excluded from these totals. Failed receiver-open attempts stopped before target publication and remain separate logs, rather than being counted as completed conditions.

The final dataset contains **368 raw SDK frames**, **128/128 intended target observations delivered**, and **384 target render subframes**. Every target has verified both-eye pixel identity/timestamp association, with zero missing target deliveries and zero timing-invalid targets. Zero-exposure conditions contain 64 target renders; 10 ms conditions contain 320 target renders (five samples each). There are 56 any-subframe-visible observations across the eight conditions: 27 at 0 ms and 29 at 10 ms. The finite window can show a ball at its edge even when the midpoint is out of FOV or near-clipped; this is why the visibility denominators differ.

The source `expected_visible` flag describes midpoint support, while `any_visible`/`all_visible` describe exposure sample support. Final evaluation explicitly uses `any_visible` for finite-exposure denominators and preserves the midpoint flag separately. Finite-exposure ball association uses the visible-subframe projected sphere-bound union, with unchanged IoU ≥0.50; both union and midpoint IoU are retained. Position errors still compare against quantized physical midpoint GT. Zero exposure retains the midpoint association rule.

Both frozen MVP architectures have low recall in this actual scene. White HSV yields 2/56 matched detections and SDK centers; edge circles yield 4/56. Successful centers occur only in zero-exposure bounce clips. No valid center exists in any direct-goal or 10 ms condition, so their position errors are **unavailable**, never zero. All algorithm-valid estimates happen to match the ball in this dataset; the evaluator nevertheless retains wrong-object estimates in all-valid errors and reports raw/matched rates separately.

### Frozen white HSV control

| Trajectory | Head | Exposure ms | Visible | Matched 2D | Tri 3D | SDK 3D | SDK world MAE / P95 mm | Max visible SDK loss ms |
|---|---|---|---|---|---|---|---|---|
| direct_goal | neutral | 0 | 9 | 0/9 | 0/9 | 0/9 | unavailable | 150.0 |
| direct_goal | neutral | 10 | 10 | 0/10 | 0/10 | 0/10 | unavailable | 167.0 |
| direct_goal | yaw_pitch | 0 | 9 | 0/9 | 0/9 | 0/9 | unavailable | 150.0 |
| direct_goal | yaw_pitch | 10 | 9 | 0/9 | 0/9 | 0/9 | unavailable | 150.0 |
| one_bounce | neutral | 0 | 4 | 1/4 | 1/4 | 1/4 | 46.9 / 46.9 | 34.0 |
| one_bounce | neutral | 10 | 5 | 0/5 | 0/5 | 0/5 | unavailable | 83.0 |
| one_bounce | yaw_pitch | 0 | 5 | 1/5 | 1/5 | 1/5 | 55.4 / 55.4 | 50.0 |
| one_bounce | yaw_pitch | 10 | 5 | 0/5 | 0/5 | 0/5 | unavailable | 83.0 |

Across all eight conditions, SDK center error is MAE 51.14 mm, P95 54.94 mm, maximum 55.36 mm, **conditional on only 2 valid samples**. Triangulation has 2 valid samples, with MAE 14.00 mm, P95 14.55 mm and maximum 14.61 mm. These tiny selected subsets do not establish reliable centimetre-level continuous localization.

### Frozen edge-circle architecture

| Trajectory | Head | Exposure ms | Visible | Matched 2D | Tri 3D | SDK 3D | SDK world MAE / P95 mm | Max visible SDK loss ms |
|---|---|---|---|---|---|---|---|---|
| direct_goal | neutral | 0 | 9 | 0/9 | 0/9 | 0/9 | unavailable | 150.0 |
| direct_goal | neutral | 10 | 10 | 0/10 | 0/10 | 0/10 | unavailable | 167.0 |
| direct_goal | yaw_pitch | 0 | 9 | 0/9 | 0/9 | 0/9 | unavailable | 150.0 |
| direct_goal | yaw_pitch | 10 | 9 | 0/9 | 0/9 | 0/9 | unavailable | 150.0 |
| one_bounce | neutral | 0 | 4 | 2/4 | 1/4 | 2/4 | 51.2 / 54.5 | 34.0 |
| one_bounce | neutral | 10 | 5 | 0/5 | 0/5 | 0/5 | unavailable | 83.0 |
| one_bounce | yaw_pitch | 0 | 5 | 2/5 | 2/5 | 2/5 | 48.1 / 54.7 | 50.0 |
| one_bounce | yaw_pitch | 10 | 5 | 0/5 | 0/5 | 0/5 | unavailable | 83.0 |

Across all eight conditions, SDK center error is MAE 49.67 mm, P95 55.36 mm, maximum 55.45 mm, **conditional on only 4 valid samples**. Triangulation has 3 valid samples, with MAE 69.68 mm, P95 100.03 mm and maximum 103.80 mm. These tiny selected subsets do not establish reliable centimetre-level continuous localization.

### Exposure comparison and failures

| Detector | Exposure ms | Visible observations | Matched detections | Matched SDK centers |
|---|---|---|---|---|
| white_hsv | 0 | 27 | 2 | 2 |
| white_hsv | 10 | 29 | 0 | 0 |
| edge_circle | 0 | 27 | 4 | 4 |
| edge_circle | 10 | 29 | 0 | 0 |

Visible-frame failures are dominated by image detection: 54 `no_ball_contour` misses for white HSV and 52 `no_circle_support` misses for edge circles. Downstream triangulation/SDK invalids largely inherit the absent detection (`missing_valid_stereo_detection` / `missing_valid_left_detection`); these are not measurements of camera geometry or depth accuracy. There are no wrong-object false positives in the completed target dataset. Source diagnostics separately contain 69 out-of-FOV windows without visible exposure support, three full near-clip/camera-sphere intersections, one partially visible exposure with a near-clipped midpoint and one partially visible exposure with an out-of-FOV midpoint. No physical body-occlusion case is established by these selected clips. The independent nine-ray geometric rule is finite and omits non-Mesh primitives; it is not perfect segmentation or a sensor visibility estimate.

Loss duration uses actual adjacent realized physical midpoints and the declared frozen-record endpoint; there are no unknown time gaps and no terminal-period assumptions in the final records. The largest visible loss is 167.0 ms in the neutral direct-goal 10 ms clip. The bounce SDK loss grows from 34.0 ms (neutral, 0 ms) / 50.0 ms (yaw-pitch, 0 ms) to 83.0 ms at 10 ms for both poses. These are separate finite replay windows, not a continuous robot-motion trial. The physical bounce occurs after the ball has left this head-camera view, so this does not demonstrate visually observed bounce tracking.

Midpoint GT-center projection MAE is 0.943 px (HSV triangulation, n=2), 3.993 px (HSV SDK, n=2), 2.230 px (edge triangulation, n=3) and 3.975 px (edge SDK, n=4). World poses remain oracle. Estimate-to-detected-pixel residuals are saved separately and are not independent position evidence; triangulation is self-consistent with its input centroid by construction.

### Exposure pixel-change evidence independent of detector success

All four 0/10 ms source-composite comparisons use identical physical midpoint schedules: 64 paired midpoints in total. Fixed diagnostic-strip regions are excluded. Values below are absolute 8-bit RGB MAE, shown left/right; the ball ROI is a GT-assisted offline union of projected visible-subframe sphere boxes and is never detector input.

| Trajectory | Head | Paired midpoints | Global MAE L / R | GT ball ROI MAE L / R | Outside-ROI MAE L / R | ROI-supported pairs L / R |
|---|---|---|---|---|---|---|
| direct_goal | neutral | 17 | 0.3818 / 0.3562 | 2.6479 / 2.2830 | 0.0581 / 0.0570 | 9 / 9 |
| direct_goal | yaw_pitch | 17 | 0.3004 / 0.3706 | 2.2725 / 1.5390 | 0.0684 / 0.0682 | 9 / 9 |
| one_bounce | neutral | 15 | 0.0986 / 0.0939 | 6.5254 / 6.1265 | 0.0656 / 0.0627 | 5 / 5 |
| one_bounce | yaw_pitch | 15 | 0.1313 / 0.1280 | 7.7739 / 7.6013 | 0.0785 / 0.0770 | 5 / 5 |

One neutral-direct visible window has no finite sphere projection because of camera/sphere near intersection. Its ball ROI is unavailable; global MAE remains measured. Background-only comparisons also exclude every window whose sphere straddles the camera plane, even if the finite nine-ray visibility rule reports invisible, because such a projected ROI is unbounded. Any available finite-subframe ROI remains explicitly partial in those windows. The ROI MAE exceeds the outside-ROI comparison for every pair, showing actual pixel changes around the moving ball even though the MVP detectors mostly fail. Rendering noise and approximate bounding boxes remain part of this diagnostic. The separate exposure-integrity checks establish the recorded subframe averaging; RGB MAE alone does not validate shutter physics, improve SDK depth or constitute successful ball perception.

Final machine-readable compact report: `docs/g6a_perception_results.json`. Full immutable predictions, masks, per-frame oracle evaluation, loss intervals, IoU variants, runtime versions and source hashes: `results/g6a/final_white_hsv01` and `results/g6a/final_edge_circle01`. Full photometric measurements: `results/g6a/final_photometric03.json`. Earlier diagnostic and intermediate-policy reports remain preserved. Camera/dataset/exposure-image verification succeeds for this experiment; reliable continuous ball localization remains an explicit failed/limited result. No orange texture, GT prediction fill, parameter search or post-inspection detector tuning was used.
