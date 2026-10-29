# G5 static perception baseline

Run with Python, NumPy, Pillow and `opencv-python-headless` installed:

The local recorded evaluation used Python 3.13.13, NumPy 2.3.3, Pillow 11.3.0 and OpenCV 5.0.0; `results/g5/local_evaluation_environment.json` records the interpreter identity. These are offline evaluator versions, separate from Isaac/SDK runtime dependencies.

```sh
python scripts/run_static_perception.py --capture results/sdk_capture --output results/g5_perception --evaluation results/sdk_capture/evaluation.jsonl
```

The output directory must not exist. Input is the unmodified standalone receiver format: `metadata.json` contains `depth_source: zed_stereo_sdk`, `depth_units: m`, `baseline_m`, and `calibration.left/right` (`fx`, `fy`, `cx`, `cy`). `frames.jsonl` carries `frame_index`, `sdk_timestamp_ns`, `width`, `height`, `left_path`, `right_path`, `depth_path`, `confidence_path`. PNG images are rectified RGB; both depth and confidence are little-endian float32 row-major arrays. Depth is optical Z, metres. SDK confidence ranges 0–100, lower is better; this baseline accepts at most 95. The reader checks the input declaration, not sender provenance or calibration accuracy. Use G2/G3 acquisition verification evidence separately.

The simple scene must contain one high-contrast orange ball with a known 0.11 m radius (override `--radius-m`). The detector thresholds OpenCV HSV H=5–35, S≥100, V≥60, finds external contours with area≥6 px² and circularity≥0.55, and fills the selected contour as its mask. Comparable competing contours, clipping at the image boundary, and missing contours return invalid. Its confidence is contour circularity, not a calibrated detection probability. Texture inside the orange range can support SDK correspondence. This detector does not claim general football recognition or reliable partial-occlusion classification.

Two independent estimates are saved. Rectified triangulation uses left/right contour centroids, rejects vertical mismatch above 2 px and nonpositive normalized disparity, and applies the actual per-eye intrinsics and baseline. Silhouette centroids approximate sphere-center projections; rasterization and off-axis perspective introduce bias. SDK sphere fitting erodes the left foreground one pixel, removes nonfinite/nonpositive/low-confidence depth, removes distant background clusters, back-projects metric points and robustly fits a known-radius sphere using iteratively reweighted least squares. The solver estimates all three center coordinates; it does not treat surface Z plus radius as the final center. Insufficient support, inconsistent fits, rank/conditioning failures and the wrong visible hemisphere return invalid. At least 12 points and 15% valid foreground are required, with at least 65% sphere inliers within max(5 mm, 0.15 radius). The initial center is offset from median foreground along the viewing ray only to initialize the optimizer. This is a geometric baseline whose quality depends on SDK surface quality and the known radius.

`predictions.jsonl` saves masks, bboxes, centroids, confidence, valid/reason, optical 3D center, valid depth fraction and fit residuals; masks are separate PNGs. The `valid` flag is the per-method tracking eligibility mask: downstream tracking must skip invalid observations. No tracking state is synthesized. `center_camera_m` is null on invalids, with no fallback from GT or the other depth method.

GT is optional and read by the offline evaluator only after every prediction has been written. Evaluation JSONL rows use `frame_index`, `depth_label_m`, `angle_label` (`left`, `center`, `right`), `ball_center_camera_m`, and optionally `T_world_camera`. For negatives use a distinct angle label (`out_of_fov`, `occluded`) and null/absent center; never score an invisible GT position as a measured prediction. The pose is the column-vector left optical-camera-to-world rigid transform. Oracle world estimates are emitted only in evaluation and labeled accordingly. This is not a visual pose estimate. Camera-frame and world-frame Euclidean error are equal under the same rigid oracle transform.

`summary.json` and `metrics.csv` report detection rate, valid counts, optical depth MAE/RMSE and 3D center MAE/RMSE separately for every depth/angle and both methods. The summary reports whether the full five-depth by three-angle grid is present. It does not silently count missing combinations as success. Negative-case false detections are visible through detection rate and method valid counts. Analytic unit fixtures cover 0.8, 1.2, 2, 3, 4 m × −20°, 0°, +20°, invalid depths, off-axis sphere fitting, background contamination, clipping, epipolar mismatch and evaluator isolation; these tests are algorithm checks, not SDK acceptance evidence. Actual capture reports must be attached separately before claiming the G5 gate.

Far-depth sensitivity is quadratic: σZ≈Z²/(fB)σd. With f=336 px, B=0.12 m, and σd=0.25 px, σZ at 6 m is 0.223 m. The triangulation result records this assumed uncertainty; it is neither a measured standard deviation nor a promise of centimetre accuracy. Confidence thresholds, fit residuals and invalid rates must remain visible alongside mean errors.

YOLO comparison and model/license audit are tracked separately; this HSV implementation does not load arbitrary remote weights.

The evaluator also requires unique prediction frame indices and unique label frame indices. Labels should specify `expected_visible` explicitly; for compatibility only, its default is whether a center is supplied. An explicitly invisible frame is never scored against its stored center, even when diagnostic GT is present. Each group reports false positives, false-positive rate for invisible scenes, missed detections, and invalid-reason counts for detection and each geometric method. Detection mask fraction and SDK foreground depth-valid fraction are reported as means and minima.

The 2D metric is centroid error against the analytic projected sphere silhouette centroid, which differs slightly off axis from projecting the 3D center. For camera-space center (X,Y,Z), the silhouette centroid is `(cx+fx*XZ/(Z²−r²), cy+fy*YZ/(Z²−r²))`. SDK surface errors are evaluated separately: for every detected foreground pixel with usable SDK confidence/depth, the evaluator intersects its camera ray with the GT sphere and compares SDK optical Z with the front intersection Z. The summary reports sample count, MAE and RMSE for these surface residuals separately from fitted-center errors. These GT ray intersections exist only in evaluation. The CSV includes 2D centroid, SDK surface, sphere center and triangulation errors alongside validity fractions. A full grid means exactly five unique positive depth labels and all three angle labels at each depth; invisible labels do not satisfy positive coverage.

## Actual SDK collection 07 (2026-10-29)

The saved standalone ZED SDK 5.1.0 PERFORMANCE capture has 185 frames, all associated to sender state by visual IDs in both SDK images and stream timestamps. All 15 cells at 1, 1.5, 2, 3, 4 m × left/center/right are present. This is actual SDK output; analytic tests above are separate evidence.

Run outputs are `results/g5/perception07/{summary.json,metrics.csv,predictions.jsonl,masks/}`. The pre-guard run is preserved at `results/g5/perception07_pre_guard`; numerical planar-support and silhouette-size guards were derived from independent synthetic false-accept cases, and did not change collection07 predictions.

HSV detected all 173 visible frames. Triangulation was valid for all 173, with overall 3D center MAE 0.0244 m, optical-Z MAE 0.0232 m and Z RMSE 0.0358 m. Mean 2D silhouette-centroid error was 0.522 px. Out-of-FOV and fully occluded scenes each have six frames, with zero detections and zero valid geometric estimates in both cases. Oracle world transforms are saved separately by the evaluator.

The SDK depth branch failed this scene: only 1/173 visible sphere fits was valid, and even that center was wrong by 0.798 m. The other 172 returned `sphere_fit_inconsistent`; SDK errors are not replaced from triangulation or GT. The SDK confidence-filtered surface MAEs below remain large despite superficially high depth-valid fractions. This collection establishes a working detection/triangulation baseline and an explicit SDK failure case; it does **not** establish acceptable SDK foreground-depth accuracy.

| Depth m | Angle | Detected/frames | Triangulation center MAE m | SDK surface Z MAE m | SDK center valid/frames |
|---|---|---|---|---|---|
| 1 | left | 6/6 | 0.0108 | 5.867 | 0/6 |
| 1 | center | 12/12 | 0.0114 | 7.511 | 0/12 |
| 1 | right | 12/12 | 0.0092 | 6.612 | 0/12 |
| 1.5 | left | 12/12 | 0.0051 | 6.447 | 0/12 |
| 1.5 | center | 12/12 | 0.0076 | 5.397 | 0/12 |
| 1.5 | right | 12/12 | 0.0043 | 4.226 | 0/12 |
| 2 | left | 12/12 | 0.0114 | 5.771 | 0/12 |
| 2 | center | 12/12 | 0.0040 | 3.258 | 0/12 |
| 2 | right | 12/12 | 0.0122 | 3.332 | 0/12 |
| 3 | left | 12/12 | 0.0303 | 2.755 | 0/12 |
| 3 | center | 12/12 | 0.0538 | 2.293 | 0/12 |
| 3 | right | 12/12 | 0.0233 | 2.000 | 0/12 |
| 4 | left | 12/12 | 0.0413 | 0.952 | 1/12 |
| 4 | center | 12/12 | 0.1026 | 1.067 | 0/12 |
| 4 | right | 11/11 | 0.0318 | 0.586 | 0/11 |

Image inspection confirmed a textured yellow-orange sphere against a grayscale textured background and an opaque gray occluder in the negative case. Raw, eroded detected-foreground depth confirms the input failure independently of GT center fitting: frame 0 has SDK depth median 8.143 m (10th–90th percentiles 6.676–13.890 m), frame 36 has median 5.339 m, frame 66 median 4.794 m, and frame 72 median 4.826 m. SDK confidence medians for these patches are 23–45, so confidence alone did not reject the wrong correspondence. Improving source texture/calibration/SDK matching requires a new independently verified experiment; these errors must remain in this report.

The sphere fitter now rejects numerically planar point support (smallest/largest centered singular value below 1e-4) and fitted silhouettes whose width or height differs from the observed contour by a factor outside 0.65–1.5. These broad image-only physical consistency guards prevent demonstrated synthetic background-plane false accepts; they are not an accuracy guarantee for curved, incorrectly matched SDK patches. All fifteen analytic sphere depth/angle cases still pass.

## SDK depth-mode control 10: ULTRA, original texture

Control10 preserves the fine orange sphere geometry/texture and changes the receiver SDK depth mode to ULTRA. All 185 frames were evaluated by the same finalized predictor/evaluator, with no threshold or radius changes. All fifteen depth/angle cells are present; HSV and triangulation remain valid on 173/173 visible frames, and both methods return invalid on all twelve negative frames. Triangulation center MAE is 0.02436 m.

ULTRA improves only the nearest-depth subset: SDK sphere fits are valid on 26/173 visible frames, with selected-valid center MAE 0.00294 m. These valid results occur exclusively at 1 m; all 1.5–4 m fits remain invalid. This conditional error must be read with the 15.0% valid rate. SDK surface errors still include wrong background/depth pixels even when robust fitting recovers a valid center.

| Depth m | Angle | SDK valid/visible | SDK valid-only center MAE m | SDK surface Z MAE m |
|---|---|---|---|---|
| 1 | left | 2/6 | 0.0037 | 2.228 |
| 1 | center | 12/12 | 0.0036 | 0.173 |
| 1 | right | 12/12 | 0.0022 | 0.610 |
| 1.5 | left | 0/12 | invalid | 5.003 |
| 1.5 | center | 0/12 | invalid | 5.610 |
| 1.5 | right | 0/12 | invalid | 4.078 |
| 2 | left | 0/12 | invalid | 5.081 |
| 2 | center | 0/12 | invalid | 5.362 |
| 2 | right | 0/12 | invalid | 3.841 |
| 3 | left | 0/12 | invalid | 3.026 |
| 3 | center | 0/12 | invalid | 3.005 |
| 3 | right | 0/12 | invalid | 2.233 |
| 4 | left | 0/12 | invalid | 0.516 |
| 4 | center | 0/12 | invalid | 0.721 |
| 4 | right | 0/11 | invalid | 0.626 |

Outputs: `results/g5/perception10/{summary.json,metrics.csv,predictions.jsonl,masks/}`. The original PERFORMANCE collection07 failure above is retained as a separate experiment.

## Texture control 11: ULTRA, coarse black/orange patches

Control11 keeps the geometry and ULTRA receiver mode while changing only the sphere appearance to coarse, high-contrast black/orange patches. The fixed HSV detector/evaluator was not changed. This control exposes an appearance limitation: only 10/173 visible left detections pass the contour filter, all at 1.5 m center. There are no valid stereo triangulation estimates because the paired right contour fails. Nine SDK sphere estimates are valid, with selected-valid center MAE 0.02975 m and detected-mask surface Z MAE 0.04828 m. The remaining 163 visible left detections return `no_ball_contour`. Both six-frame negative groups remain invalid.

Image inspection and raw contour measurement independently explain the failure. Black surface patches reach the silhouette and notch the thresholded orange contour. In representative frames, largest-orange-contour circularity is 0.520 at 1 m center, 0.535 at 2 m center and 0.363 at 4 m left, below the unchanged 0.55 requirement. Orange pixel support is present; this is a shape/appearance rejection, not an absent rendered ball. The source does not preserve the baseline detector's solid-orange exterior assumption. No rejected mask is replaced by a GT projection.

| Depth m | Angle | Left detected/visible | Triangulation valid/visible | SDK valid/visible | SDK valid-only center MAE m |
|---|---|---|---|---|---|
| 1 | left | 0/6 | 0/6 | 0/6 | invalid |
| 1 | center | 0/11 | 0/11 | 0/11 | invalid |
| 1 | right | 0/12 | 0/12 | 0/12 | invalid |
| 1.5 | left | 0/12 | 0/12 | 0/12 | invalid |
| 1.5 | center | 10/12 | 0/12 | 9/12 | 0.0297 |
| 1.5 | right | 0/12 | 0/12 | 0/12 | invalid |
| 2 | left | 0/12 | 0/12 | 0/12 | invalid |
| 2 | center | 0/12 | 0/12 | 0/12 | invalid |
| 2 | right | 0/12 | 0/12 | 0/12 | invalid |
| 3 | left | 0/12 | 0/12 | 0/12 | invalid |
| 3 | center | 0/12 | 0/12 | 0/12 | invalid |
| 3 | right | 0/12 | 0/12 | 0/12 | invalid |
| 4 | left | 0/12 | 0/12 | 0/12 | invalid |
| 4 | center | 0/12 | 0/12 | 0/12 | invalid |
| 4 | right | 0/12 | 0/12 | 0/12 | invalid |

Outputs: `results/g5/perception11/{summary.json,metrics.csv,predictions.jsonl,masks/}`. All fifteen sampling cells are present, but appearance failures make this a partial conditional SDK assessment. Missing surface errors are unavailable measurements, not zero error.

| Collection | SDK mode | Appearance | Left detected/visible | Triangulation valid/visible | Triangulation center MAE m | SDK valid/visible | SDK valid-only center MAE m | Negative false detections |
|---|---|---|---|---|---|---|---|---|
| 07 | PERFORMANCE | fine orange | 173/173 | 173/173 | 0.02436 | 1/173 | 0.79840 | 0/12 |
| 10 | ULTRA | fine orange | 173/173 | 173/173 | 0.02436 | 26/173 | 0.00294 | 0/12 |
| 11 | ULTRA | coarse black/orange | 10/173 | 0/173 | unavailable | 9/173 | 0.02975 | 0/12 |

These controls separate SDK depth-mode sensitivity from detector appearance sensitivity. They do not demonstrate usable SDK ball-center coverage across the full five-depth grid. The original failure remains part of the acceptance evidence. YOLO follow-up was explicitly deferred by the user and is not a completion prerequisite for this experiment report.

## Final SDK mode control 13: NEURAL, original fine-orange texture

Collection13 uses the original fine-orange sphere and the SDK NEURAL depth mode. The SDK model was already bundled; its TensorRT optimization is an SDK runtime preparation step. The finalized HSV detector, physical radius prior, fit guards and evaluator are unchanged from the preceding experiments. All 185 SDK frames were independently associated to source state by both-eye visual identity and timestamps. All fifteen positive sampling cells and both negative cases are present. Image inspection confirms the original fine-orange appearance.

HSV and triangulation are valid on 173/173 visible frames. SDK sphere fitting is valid on **164/173 (94.80%)**, with valid-only 3D center MAE **0.02721 m**, P95 **0.08100 m**, and maximum **0.11849 m**. All nine SDK invalids occur at 4 m right and return `sphere_fit_inconsistent`; that cell has only 3/12 valid estimates. The aggregate must therefore be read with its invalid rate and per-cell errors. The 4 m right valid-only error represents three samples, not all twelve frames.

SDK surface residuals are evaluated on all detected usable foreground points, including frames whose sphere fits are invalid. Across 306,349 pixel samples, pooled optical-Z MAE is **0.02733 m** and RMSE is **0.12439 m**. Pooling gives nearer, larger projected spheres more weight; averaging each frame's surface MAE equally gives **0.06652 m**. Far-cell surface errors and outliers remain visible below. Triangulation center MAE is 0.02447 m, P95 0.10068 m and maximum 0.11522 m. Mean 2D silhouette-centroid error is 0.521 px.

| Depth m | Angle | SDK valid/visible | Center MAE m | Center P95 m | Center max m | Surface Z MAE m | Surface Z RMSE m | SDK invalid reason/count |
|---|---|---|---|---|---|---|---|---|
| 1 | left | 6/6 | 0.0066 | 0.0068 | 0.0069 | 0.0073 | 0.0546 | none |
| 1 | center | 11/11 | 0.0059 | 0.0063 | 0.0064 | 0.0078 | 0.0233 | none |
| 1 | right | 12/12 | 0.0019 | 0.0024 | 0.0025 | 0.0055 | 0.0284 | none |
| 1.5 | left | 12/12 | 0.0057 | 0.0063 | 0.0065 | 0.0145 | 0.0968 | none |
| 1.5 | center | 12/12 | 0.0115 | 0.0127 | 0.0128 | 0.0173 | 0.0545 | none |
| 1.5 | right | 12/12 | 0.0078 | 0.0105 | 0.0115 | 0.0195 | 0.1081 | none |
| 2 | left | 12/12 | 0.0240 | 0.0282 | 0.0286 | 0.0564 | 0.1932 | none |
| 2 | center | 12/12 | 0.0229 | 0.0264 | 0.0272 | 0.0452 | 0.1580 | none |
| 2 | right | 12/12 | 0.0266 | 0.0285 | 0.0288 | 0.0448 | 0.1886 | none |
| 3 | left | 12/12 | 0.0320 | 0.0386 | 0.0403 | 0.1068 | 0.3025 | none |
| 3 | center | 12/12 | 0.0399 | 0.0444 | 0.0444 | 0.1206 | 0.2922 | none |
| 3 | right | 12/12 | 0.0321 | 0.0428 | 0.0440 | 0.0854 | 0.2062 | none |
| 4 | left | 12/12 | 0.0755 | 0.1108 | 0.1185 | 0.1278 | 0.2658 | none |
| 4 | center | 12/12 | 0.0713 | 0.0971 | 0.1048 | 0.1860 | 0.3328 | none |
| 4 | right | 3/12 | 0.0470 | 0.0576 | 0.0588 | 0.1175 | 0.2524 | sphere_fit_inconsistent: 9 |

Both negative groups contain six frames. Out-of-FOV: zero detections, zero triangulation estimates, zero SDK sphere estimates. Fully occluded: the same zero/zero/zero result. No GT position or cross-method estimate fills invalid outputs. Oracle-world estimates remain a separate pose experiment.

| Collection | SDK mode | Sphere appearance | Left detected/visible | Triangulation valid/visible | SDK valid/visible | SDK center MAE m (valid only) | Negative false detections |
|---|---|---|---|---|---|---|---|
| 07 | PERFORMANCE | fine orange | 173/173 | 173/173 | 1/173 | 0.79840 | 0/12 |
| 10 | ULTRA | fine orange | 173/173 | 173/173 | 26/173 | 0.00294 | 0/12 |
| 11 | ULTRA | coarse black/orange | 10/173 | 0/173 | 9/173 | 0.02975 | 0/12 |
| 13 | NEURAL | fine orange | 173/173 | 173/173 | 164/173 | 0.02721 | 0/12 |

NEURAL provides usable SDK ball-center observations across all fifteen sampled cells, with the reported weak 4 m right cell. PERFORMANCE and ULTRA failures and the coarse-texture detector failure remain separate evidence. This establishes the static G5 experiment and its valid/invalid contract for the tested simple scene; it does not establish robust recognition of arbitrary football appearance or dynamic/occluded tracking. The 1–4 m results do not justify extending centimetre claims to farther depths.

Final outputs: `results/g5/perception13/{summary.json,metrics.csv,predictions.jsonl,masks/,aggregate_review.json}`. The aggregate report adds P95/maximum center errors by joining the unchanged saved per-frame predictions/evaluation labels. Earlier outputs remain preserved.
