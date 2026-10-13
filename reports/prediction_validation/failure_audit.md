# Stage G: frozen Stage F failure audit

Offline audit of saved 1 ms trajectories, delivered commands, 20 ms decisions and native MuJoCo collision geometry. No physics step, controller call, visual mesh collision assumption, or A–F evidence rewrite.

## Why position −0.40 takes 8.624 s

Both signs issue one 2.44 s lateral burst from t=2.00 to 4.44. The predictive stop uses filtered velocity ×0.4 s and stops with about 0.032–0.035 m predicted residual. The actual first burst leaves the robot near the ±0.30 m tolerance boundary, rather than near the predicted ±0.367 m stopping point.

These times identify high-level decision boundaries. The raw post-step recorder first shows the updated commands 1 ms later (2.001 and 4.441); both layers are preserved in the JSON.

Negative motion first enters tolerance at t=4.726, exits at 4.912, reenters at 4.974, then slow standing drift crosses out at 10.591. At the next 20 ms update (10.60), error exceeds 0.10 m and the controller restarts; permanent hold begins at 10.624, giving 8.624 s after enable. The positive case enters tolerance at 4.733 and stays inside to the trace end, giving 2.733 s. F’s arrival metric requires remaining inside until the end, in addition to at least 1 s observation.

There are zero turn-command samples in both cases. Maximum tracked yaw error is 0.073716 rad negative and 0.037214 rad positive, below the 0.12 rad turn-entry threshold. Both remain safe and upright. This trace establishes asymmetric post-stop drift interacting with stop prediction and the 0.10 m deadband; it does not identify the policy’s internal cause of that drift.

| Case | Permanent hold after enable (s) | Final lateral (m) | Lateral command duration (s) | Turn samples |
|---|---:|---:|---:|---:|
| position_-0.40 | 8.624 | -0.348453 | 3.840 | 0 |
| position_+0.40 | 2.733 | +0.305176 | 2.440 | 0 |

## Failed shots

Every listed active shot has zero nonzero delivered command samples, no effective contact, and a right-goal outcome identical to the matched absent trajectory. Prediction correctness and center reachability are separate from physical collision coverage.

| Case | First usable time (s) | Contact-bound time left (s) | Max center budget (m) | Closest native surface distance (m) | Attribution |
|---|---:|---:|---:|---:|---|
| wide_left | 2.02 | 3.1808 | 0.0281 | 0.6029 | insufficient modeled time + safe-center clamp + clipped-target deadband |
| wide_right | 2.02 | 3.1808 | 0.0281 | 0.5983 | insufficient modeled time + safe-center clamp + clipped-target deadband |
| air_wide | 2.02 | 0.6803 | 0.0000 | 0.2352 | short flight; zero modeled travel budget; missing fast lateral action |
| air_high | 2.02 | 0.6803 | 0.0000 | 0.3059 | center reachable; ball above native body; missing high interception action |
| bouncing | 2.70 | 3.2369 | 0.0337 | 0.1108 | initial unmodeled bounce abstention; later budget below movement deadband |

Wide ±0.9 m: lateral forecast error is zero; whole-ball timing error ≤0.020842 s. Both targets exceed the ±0.65 m safe-center clamp; initial conservative travel is only 0.028084 m. The clipped decision target is inside the 0.10 m controller startup deadband, so it never starts. These are failures of the available action/time budget, compounded by how the conservative planner feeds the deadband; the trace does not prove physical impossibility for another action.

Air-wide: ballistic whole-ball timing error is at numerical roundoff and height error ≤0.004263 m. The first robot-bound deadline has only 0.680314 s left, below the 1 s startup +1 s braking reserve; travel budget is zero. The lateral gap is about 0.599 m.

Air-high: the lateral center is already reachable. Height prediction error is only about 4.3 mm. Native surface distances and top supports are reconstructed from saved qpos for all 20 ball-eligible collision geoms, including compiled collision meshes. Disabled visual meshes are excluded. Through the complete ±0.81 m robot passage bound (t=2.701–2.902), native body top is at most 0.953612 m and ball bottom is at least 0.987296 m. Minimum simultaneous vertical clearance is 0.033684 m; closest native surface separation is 0.305873 m. This establishes that this shot is above the modeled body, rather than passing through a missing local body patch.

Bouncing: initial goal extrapolation is explicitly flagged unmodeled (first whole-ball time error −1.8282 s; height error −44.5408 m). The controller abstains until t=2.70, 0.68 s after the first observed launch state. Once near-ground estimates become usable, the maximum budget is only 0.033688 m, below the 0.10 m startup deadband, so commands remain zero. The large negative raw heights belong to rejected predictions, not accepted forecasts.

Missing native collision regions around shoulder_pitch, hip_pitch and ankle_pitch remain a modeled coverage limitation. These traces show no contact and substantial lateral or vertical separation; they do not establish those missing patches as the cause of these particular failures. A jump, reach or dive was absent from F’s lateral-only action set and is not evaluated by this audit.

## Reproduce and integrity

```bash
.venv/bin/python scripts/audit_goalkeeper_failures.py
```

Historical reports unchanged during audit: True. Saved trajectory hashes and decision-byte parity all pass: True.

JSON contains every command segment and switch, tolerance transition, first prediction and usable decision, prediction error bounds, closest collision state and exact native vertical support bounds. The geometry checks are discrete at the saved 1 ms resolution; they are evidence about these saved experiments, not a proof of success for other actions or unseen shots.
