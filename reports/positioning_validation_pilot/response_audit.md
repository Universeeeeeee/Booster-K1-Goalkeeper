# Stage F independent response audit

Official pose faces world −X. Goal lateral +X is world +Y, so command body −Y to move toward positive goal lateral. All velocities below come from recorded C/D displacement, not command magnitude.

Use **0.10 m/s**, **1.0 s startup delay**, **0.02 m startup/reversal excursion reserve**, **0.07 m stopping reserve**, and **1.0 s stopping onset / 2.0 s confirmation allowance** for ±0.2 m/s body lateral commands. The samplewise lower envelope `0.10*max(t−1,0)−0.02` passes every measured lateral segment.

Across 20 segments, minimum steady 0.5 s sliding speed is 0.106815 m/s; zero-start minimum is 0.138626 m/s. D ±0.2 steady means are about 0.1776 m/s. Maximum measured startup adverse excursion is 0.010922 m; maximum whole-recovery stop excursion is 0.059930 m. Verified low-speed onset is at most 0.8 s, requiring one more second of observation. Signed direction and stopping response are asymmetric; see per-direction fields and original measurements.

Use a **±0.65 m center clamp** for a conservative geometric guarantee. Post inner faces are ±1.50 m. The compiled official collision model's maximum root-to-geometry sphere bound is 0.693084 m under arbitrary articulation. Subtracting 0.05 m clearance, 0.07 m stop, and 0.02 m reversal reserve gives ±0.666916 m. A ±0.95 m clamp needs independent moving-body support ≤0.41 m; base-only C/D traces do not establish it. The articulation bound is deliberately restrictive and includes the robot's whole collision geometry.

Method: hash every original JSON/NPZ and existing D scene; project base displacement onto initial body lateral direction; compute overlapping steady 0.5/1 s displacement windows; compute full 5 s stopping excursions; independently compare D world velocities with position finite differences; compile unchanged D scene and use articulated link-length triangle bounds plus actual `geom_rbound`. Full definitions, per-case measurements, hashes, and limitations are in `response_audit.json`.

These are conservative empirical simulation parameters under measured 3–6 s command protocols, not hardware or arbitrary pulse guarantees. Stopping means sustained low velocity, not an exactly fixed position. No E collision findings, robot assets, or source code were changed.
