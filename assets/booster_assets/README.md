# Booster Assets

This repository contains Booster robot models, motion data, and a simple helper package `booster_assets` for local development and tooling.

Robot Configurations
--------------------

### K1 Robot Models

| Configuration                 | Description                                         | Available Formats    |
|------------------------------|-----------------------------------------------------|----------------------|
| K1 (22 DoF)                  | 22-DoF layout: head 2 joints, arms 4 joints ×2, legs 6 joints ×2 | URDF (`robots/K1/K1_22dof.urdf`), XML (`robots/K1/K1_22dof.xml`) |
| K1 Locomotion                | Locomotion variant (fixed heads and arms)          | URDF (`robots/K1/K1_locomotion.urdf`) |

### T1 Robot Models

| Configuration                 | Description                                         | Available Formats    |
|------------------------------|-----------------------------------------------------|----------------------|
| T1 (23 DoF)           | 23-DoF layout: head 2 joints, arms 4 joints ×2, waist 1 joint, legs 6 joints ×2 | URDF (`robots/T1/T1_23dof.urdf`), XML (`robots/T1/T1_23dof.xml`) |
| T1 Locomotion (23 DoF)       | 23-DoF locomotion variant (fixed head, arms, waist where applicable)    | URDF (`robots/T1/T1_locomotion.urdf`), XML (`robots/T1/T1_locomotion.xml`) |
| T1 with 7-DoF Arms (29 DoF)       | 29-DoF layout: arm joints become 7 per arm (head 2, arms 7×2, waist 1, legs 6×2) | URDF (`robots/T1/T1_29dof.urdf`) |

### T2 Robot Models

| Configuration                 | Description                                         | Available Formats    |
|------------------------------|-----------------------------------------------------|----------------------|
| T2 (31 DoF)                  | 31-DoF layout: head 2 joints, arms 7 joints ×2, waist 3 joints, legs 6 joints ×2 | URDF (`robots/T2/T2_31dof/T2_31dof.urdf`), XML (`robots/T2/T2_31dof/T2_31dof.xml`) |

Motion and Data Files
---------------------

- `motions/` contains retargeted motion data for Booster robots. The currently provided motion CSV files are sampled at 50 Hz.

### Motion CSV Format

- Each row represents one frame of a trajectory.
- The first 7 columns are the generalized root pose: root position `x, y, z` and root orientation quaternion `x, y, z, w`.
- The remaining columns are joint positions (radians).
- The joint orders for K1, T1, and T2 are defined in `src/booster_assets/motions.py` (`K1_JOINT_NAMES`, `T1_JOINT_NAMES`, `T2_JOINT_NAMES`).

### Motion List

| File Name                  | Fps | Description                                        |
|----------------------------|-----|----------------------------------------------------|
| k1_fight_001.csv           | 50  | Fighting motion sequence                           |
| k1_mj2_seg1.csv            | 50  | MJ dance segment                                   |


Python installation
-----------------------------

Install the package in editable (development) mode:

```bash
python3 -m pip install -e .
```
