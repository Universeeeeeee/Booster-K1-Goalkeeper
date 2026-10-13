# Booster Deploy

Booster Deploy is a lightweight deployment framework that supports running control policies on Booster robots (sim2real) and MuJoCo (sim2sim). The system adopts many well-established designs from IsaacLab to provide modular abstractions, allowing unified policy execution across simulated and real platforms.


## Prerequisites

| Environment | Notes |
|-------------|-------|
| Booster firmware >= v1.7.2 | Required for real robot deployments. |
| Python 3.10+ | Already installed on the robot |
| ROS 2 with `booster_interface` | Required for the DDS-backed `/low_state`, `/joint_ctrl`, and RPC interfaces. Already installed on the robot. |
| MuJoCo | Optional; install if you plan to run simulation locally. |


## Running Deployments

### Python environment

Create and activate a local virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Debian/Ubuntu, install `python3-venv` if needed. On the robot, activate
`.venv` before loading ROS 2; `booster_interface` is provided by the robot:

```bash
source .venv/bin/activate
source /opt/booster/BoosterRos2Interface/install/setup.bash
```

### Add and list tasks:
   1. Create a subfolder under `tasks/` for your task.
   2. Implement a `Policy`/`PolicyCfg` and provide a `ControllerCfg` referencing the policy.
   3. Place policy checkpoints under `models/` and reference the path in the config.
   4. Register your `ControllerCfg` config in the task registry (see existing tasks for the registration pattern).
   5. Check all available tasks:
      ```bash
      python scripts/deploy.py --list
      ```

### Policy inference backends

The checkpoint suffix selects the inference backend automatically:

- `.pt`, `.jit`, `.torchscript`: TorchScript
- `.onnx`: ONNX Runtime with the CPU execution provider
make sure `onnxruntime` is installed in the deployment environment.

### Run Sim2Sim (MuJoCo)

- Download and install BoosterAssets:
   - Clone the [booster_assets](https://github.com/BoosterRobotics/booster_assets) which contains Booster robot models and resources.
   - Install booster_assets python helper following the instructions in the repository.

- Install Python dependencies in the activated virtual environment:
   ```
   python -m pip install -r requirements.txt
   ```

- Launch the task in mujoco:
   ```bash
   python scripts/deploy.py --task <TASK_NAME> --mujoco
   ```

### Run Sim2Real (Real Robots)

**IMPORTANT**: Make sure to install [Booster Firmware](https://booster.feishu.cn/wiki/E3q5wF5SnitXZgkY18Uc8odBnXb) >= v1.4 on the robot before proceeding.

- After you finish testing your task with Sim2Sim locally, copy the project to the robot.

- Install Python dependencies in the activated virtual environment on the robot:
   ```
   python -m pip install -r requirements.txt
   ```

- SSH into the robot and start the ROS 2 environment by sourcing the provided setup script:
   ```bash
   source /opt/booster/BoosterRos2Interface/install/setup.bash
   ```

- Launch the task on the robot and follow the prompts shown in the command line..
   ```bash
   python scripts/deploy.py --task <TASK_NAME>
   ```

#### PD damping (`Kd`) on the real robot

For parallel-actuated joints, `robot.joint_damping` is sent directly to the
motors, so do not reuse the training-simulator `Kd`. Compute the motor-side
value as:

```text
Kd = 2 * zeta * J_eq * (2 * pi * f_n)
```

where `J_eq` is the armature of the parallel-actuated joint, `f_n` is the
natural frequency, and `zeta` is the damping ratio.


#### Controller exit mode

`booster.exit_mode` controls the robot mode entered after the custom
controller exits. It applies to robots whose firmware supports the
corresponding DDS RPC mode-switch API:

- `"damping"`: switch to damping mode
- `"walking"`: switch to walking mode (default)

The value can be set in the task controller configuration, for example:

```python
booster = BoosterRobotControllerCfg(exit_mode="damping")
```

You can override the task configuration at startup when damping is preferred:

```bash
python3 scripts/deploy.py --task <TASK_NAME> --exit-mode damping
```

The value `"walk"` is also accepted as an alias for `"walking"` in Python configuration.


#### Robot preparation mode

`robot.prepare_mode` controls what happens after pressing `X` to enter Custom
mode. Set it independently in each robot configuration (T1, T2, or K1):

- `"walking"` (default): read the current joint positions from `/low_state`,
  publish one position-hold command using the robot's `prepare_state` `kp/kd`,
  switch to Custom, then start the matching robot locomotion policy with all
  velocity commands masked to zero. Press `A` on the remote (or `r` on the
  keyboard) to stop the preparation policy and start the task selected by
  `--task`.
- `"standing"`: publish the current-position hold command, switch to Custom,
  and interpolate for approximately one second to the configured
  `prepare_state.joint_pos`. Press `A`/`r` to start the selected task policy.

The mode can be set in a robot configuration, for example:

```python
robot = T2_31DOF_CFG.replace(prepare_mode="walking")
```

### Remote Controller

The deployment supports both remote controllers and keyboard input:

- GameSir: detected automatically.
- Booster remote: used through `/remote_controller_state`.
- Keyboard: available.

<table>
  <tr>
    <th align="left">GameSir</th>
    <th align="left">Booster remote</th>
  </tr>
  <tr>
    <td valign="top"><img src="docs/images/gamesir.jpg" alt="GameSir remote" width="320"></td>
    <td valign="top"><img src="docs/images/booster_remote.jpg" alt="Booster remote" width="320"></td>
  </tr>
</table>

On either remote controller, use the left stick for forward/lateral motion, the
right stick for rotation, `X` to start Custom mode, and `A` to start RL mode.

| Control | Action |
|---------|--------|
| Left stick forward/back | Increase/decrease forward velocity (`vx`) |
| Left stick left/right | Increase/decrease lateral velocity (`vy`) |
| Right stick left/right | Rotate left/right (`vyaw`) |
| Joystick `X` | Start Custom mode |
| Joystick `A` | Start RL mode |

Keyboard:

| Key | Action |
|-----|--------|
| `w` / `s` | Increase/decrease `vx` by `0.1` |
| `a` / `d` | Increase/decrease `vy` by `0.1` |
| `q` / `e` | Increase/decrease `vyaw` by `0.1` |
| `x` | Start Custom mode |
| `r` | Start RL mode |
| `Space` | Set all velocity commands to zero |

With `prepare_mode="walking"`, `X` starts zero-command locomotion preparation and
`A`/`r` starts the selected task policy. With `prepare_mode="standing"`, `X`
first performs the one-second transition to `prepare_state.joint_pos`, and
`A`/`r` then starts the selected task policy. Stop the deployment with `Ctrl+C`.


## Repository Layout

```
booster_deploy/
├─ booster_deploy/           # Controllers, policies, utilities
│  └─ robots/                # Robot model configurations
│     ├─ k1.py               # K1 configuration
│     ├─ t1.py               # T1 23-DOF configuration
│     ├─ t2.py               # T2 31-DOF configuration
│     ├─ __init__.py         # Public configuration exports
│     └─ booster.py          # Backward-compatible import path
├─ scripts/                  # Entry-point scripts (deploy.py)
├─ tasks/                    # Task registry and configs
└─ requirements.txt          # Python dependencies
```

Key modules:
- `booster_deploy/`: Core module providing a unified abstraction for MuJoCo and physical robots. Real-robot communication uses ROS 2 DDS (a `/low_state` subscriber, `/joint_ctrl` publisher, and RPC client).
- `booster_deploy/robots/`: Robot configuration modules. Each robot has a dedicated module that defines a `RobotCfg` describing:
    - `k1.py`: `K1_CFG`
    - `t1.py`: `T1_23DOF_CFG`
    - `t2.py`: `T2_31DOF_CFG`
    - joint names and body names
    - default joint positions
    - default joint stiffness (`joint_stiffness`) and damping (`joint_damping`)
    - effort limits
    - `mjcf_path` for MuJoCo model loading
    - `prepare_state` (prepare pose, stiffness and damping used when entering custom mode)

  Import configurations from the package or from the robot-specific module:

  ```python
  from booster_deploy.robots import K1_CFG
  # Equivalent:
  from booster_deploy.robots.k1 import K1_CFG
  ```

  `booster_deploy.robots.booster` remains available as a backward-compatible import path for existing deployments.

 - `tasks/`: User task definitions and implementations. Each task module contains:
    - `Policy`/`PolicyCfg` class implementing the inference logic;
    - a `ControllerCfg` class describing the task configuration including the policy;
    - registering a task with a `ControllerCfg` instance.

   Typical task layout (example):

   ```text
   tasks/my_task/
   ├─ __init__.py        # registers the task via utils.register.register_task
   ├─ task.py            # Policy and ControllerCfg implementation
   ├─ models/            # optional policy checkpoints
   └─ motions/           # optional motion primitives or recordings
   ```
