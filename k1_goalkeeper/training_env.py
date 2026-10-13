"""I1 single-environment contract; current truth observations, no training loop."""
from dataclasses import asdict
from pathlib import Path
import tempfile

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .detection import GoalDetector
from .scene import build_scene
from .simulation import Shot
from .training_actions import BAD_WARNINGS, create_action_backend


def baseline_reward(transition):
    """Untuned event-only baseline. Contact and timeout do not imply a save."""
    return {'right_goal': -float(transition['right_goal']),
            'fall': -float(transition['fell'])}


class K1GoalkeeperEnv(gym.Env):
    """Actor-only Gymnasium observation with explicit privileged/offline getters.

    `joint_targets` is independent of a learned gait; `body_velocity` is an
    explicitly selected reference controller. Physics stays at 1 ms.
    """
    metadata = {'render_modes': [], 'render_fps': 50}
    render_mode = None

    def __init__(self, model=None, action_mode='joint_targets', control_dt=.02,
                 episode_duration=10., reward_fn=baseline_reward, backend_factory=None):
        super().__init__()
        self._directory = None
        if model is None:
            self._directory = tempfile.TemporaryDirectory(prefix='k1-training-')
            model = mujoco.MjModel.from_xml_path(str(build_scene(Path(self._directory.name)/'scene.xml')))
        if not np.isclose(model.opt.timestep, .001, rtol=0, atol=1e-12):
            raise ValueError('I1 requires 0.001 s physics timestep')
        self.model = model
        self.decimation = self._ticks(control_dt, 'control_dt')
        self.control_dt = self.decimation*.001
        self._default_limit = self._ticks(episode_duration, 'episode_duration')
        self.action_mode = action_mode if backend_factory is None else 'custom'
        self.backend = (create_action_backend(model, action_mode) if backend_factory is None
                        else backend_factory(model))
        if (self.backend.model is not model or not isinstance(self.backend.action_space, spaces.Box)
                or len(self.backend.action_space.shape) != 1 or len(self.backend.mapping.joint_names) != 22):
            raise ValueError('backend must use this model, a one-dimensional Box and the 22 K1 joints')
        self.data, self.mapping = self.backend.data, self.backend.mapping
        self.action_space = self.backend.action_space
        self.default_action = self.backend.default_action.copy()
        self.reward_fn = reward_fn
        self.ball_geom = model.geom('ball_collision').id
        self.radius = float(model.geom_size[self.ball_geom, 0])
        self.robot_geoms = set()
        trunk = model.body('trunk').id
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            while body and body != trunk:
                body = int(model.body_parentid[body])
            if body == trunk:
                self.robot_geoms.add(geom)
        self.observation_layout = {}
        offset = 0
        for name, count in [('projected_gravity', 3), ('body_linear_velocity', 3),
                            ('body_angular_velocity', 3), ('goal_relative_position', 3),
                            ('world_heading_cos_sin', 2), ('joint_position_offset', 22),
                            ('joint_velocity', 22), ('ball_relative_position', 3),
                            ('ball_relative_linear_velocity', 3), ('ball_observation_valid', 1),
                            ('previous_action', self.action_space.shape[0])]:
            self.observation_layout[name] = (offset, offset+count)
            offset += count
        self.observation_space = spaces.Box(-np.inf, np.inf, (offset,), dtype=np.float32)
        self.privileged_observation_space = spaces.Dict({
            'qpos': spaces.Box(-np.inf, np.inf, (model.nq,), dtype=np.float32),
            'qvel': spaces.Box(-np.inf, np.inf, (model.nv,), dtype=np.float32),
            'qacc': spaces.Box(-np.inf, np.inf, (model.nv,), dtype=np.float32),
            'actuator_torque': spaces.Box(-np.inf, np.inf, (22,), dtype=np.float32),
            'ball_angular_velocity_body': spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            'body_mass': spaces.Box(0, np.inf, (model.nbody,), dtype=np.float32),
            'body_inertia': spaces.Box(0, np.inf, (model.nbody, 3), dtype=np.float32),
        })
        self._ready = self._closed = False
        self._finished = False

    @staticmethod
    def _ticks(value, name, allow_zero=False):
        if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f'{name} must be finite and positive')
        ticks = round(value/.001)
        if not np.isclose(value/.001, ticks, rtol=0, atol=1e-8):
            raise ValueError(f'{name} must be a multiple of 0.001 s')
        return ticks

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError('environment is closed')
        options = dict(options or {})
        if set(options)-{'task', 'shot', 'launch_time', 'time_limit'}:
            raise ValueError('unknown reset option')
        task = options.get('task', 'standing')
        if task not in ('standing', 'single_shot'):
            raise ValueError('task must be standing or single_shot')
        limit = self._ticks(options.get('time_limit', self._default_limit*.001), 'time_limit')
        shot = options.get('shot', Shot())
        if isinstance(shot, dict):
            shot = Shot(**shot)
        if not isinstance(shot, Shot):
            raise ValueError('shot must be a Shot or its constructor dictionary')
        shot = Shot(**{**asdict(shot), **{key: tuple(getattr(shot, key)) for key in
                                        ('position', 'velocity', 'angular_velocity')}})
        launch = self._ticks(options.get('launch_time', 2.), 'launch_time', allow_zero=True)
        if task == 'single_shot' and launch >= limit:
            raise ValueError('shot launch must precede the time limit')
        if task == 'standing' and any(key in options for key in ('shot', 'launch_time')):
            raise ValueError('standing task does not schedule a shot')
        super().reset(seed=seed)
        self._ready = False
        self.backend.reset()
        self.data, self.mapping = self.backend.data, self.backend.mapping
        self._task, self._shot = task, shot
        self._launch_tick, self._limit_tick = launch, limit
        self._physics_steps = self._decision_steps = 0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._launched = self._robot_contact_seen = False
        self._launch_record = None
        self._event = self._numerical_error = None
        self._contacts = []
        self._finished = False
        self._warning_counts = np.asarray([self.data.warning[w].number for w in BAD_WARNINGS])
        self.detector = GoalDetector(radius=self.radius)
        # A real parked ball, inside the pitch and far from the keeper. No hidden
        # launch state is encoded in actor or privileged observations.
        q, v = self.mapping.ball_qpos_address, self.mapping.ball_dof_address
        self.data.qpos[q:q+7] = (-7., 5., self.radius+.001, 1., 0., 0., 0.)
        self.data.qvel[v:v+6] = 0
        mujoco.mj_forward(self.model, self.data)
        if task == 'single_shot' and launch == 0:
            self._launch()
        self._check_simulation()
        self._last_observation = self._observe()
        self._last_observation_time = float(self.data.time)
        self._cache_valid_state()
        self._ready = True
        return self._last_observation.copy(), self._info(None, {}, 0., True)

    def _cache_valid_state(self):
        self._last_valid_state = {'time': float(self.data.time), 'qpos': self.data.qpos.copy(),
                                  'qvel': self.data.qvel.copy(), 'ctrl': self.data.ctrl.copy()}

    def _launch(self):
        q, v = self.mapping.ball_qpos_address, self.mapping.ball_dof_address
        self.data.qpos[q:q+7] = (*self._shot.position, 1., 0., 0., 0.)
        self.data.qvel[v:v+6] = (*self._shot.velocity, *self._shot.angular_velocity)
        mujoco.mj_forward(self.model, self.data)
        self._launched = True
        self._launch_record = {'time': float(self.data.time), 'position': list(self._shot.position),
                               'velocity': list(self._shot.velocity),
                               'angular_velocity_body': list(self._shot.angular_velocity)}
        # Teleporting from parking is reset data, never a ball-line crossing.
        self.detector = GoalDetector(radius=self.radius)
        self._event = self.detector.update(float(self.data.time), self.data.qpos[q:q+3])

    def _rotation(self):
        q = self.mapping.base_qpos_address
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[q+3:q+7])
        return rotation.reshape(3, 3)

    def _observe(self):
        m, d = self.mapping, self.data
        q, v, bq, bv = m.base_qpos_address, m.base_dof_address, m.ball_qpos_address, m.ball_dof_address
        rotation = self._rotation()
        # Free-joint angular velocities are local; linear velocities are world.
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        values = [rotation.T @ [0., 0., -1.], rotation.T @ d.qvel[v:v+3],
                  d.qvel[v+3:v+6], [d.qpos[q+1], 11.-d.qpos[q], d.qpos[q+2]],
                  [np.cos(yaw), np.sin(yaw)],
                  d.qpos[m.qpos_addresses]-m.default_positions, d.qvel[m.dof_addresses],
                  rotation.T @ (d.qpos[bq:bq+3]-d.qpos[q:q+3]),
                  rotation.T @ (d.qvel[bv:bv+3]-d.qvel[v:v+3]), [1.], self._previous_action]
        result = np.concatenate(values).astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError('nonfinite actor observation')
        return result

    def _check_simulation(self):
        if not all(np.isfinite(value).all() for value in
                   (self.data.qpos, self.data.qvel, self.data.qacc, self.data.ctrl)):
            raise RuntimeError('MuJoCo nonfinite state or control')
        if not np.array_equal(self._warning_counts, [self.data.warning[w].number for w in BAD_WARNINGS]):
            raise RuntimeError('MuJoCo BAD warning')
        if not np.isclose(self.data.time, self._physics_steps*.001, rtol=0, atol=1e-9):
            raise RuntimeError('MuJoCo unexpected time reset')

    def _fell(self):
        return bool(self.data.qpos[self.mapping.base_qpos_address+2] < .3 or self._rotation()[2, 2] < .5)

    def _read_contacts(self):
        contacts = []
        for index, contact in enumerate(self.data.contact[:self.data.ncon]):
            if contact.efc_address < 0:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            if force[0] <= 1e-6:
                continue
            pair = (int(contact.geom1), int(contact.geom2))
            ball_robot = self.ball_geom in pair and any(g in self.robot_geoms for g in pair)
            self._robot_contact_seen |= ball_robot
            contacts.append({'geom_ids': pair, 'normal_force_n': float(force[0]),
                             'ball_robot': bool(ball_robot)})
        self._contacts = contacts

    def _info(self, reason, terms, dt, valid):
        return {'action_mode': self.action_mode, 'task': self._task,
                'simulation_time': float(self.data.time) if np.isfinite(self.data.time) else None,
                'returned_observation_time': self._last_observation_time,
                'last_valid_physics_time': self._last_valid_state['time'], 'physics_steps': self._physics_steps,
                'decision_steps': self._decision_steps, 'elapsed_dt': dt,
                'termination_reason': reason, 'reward_terms': dict(terms),
                'exit_event': self._event.to_dict() if self._event else None,
                'robot_contact_seen': bool(self._robot_contact_seen),
                'active_contact_count': len(self._contacts), 'transition_valid': valid,
                'discard_transition': not valid,
                'invalid_state': not valid, 'numerical_error': self._numerical_error,
                'bootstrap_allowed': bool(valid and reason in (None, 'time_limit')),
                'observation_source': 'current_simulation_truth'}

    def step(self, action):
        if self._closed or not self._ready or self._finished:
            raise RuntimeError('call reset before step on an inactive episode')
        # Validate before changing episode state, advancing physics or history.
        value = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(value) or not np.isfinite(value).all():
            raise ValueError('action shape, bounds or finiteness invalid')
        self.backend.set_action(value)
        self._previous_action = value.copy()
        begin = float(self.data.time)
        reason, valid = None, True
        terminated = truncated = False
        self._decision_steps += 1
        try:
            self._check_simulation()
            if self._fell():
                reason, terminated = 'fall', True
            for _ in range(self.decimation):
                if terminated:
                    break
                if self._task == 'single_shot' and not self._launched and self._physics_steps == self._launch_tick:
                    self._launch()
                if self._event:
                    reason, terminated = self._event.reason, True
                    break
                before = float(self.data.time)
                try:
                    self.backend.step_physics()
                finally:
                    if np.isclose(self.data.time, before+.001, rtol=0, atol=1e-9):
                        self._physics_steps += 1
                self._check_simulation()
                self._cache_valid_state()
                self._read_contacts()
                if self._launched:
                    q = self.mapping.ball_qpos_address
                    self._event = self.detector.update(float(self.data.time), self.data.qpos[q:q+3])
                if self._event:
                    reason, terminated = self._event.reason, True
                elif self._fell():
                    reason, terminated = 'fall', True
                elif self._physics_steps >= self._limit_tick:
                    reason, truncated = 'time_limit', True
                if terminated or truncated:
                    break
            observation = self._observe()
            self._last_observation = observation.copy()
            self._last_observation_time = float(self.data.time)
        except RuntimeError as error:
            if 'safety fallback' in str(error) and self._fell():
                reason, terminated = 'fall', True
                observation = self._observe()
                self._last_observation = observation.copy()
                self._last_observation_time = float(self.data.time)
            else:
                reason, valid, terminated, truncated = 'numerical_error', False, False, True
                self._numerical_error = {'message': str(error),
                    'warning_counts': {w.name: int(self.data.warning[w].number) for w in BAD_WARNINGS}}
                observation = self._last_observation.copy()
        transition = {'right_goal': bool(self._event and self._event.outcome == 'Goal' and self._event.goal == 'right'),
                      'fell': reason == 'fall', 'numerical_error': not valid,
                      'elapsed_dt': float(self.data.time)-begin if valid else self._physics_steps*.001-begin,
                      'termination_reason': reason, 'transition_valid': valid,
                      'actor_observation': observation.copy(),
                      'privileged_state': self.get_privileged_state() if valid else None}
        try:
            terms = dict(self.reward_fn(transition))
            if not all(np.isfinite(v) for v in terms.values()):
                raise ValueError('reward terms must be finite')
        except Exception:
            self._finished = True
            raise
        reward = float(sum(terms.values()))
        self._finished = bool(terminated or truncated)
        return observation, reward, terminated, truncated, self._info(reason, terms, transition['elapsed_dt'], valid)

    def get_privileged_state(self):
        """Critic channel: current exact physics only, no pending shot or labels."""
        if not self._ready or self._closed:
            raise RuntimeError('call reset before reading privileged state')
        self._check_simulation()
        m = self.mapping
        return {'qpos': self.data.qpos.astype(np.float32).copy(),
                'qvel': self.data.qvel.astype(np.float32).copy(),
                'qacc': self.data.qacc.astype(np.float32).copy(),
                'actuator_torque': self.data.qfrc_actuator[m.dof_addresses].astype(np.float32).copy(),
                'ball_angular_velocity_body': self.data.qvel[m.ball_dof_address+3:m.ball_dof_address+6].astype(np.float32).copy(),
                'body_mass': self.model.body_mass.astype(np.float32).copy(),
                'body_inertia': self.model.body_inertia.astype(np.float32).copy()}

    def get_evaluation_state(self):
        """Offline-only snapshot; includes scheduled input, never fed to Actor/Critic."""
        if not self._ready or self._closed:
            raise RuntimeError('call reset before reading evaluation state')
        return {'time': float(self.data.time), 'qpos': self.data.qpos.copy(),
                'qvel': self.data.qvel.copy(), 'ctrl': self.data.ctrl.copy(),
                'contacts': [dict(contact) for contact in self._contacts],
                'current_state_valid': self._numerical_error is None,
                'last_valid_state': {key: value.copy() if isinstance(value, np.ndarray) else value
                                     for key, value in self._last_valid_state.items()},
                'scheduled_shot': asdict(self._shot) if self._task == 'single_shot' else None,
                'scheduled_launch_time': self._launch_tick*.001 if self._task == 'single_shot' else None,
                'delivered_launch': None if self._launch_record is None else {k: list(v) if isinstance(v, list) else v
                                                                           for k, v in self._launch_record.items()},
                'exit_event': self._event.to_dict() if self._event else None}

    def close(self):
        if not self._closed:
            simulation = getattr(self.backend, 'simulation', None)
            if simulation is not None:
                simulation.controller.stop()
            if self._directory is not None:
                self._directory.cleanup()
            self._closed = True
