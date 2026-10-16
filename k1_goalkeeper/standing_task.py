"""Small standing task on the opt-in I2 model, with causal fixed-scale inputs."""

from pathlib import Path
import tempfile

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .training_env import K1GoalkeeperEnv
from .training_model import build_training_scene


TASK_VERSION = 'standing_v1'
ACTION_SCALE = .15  # radians around the authenticated reference posture
PUSH_START = 1.0
PUSH_DURATION = .10
TRAIN_PUSH_MAX = 10.  # horizontal world-frame N, sampled once per episode


def standing_reward(state, action, previous_action, dt, fell):
    """Per-second rates integrated over delivered time, plus one fall penalty."""
    rates = {
        'upright': 1.5 * np.exp(-np.sum(state['gravity'][:2]**2)/.04),
        'height': .5 * np.exp(-(state['position_error'][2]/.04)**2),
        'position': .5 * np.exp(-np.sum(state['position_error'][:2]**2)/.04),
        'body_motion': -.1*np.sum(state['linear_velocity']**2)-.05*np.sum(state['angular_velocity']**2),
        'posture': -.5*np.mean((state['joint_offset']/ACTION_SCALE)**2),
        'joint_velocity': -.002*np.mean(state['joint_velocity']**2),
        'effort': -.02*np.mean(state['torque_ratio']**2),
        'action_rate': -.05*np.mean((action-previous_action)**2),
    }
    return {**{key: float(value*dt) for key, value in rates.items()}, 'fall': -5.*bool(fell)}


class StandingTask(gym.Wrapper):
    """22 normalized joint actions and 78 current-state observations.

    The optional push is external physics, never a policy input. Critic uses
    the same observations in the first PPO experiment; I1 privileged getters
    remain available solely for diagnostics here.
    """

    def __init__(self, model=None, episode_duration=5.):
        self._model_directory = None
        if model is None:
            self._model_directory = tempfile.TemporaryDirectory(prefix='k1-standing-')
            path = build_training_scene(Path(self._model_directory.name)/'scene.xml')
            model = mujoco.MjModel.from_xml_path(str(path))
        super().__init__(K1GoalkeeperEnv(model=model, episode_duration=episode_duration))
        self.action_space = spaces.Box(-1., 1., (22,), dtype=np.float32)
        self.observation_space = spaces.Box(-10., 10., (78,), dtype=np.float32)
        self.reference = self.env.default_action.copy()
        self.origin = model.qpos0[self.env.mapping.base_qpos_address:self.env.mapping.base_qpos_address+3].copy()
        self.trunk = model.body('trunk').id
        self.previous_action = np.zeros(22, dtype=np.float32)
        self.push_force = np.zeros(2)
        limits = self.env.mapping.joint_ranges
        if np.any(self.reference-ACTION_SCALE < limits[:, 0]) or np.any(self.reference+ACTION_SCALE > limits[:, 1]):
            raise ValueError('Standing action neighbourhood exceeds joint limits')

    def _state(self, raw):
        def feature(name):
            return raw[slice(*self.env.observation_layout[name])]
        goal_position = feature('goal_relative_position')
        position = np.array([11.-goal_position[1], goal_position[0], goal_position[2]])
        mapping = self.env.mapping
        return {'gravity': feature('projected_gravity'),
                'position_error': position-self.origin,
                'linear_velocity': feature('body_linear_velocity'),
                'angular_velocity': feature('body_angular_velocity'),
                'joint_offset': feature('joint_position_offset'),
                'joint_velocity': feature('joint_velocity'),
                'torque_ratio': self.env.data.qfrc_actuator[mapping.dof_addresses]/mapping.effort}

    def _observation(self, state):
        value = np.concatenate((state['gravity'], state['linear_velocity'],
                                state['angular_velocity']/5., state['position_error']/np.array([.2, .2, .1]),
                                state['joint_offset']/ACTION_SCALE, state['joint_velocity']/10.,
                                self.previous_action))
        return np.clip(value, -10., 10.).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if set(options)-{'push_force'}:
            raise ValueError('Standing reset only accepts push_force')
        force = options.get('push_force')
        if force is not None:
            force = np.asarray(force, dtype=float)
            if force.shape != (2,) or not np.isfinite(force).all():
                raise ValueError('push_force must contain two finite world-frame force components')
        raw, info = self.env.reset(seed=seed, options={'task': 'standing'})
        self.push_force = (self.np_random.uniform(-TRAIN_PUSH_MAX, TRAIN_PUSH_MAX, 2)
                           if force is None else force.copy())
        self.previous_action = np.zeros(22, dtype=np.float32)
        return self._observation(self._state(raw)), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action) or not np.isfinite(action).all():
            raise ValueError('Standing action must be finite, shape (22,), and within [-1, 1]')
        # Integer physics ticks avoid floating point errors at pulse boundaries.
        tick = self.env.backend.physics_steps
        pushing = round(PUSH_START/.001) <= tick < round((PUSH_START+PUSH_DURATION)/.001)
        self.env.data.xfrc_applied[self.trunk, :2] = self.push_force if pushing else 0.
        raw, _, terminated, truncated, info = self.env.step(self.reference+ACTION_SCALE*action)
        if not info['transition_valid']:
            # Abort before SB3 can store or bootstrap a cached invalid transition.
            raise FloatingPointError(f"Invalid standing transition: {info['numerical_error']}")
        state = self._state(raw)
        terms = standing_reward(state, action, self.previous_action, info['elapsed_dt'],
                                info['termination_reason'] == 'fall')
        self.previous_action = action.copy()
        info = {**info, 'reward_terms': terms, 'applied_push_force': self.env.data.xfrc_applied[self.trunk, :2].tolist()}
        return self._observation(state), sum(terms.values()), terminated, truncated, info

    def close(self):
        self.env.close()
        if self._model_directory is not None:
            self._model_directory.cleanup()
