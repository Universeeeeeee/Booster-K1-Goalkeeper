"""Train one CPU PPO standing experiment and verify checkpoint delivery."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from k1_goalkeeper.standing_task import (ACTION_SCALE, PUSH_DURATION, PUSH_START,
                                        TASK_VERSION, TRAIN_PUSH_MAX, StandingTask)
from k1_goalkeeper.training_model import build_training_scene
from scripts.validate_training_env import file_hash, write_json


PPO_CONFIG = {'n_steps': 512, 'batch_size': 128, 'n_epochs': 5,
              'learning_rate': 3e-4, 'gamma': .99, 'gae_lambda': .95,
              'clip_range': .2, 'ent_coef': 0., 'vf_coef': .5,
              'max_grad_norm': .5, 'target_kl': .02,
              'policy_kwargs': {'net_arch': {'pi': [64, 64], 'vf': [64, 64]}, 'log_std_init': -.7}}
SOURCES = ('k1_goalkeeper/standing_task.py', 'k1_goalkeeper/training_model.py',
           'k1_goalkeeper/training_env.py', 'k1_goalkeeper/training_actions.py',
           'scripts/train_standing.py', 'scripts/evaluate_standing.py', 'requirements-ppo.txt')


def evaluation_cases():
    return [{'name': 'unperturbed', 'force': [0., 0.], 'duration': 30., 'seed': 101},
            {'name': 'push_x_positive', 'force': [15., 0.], 'duration': 10., 'seed': 103},
            {'name': 'push_x_negative', 'force': [-15., 0.], 'duration': 10., 'seed': 107},
            {'name': 'push_y_positive', 'force': [0., 15.], 'duration': 10., 'seed': 109},
            {'name': 'push_y_negative', 'force': [0., -15.], 'duration': 10., 'seed': 113}]


class ExperimentLog(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.episode_terms = {}
        self.episodes = self.rollouts = 0

    def _on_step(self):
        info = self.locals['infos'][0]
        for key, value in info['reward_terms'].items():
            self.episode_terms[key] = self.episode_terms.get(key, 0.)+value
        if self.locals['dones'][0]:
            self.episodes += 1
            with self.path.open('a') as stream:
                stream.write(json.dumps({'episode': self.episodes, 'timesteps': self.num_timesteps,
                                         'reason': info['termination_reason'], 'reward_terms': self.episode_terms,
                                         'monitor': info['episode']})+'\n')
            self.episode_terms = {}
        return True

    def _on_rollout_end(self):
        buffer = self.model.rollout_buffer
        for name in ('observations', 'actions', 'rewards', 'values', 'advantages', 'returns', 'log_probs'):
            if not np.isfinite(getattr(buffer, name)).all():
                raise FloatingPointError(f'Nonfinite PPO buffer: {name}')
        self.rollouts += 1
        print(f'rollout={self.rollouts} steps={self.num_timesteps} episodes={self.episodes}', flush=True)


def train(output, steps=16384, seed=7):
    if steps <= 0 or steps % PPO_CONFIG['n_steps']:
        raise ValueError('steps must be a positive multiple of 512')
    output.mkdir(parents=True, exist_ok=False)
    model_path = build_training_scene(output/'training.xml')
    protocol = {'task_version': TASK_VERSION, 'seed': seed, 'total_steps': steps,
                'algorithm': 'stable_baselines3.PPO / MlpPolicy', 'ppo': PPO_CONFIG,
                'training_episode_s': 5., 'physics_dt_s': .001, 'control_dt_s': .02,
                'action_scale_rad': ACTION_SCALE, 'actor_dimensions': 78,
                'critic': 'same 78 current-state observations; no privileged or future inputs',
                'training_push': {'world_xy_uniform_per_axis_n': [-TRAIN_PUSH_MAX, TRAIN_PUSH_MAX],
                                  'start_s': PUSH_START, 'duration_s': PUSH_DURATION},
                'evaluation_cases': evaluation_cases(), 'evaluation_repeats': 2,
                'evaluation_stable_criteria': {'completed': True, 'max_tilt_degrees': 10., 'max_xy_drift_m': .1},
                'model_sha256': file_hash(model_path),
                'sources': {name: file_hash(ROOT/name) for name in SOURCES},
                'packages': {name: importlib.metadata.version(name) for name in ('mujoco', 'gymnasium', 'torch', 'stable-baselines3', 'numpy')}}
    write_json(output/'protocol.json', protocol)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env = Monitor(StandingTask(mujoco.MjModel.from_xml_path(str(model_path))), str(output/'monitor'))
    try:
        agent = PPO('MlpPolicy', env, seed=seed, device='cpu', **PPO_CONFIG)
        agent.set_logger(configure(str(output/'logs'), ['csv']))
        agent.save(output/'initial')
        before = {name: value.detach().clone() for name, value in agent.policy.state_dict().items()}
        callback = ExperimentLog(output/'episodes.jsonl')
        start = time.monotonic()
        agent.learn(total_timesteps=steps, callback=callback)
        elapsed = time.monotonic()-start
        agent.save(output/'final')
        after = agent.policy.state_dict()
        changed = {name: float(torch.linalg.vector_norm(after[name]-value)) for name, value in before.items()}
        reloaded = PPO.load(output/'final.zip', device='cpu')
        observation, _ = env.reset(seed=999, options={'push_force': [0., 0.]})
        probes = [observation]
        for _ in range(25):
            observation, _, _, _, _ = env.step(np.zeros(22, dtype=np.float32))
            probes.append(observation)
        probes = np.asarray(probes)
        expected, _ = agent.predict(probes, deterministic=True)
        actual, _ = reloaded.predict(probes, deterministic=True)
        np.savez_compressed(output/'reload_check.npz', observations=probes, before_save_actions=expected,
                            reloaded_actions=actual)
        result = {'actual_steps': agent.num_timesteps, 'optimizer_epochs': agent._n_updates,
                  'completed_episodes': callback.episodes, 'rollouts': callback.rollouts,
                  'wall_seconds': elapsed, 'steps_per_second': steps/elapsed,
                  'parameter_l2_changes': changed,
                  'actor_weights_changed': any(v > 0 for k, v in changed.items() if 'policy_net' in k or 'action_net' in k),
                  'parameters_finite': all(bool(torch.isfinite(v).all()) for v in after.values()),
                  'reload_actions_identical': bool(np.array_equal(expected, actual)),
                  'checkpoint_sha256': {name: file_hash(output/f'{name}.zip') for name in ('initial', 'final')}}
        result['passed'] = bool(result['actual_steps'] == steps and result['actor_weights_changed'] and
                                result['parameters_finite'] and result['reload_actions_identical'])
        write_json(output/'training_result.json', result)
        if not result['passed']:
            raise RuntimeError('Training delivery checks failed')
        return result
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(train(args.output.resolve(), args.steps, args.seed), indent=2))
