"""Evaluate saved PPO checkpoints and fixed posture on frozen standing cases."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from k1_goalkeeper.standing_task import StandingTask
from scripts.validate_training_env import array_hash, file_hash, stacked, write_json


def run_case(model, agent, case, output, criteria):
    output.mkdir(parents=True, exist_ok=False)
    with StandingTask(model, episode_duration=case['duration']) as task:
        obs, _ = task.reset(seed=case['seed'], options={'push_force': case['force']})
        env, mapping = task.env, task.env.mapping
        physics, contacts, decisions = [], [], []

        def snapshot():
            return {'time': float(env.data.time), 'qpos': env.data.qpos.copy(),
                    'qvel': env.data.qvel.copy(),
                    'torque': env.data.qfrc_actuator[mapping.dof_addresses].copy(),
                    'push_force': env.data.xfrc_applied[task.trunk, :2].copy()}

        physics.append(snapshot())
        original_step = env.backend.step_physics

        def record_physics():
            original_step()
            physics.append(snapshot())
            for index, contact in enumerate(env.data.contact):
                if contact.efc_address < 0:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, env.data, index, force)
                if force[0] > 1e-6:
                    pair = [int(contact.geom1), int(contact.geom2)]
                    self_contact = all(g in env.robot_geoms for g in pair)
                    contacts.append([float(env.data.time), *pair, float(contact.dist), float(force[0]), int(self_contact)])

        env.backend.step_physics = record_physics
        while True:
            action = (np.zeros(22, dtype=np.float32) if agent is None else
                      agent.predict(obs, deterministic=True)[0])
            previous = obs.copy()
            obs, reward, terminated, truncated, info = task.step(action)
            decisions.append({'time': float(env.data.time), 'observation': previous,
                              'action': np.asarray(action).copy(), 'next_observation': obs.copy(),
                              'reward': reward, 'terminated': terminated, 'truncated': truncated})
            if terminated or truncated:
                break
        arrays = stacked(physics)
        arrays['contacts'] = np.asarray(contacts, dtype=float).reshape(-1, 6)
        transitions = stacked(decisions)
        np.savez_compressed(output/'physics.npz', **arrays)
        np.savez_compressed(output/'transitions.npz', **transitions)
        # Read delivered arrays independently before reporting physical metrics.
        with np.load(output/'physics.npz') as saved:
            q = mapping.base_qpos_address
            position = saved['qpos'][:, q:q+3]
            quat = saved['qpos'][:, q+3:q+7]
            upright = 1.-2.*(quat[:, 1]**2+quat[:, 2]**2)
            tilt = np.degrees(np.arccos(np.clip(upright, -1, 1)))
            drift = np.linalg.norm(position[:, :2]-task.origin[:2], axis=1)
            impulse = saved['push_force'][1:].sum(axis=0)*.001
            completed = info['termination_reason'] == 'time_limit' and np.isclose(saved['time'][-1], case['duration'])
            summary = {'case': case, 'completed': bool(completed), 'duration_s': float(saved['time'][-1]),
                       'termination_reason': info['termination_reason'],
                       'maximum_tilt_degrees': float(tilt.max()), 'rms_tilt_degrees': float(np.sqrt(np.mean(tilt**2))),
                       'maximum_xy_drift_m': float(drift.max()), 'minimum_base_height_m': float(position[:, 2].min()),
                       'return': float(transitions['reward'].sum()),
                       'actual_push_impulse_ns': impulse.tolist(),
                       'push_delivery_matches': bool(np.allclose(impulse, np.asarray(case['force'])*.1, atol=1e-10)),
                       'positive_force_self_contact_samples': int(saved['contacts'][:, 5].sum()),
                       'torque_saturation_fraction': float(np.mean(np.abs(saved['torque']) >= .99*mapping.effort)),
                       'finite_physics': all(bool(np.isfinite(saved[k]).all()) for k in saved.files),
                       'bad_warning_count': sum(int(env.data.warning[w].number) for w in
                                                (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                                                 mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL))}
        summary['stable'] = bool(completed and summary['maximum_tilt_degrees'] <= criteria['max_tilt_degrees']
                                 and summary['maximum_xy_drift_m'] <= criteria['max_xy_drift_m'])
        summary['array_sha256'] = {'physics': array_hash(arrays), 'transitions': array_hash(transitions)}
        summary['file_sha256'] = {name: file_hash(output/f'{name}.npz') for name in ('physics', 'transitions')}
        write_json(output/'summary.json', summary)
        return summary


def evaluate(run, output):
    protocol = json.loads((run/'protocol.json').read_text())
    training = json.loads((run/'training_result.json').read_text())
    if file_hash(run/'training.xml') != protocol['model_sha256']:
        raise ValueError('Training model hash changed')
    for source, expected in protocol['sources'].items():
        if file_hash(ROOT/source) != expected:
            raise ValueError(f'Experiment source changed: {source}')
    for name, expected in training['checkpoint_sha256'].items():
        if file_hash(run/f'{name}.zip') != expected:
            raise ValueError(f'Checkpoint changed: {name}')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    agents = {'reference': None, **{name: PPO.load(run/f'{name}.zip', device='cpu') for name in ('initial', 'final')}}
    model = mujoco.MjModel.from_xml_path(str(run/'training.xml'))
    results = []
    for name, agent in agents.items():
        for case in protocol['evaluation_cases']:
            repeats = [run_case(model, agent, case, output/name/case['name']/f'repeat_{r}',
                                protocol['evaluation_stable_criteria']) for r in range(protocol['evaluation_repeats'])]
            repeated = all(r['array_sha256'] == repeats[0]['array_sha256'] for r in repeats)
            result = {'controller': name, 'case': case['name'], 'deterministic': repeated, 'repeats': repeats}
            results.append(result)
            print(name, case['name'], 'stable=', repeats[0]['stable'], 'tilt=', round(repeats[0]['maximum_tilt_degrees'], 3), flush=True)
    report = {'protocol_sha256': file_hash(run/'protocol.json'), 'results': results,
              'controller_summary': {name: {'stable_cases': sum(r['repeats'][0]['stable'] for r in results if r['controller'] == name),
                                            'total_cases': len(protocol['evaluation_cases'])} for name in agents},
              'verification_passed': all(r['deterministic'] and all(s['finite_physics'] and s['push_delivery_matches']
                                                                   and s['bad_warning_count'] == 0 for s in r['repeats']) for r in results)}
    write_json(output/'results.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.run.resolve(), args.output.resolve())
    if not report['verification_passed']:
        raise SystemExit('Evaluation verification failed; see results.json')
