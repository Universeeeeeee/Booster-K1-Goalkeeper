"""Serial I1 interface validation with separate causal and offline archives."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from k1_goalkeeper.scene import build_scene
from k1_goalkeeper.simulation import Shot
from k1_goalkeeper.training_actions import BAD_WARNINGS
from k1_goalkeeper.training_env import K1GoalkeeperEnv


PROTECTED = ROOT / 'reports/training_interface_validation/protected_manifest.json'
SOURCES = ('k1_goalkeeper/training_actions.py', 'k1_goalkeeper/training_env.py',
           'scripts/validate_training_env.py', 'requirements-training.txt',
           'requirements-walk.txt', 'requirements.txt')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def verify_protected():
    manifest = json.loads(PROTECTED.read_text())
    changed = [name for name, digest in manifest.items() if file_hash(ROOT/name) != digest]
    if changed:
        raise RuntimeError(f'Protected evidence changed: {changed}')
    return {'file_count': len(manifest), 'manifest_sha256': file_hash(PROTECTED), 'all_match': True}


def case_definitions():
    return [
        {'name': 'body_standing', 'mode': 'body_velocity', 'task': 'standing', 'time_limit': 30.},
        {'name': 'joint_standing', 'mode': 'joint_targets', 'task': 'standing', 'time_limit': 30.},
        {'name': 'body_single_goal', 'mode': 'body_velocity', 'task': 'single_shot',
         'time_limit': 8., 'launch_time': 2.,
         'shot': asdict(Shot('body_single_goal', (7., .8, .111), (4., 0., 0.),
                             (0., 4./.11, 0.), duration=6., seed=7))},
        {'name': 'joint_single_goal', 'mode': 'joint_targets', 'task': 'single_shot',
         'time_limit': 1., 'launch_time': 0.,
         'shot': asdict(Shot('joint_single_goal', (10.9, 0., .5), (8., 0., 0.),
                             duration=1., seed=7))},
        {'name': 'body_single_contact', 'mode': 'body_velocity', 'task': 'single_shot',
         'time_limit': 8., 'launch_time': 2.,
         'shot': asdict(Shot('body_single_contact', (7., 0., .111), (2., 0., 0.),
                             (0., 2./.11, 0.), duration=6., seed=7))},
    ]


def options_for(case):
    return {key: case[key] for key in ('task', 'time_limit', 'shot', 'launch_time') if key in case}


def stacked(rows):
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def physics_snapshot(env):
    d, m = env.data, env.mapping
    q, v, bq, bv = m.base_qpos_address, m.base_dof_address, m.ball_qpos_address, m.ball_dof_address
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, d.qpos[q+3:q+7])
    rotation = rotation.reshape(3, 3)
    return {'time': float(d.time), 'qpos': d.qpos.copy(), 'qvel': d.qvel.copy(),
            'ctrl': d.ctrl.copy(), 'ball_position': d.qpos[bq:bq+3].copy(),
            'ball_velocity': d.qvel[bv:bv+3].copy(),
            'ball_angular_velocity_body': d.qvel[bv+3:bv+6].copy(),
            'base_position': d.qpos[q:q+3].copy(), 'base_quaternion': d.qpos[q+3:q+7].copy(),
            'body_linear_velocity': rotation.T @ d.qvel[v:v+3],
            'body_angular_velocity': d.qvel[v+3:v+6].copy(), 'upright_z': rotation[2, 2],
            'motor_controls': d.ctrl[m.actuator_ids].copy(),
            'applied_torque': d.qfrc_actuator[m.dof_addresses].copy()}


def run_case(model, case, directory):
    directory.mkdir(parents=True, exist_ok=True)
    env = K1GoalkeeperEnv(model, action_mode=case['mode'])
    observation, info = env.reset(seed=7, options=options_for(case))
    actor = [{'time': info['returned_observation_time'], 'observation': observation.copy(),
              'action': np.zeros_like(env.default_action), 'reward': 0., 'terminated': False,
              'truncated': False, 'transition_valid': True, 'is_transition': False}]
    critic = [{'time': float(env.data.time), **env.get_privileged_state()}]
    evaluation = [physics_snapshot(env)]
    contact_rows = []
    original_step = env.backend.step_physics
    initial_overlap = False
    launch_seen = False

    def record_physics():
        nonlocal initial_overlap, launch_seen
        # At scheduled teleport time, replace the parked-ball boundary sample.
        current = physics_snapshot(env)
        if current['time'] == evaluation[-1]['time']:
            evaluation[-1] = current
        if env._launched and not launch_seen:
            launch_seen = True
            initial_overlap = any(c.dist < 0 and env.ball_geom in (c.geom1, c.geom2)
                                  and any(g in env.robot_geoms for g in (c.geom1, c.geom2))
                                  for c in env.data.contact[:env.data.ncon])
        original_step()
        evaluation.append(physics_snapshot(env))
        for index, contact in enumerate(env.data.contact[:env.data.ncon]):
            if contact.efc_address < 0:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, env.data, index, force)
            if force[0] > 1e-6:
                g1, g2 = int(contact.geom1), int(contact.geom2)
                ball_robot = env.ball_geom in (g1, g2) and any(g in env.robot_geoms for g in (g1, g2))
                contact_rows.append([float(env.data.time), g1, g2, float(force[0]), int(ball_robot)])

    env.backend.step_physics = record_physics
    infos = []
    while True:
        action = env.default_action.copy()
        observation, reward, terminated, truncated, info = env.step(action)
        actor.append({'time': info['returned_observation_time'], 'observation': observation.copy(),
                      'action': action, 'reward': reward, 'terminated': terminated,
                      'truncated': truncated, 'transition_valid': info['transition_valid'],
                      'is_transition': True})
        if info['transition_valid']:
            critic.append({'time': float(env.data.time), **env.get_privileged_state()})
        infos.append(info)
        if terminated or truncated:
            break
    state = env.get_evaluation_state()
    actor_arrays, critic_arrays, evaluation_arrays = stacked(actor), stacked(critic), stacked(evaluation)
    evaluation_arrays['native_contacts'] = np.asarray(contact_rows, dtype=float).reshape(-1, 5)
    for name, arrays in (('actor', actor_arrays), ('critic', critic_arrays), ('evaluation', evaluation_arrays)):
        np.savez_compressed(directory/f'{name}.npz', **arrays)
    contacts = evaluation_arrays['native_contacts']
    ball_contacts = contacts[contacts[:, 4] == 1]
    first_contact = float(ball_contacts[0, 0]) if len(ball_contacts) else None
    event = state['exit_event']
    valid = all(row['transition_valid'] for row in infos)
    continued = first_contact is not None and state['time'] > first_contact+.001+1e-9
    contact_pair_ids = sorted({(int(row[1]), int(row[2])) for row in ball_contacts})
    summary = {'case': case, 'seed': 7, 'action_semantics': 'constant backend default_action',
               'actual_outcome': event['outcome'] if event else info['termination_reason'],
               'exit_event': event, 'final_info': info, 'right_goal': bool(event and event['outcome'] == 'Goal' and event['goal'] == 'right'),
               'scheduled_shot': state['scheduled_shot'], 'scheduled_launch_time': state['scheduled_launch_time'],
               'scheduled_initial_velocity_norm_m_s': float(np.linalg.norm(state['scheduled_shot']['velocity'])) if state['scheduled_shot'] else None,
               'delivered_launch': state['delivered_launch'], 'launch_initial_robot_overlap': initial_overlap,
               'robot_contact_seen': bool(info['robot_contact_seen']),
               'native_contact_flag_matches_info': bool(bool(len(ball_contacts)) == bool(info['robot_contact_seen'])),
               'native_ball_robot_contact_samples': len(ball_contacts), 'first_ball_robot_contact_time': first_contact,
               'contact_did_not_immediately_terminate': bool(continued),
               'contact_geom_pairs': [[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in pair] for pair in contact_pair_ids],
               'maximum_ball_robot_normal_force_n': float(ball_contacts[:, 3].max()) if len(ball_contacts) else 0.,
               'minimum_base_height_m': float(evaluation_arrays['base_position'][:, 2].min()),
               'maximum_tilt_degrees': float(np.degrees(np.arccos(np.clip(evaluation_arrays['upright_z'], -1, 1))).max()),
               'maximum_absolute_motor_control_nm': np.abs(evaluation_arrays['motor_controls']).max(axis=0).tolist(),
               'warning_counts': {w.name: int(env.data.warning[w].number) for w in BAD_WARNINGS},
               'all_transitions_valid': valid, 'standing_capability_claim': False,
               'save_or_recovery_claim': False,
               'channel_array_sha256': {name: array_hash(arrays) for name, arrays in
                                        (('actor', actor_arrays), ('critic', critic_arrays), ('evaluation', evaluation_arrays))},
               'archive_sha256': {name: file_hash(directory/f'{name}.npz') for name in ('actor', 'critic', 'evaluation')}}
    summary['interface_passed'] = bool(valid and not initial_overlap and summary['native_contact_flag_matches_info'] and
        np.isfinite(actor_arrays['observation']).all() and
        np.isclose(evaluation_arrays['time'][-1], info['physics_steps']*.001, rtol=0, atol=1e-9))
    if case['name'].endswith('single_goal'):
        summary['interface_passed'] &= summary['right_goal']
    if case['name'] == 'body_standing':
        summary['interface_passed'] &= info['termination_reason'] == 'time_limit'
    if case['name'] == 'body_single_contact':
        summary['interface_passed'] &= bool(len(ball_contacts) and continued)
    write_json(directory/'evaluation_summary.json', summary)
    env.close()
    return summary


def prefix_proofs(model, mode):
    """Four alternating reset episodes; pending shots cannot change causal channels."""
    env = K1GoalkeeperEnv(model, action_mode=mode)
    shots = (Shot('pending_a', (7., .8, .111), (4., 0., 0.)),
             Shot('pending_b', (8., -.9, .5), (8., 0., 1.)))
    records = []
    for index in (0, 1, 0, 1):
        observation, info = env.reset(seed=7, options={'task': 'single_shot', 'shot': shots[index],
                                                     'launch_time': 2., 'time_limit': 3.})
        actors, critics, infos = [observation.copy()], [env.get_privileged_state()], [info]
        reset_state = {'qpos': env.data.qpos.copy(), 'qvel': env.data.qvel.copy(), 'ctrl': env.data.ctrl.copy()}
        for _ in range(50):
            observation, _, terminated, truncated, info = env.step(env.default_action.copy())
            if terminated or truncated:
                raise RuntimeError('Prefix proof ended before the one-second observation window')
            actors.append(observation.copy())
            critics.append(env.get_privileged_state())
            infos.append(info)
        policy_state = {}
        simulation = getattr(env.backend, 'simulation', None)
        if simulation is not None:
            controller = simulation.controller
            policy_state = {'history': controller.policy.obs_history.numpy().copy(),
                            'last_action': controller.policy.last_action.numpy().copy(),
                            'filter': controller.policy.filtered_dof_target.numpy().copy(),
                            'controller_steps': np.asarray(controller._step_count),
                            'controller_elapsed_s': np.asarray(controller._elapsed_s)}
        records.append({'pending_index': index, 'actor_sha256': array_hash({'observation': np.asarray(actors)}),
                        'critic_sha256': array_hash(stacked(critics)),
                        'info_sha256': hashlib.sha256(json.dumps(infos, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                        'reset_state_sha256': array_hash(reset_state), 'policy_state_sha256': array_hash(policy_state),
                        'scheduled_shot': env.get_evaluation_state()['scheduled_shot']})
    keys = ('actor_sha256', 'critic_sha256', 'info_sha256', 'reset_state_sha256', 'policy_state_sha256')
    same = all(len({row[key] for row in records}) == 1 for key in keys)
    offline_different = records[0]['scheduled_shot'] != records[1]['scheduled_shot']
    env.close()
    return {'mode': mode, 'seed': 7, 'prefix_seconds': 1., 'reset_sequence': [0, 1, 0, 1],
            'causal_channels_and_policy_history_identical': same,
            'offline_pending_shots_different': offline_different,
            'passed': bool(same and offline_different), 'episodes': records}


def gym_checks(model):
    results = []
    for mode in ('joint_targets', 'body_velocity'):
        env = K1GoalkeeperEnv(model, action_mode=mode)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            check_env(env, skip_render_check=True)
        results.append({'mode': mode, 'passed': True, 'advisory_warnings': [str(w.message) for w in caught]})
        env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/training_interface_validation')
    parser.add_argument('--case', action='append', choices=[c['name'] for c in case_definitions()],
                        help='Run selected cases for a temporary pilot; omitted runs all five')
    parser.add_argument('--skip-proofs', action='store_true', help='Temporary pilot only')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists() or (output/'results.json').exists():
        raise RuntimeError('Use a fresh output directory; existing protocol/results are never overwritten')
    before = verify_protected()
    import torch
    torch.set_num_threads(1)
    cases = [c for c in case_definitions() if not args.case or c['name'] in args.case]
    for case in cases:
        if 'shot' in case:
            case['initial_velocity_norm_m_s'] = float(np.linalg.norm(case['shot']['velocity']))
    code = {name: file_hash(ROOT/name) for name in SOURCES}
    versions = {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in
                                                       ('numpy', 'mujoco', 'gymnasium', 'torch')}}
    protocol = {'stage': 'I1', 'scope': 'Serial interface validation; no training, optimization, collision modification or throughput benchmark',
                'seed': 7, 'repetitions': 2, 'physics_dt_s': .001, 'control_dt_s': .02,
                'torch_threads': 1,
                'cases': cases, 'source_sha256': code, 'versions': versions,
                'protected_before': before, 'command': sys.argv, 'pilot': bool(args.case or args.skip_proofs),
                'channels': {'actor': 'Current simulation truth observations, prior delivered action, transition reward/flags',
                             'critic': 'Current exact privileged state only',
                             'evaluation': 'Actual native physics and final offline labels; never actor input'},
                'joint_standing_expectation': 'Valid fall is an interface result; default PD targets are not a learned standing policy',
                'contact_expectation': 'Native positive-force ball/robot contact and continued physics; no save/recovery claim',
                'gym_advisories': 'Raw physical actions are not normalized; unrestricted current-state observations have infinite Box bounds'}
    write_json(output/'protocol.json', protocol)
    model = mujoco.MjModel.from_xml_path(str(build_scene(output/'scene.xml')))
    results = {'stage': 'I1', 'protocol_sha256': file_hash(output/'protocol.json'),
               'scene_sha256': file_hash(output/'scene.xml'),
               'model_dimensions': {'nq': model.nq, 'nv': model.nv, 'nu': model.nu}, 'cases': []}
    for case in cases:
        runs = [run_case(model, case, output/f"{case['name']}_r{repeat}") for repeat in range(2)]
        same = runs[0]['channel_array_sha256'] == runs[1]['channel_array_sha256']
        results['cases'].append({'name': case['name'], 'repeats_identical': same, 'runs': runs,
                                 'passed': bool(same and all(run['interface_passed'] for run in runs))})
        print(json.dumps({'case': case['name'], 'outcome': runs[0]['actual_outcome'],
                          'contact': runs[0]['robot_contact_seen'], 'passed': results['cases'][-1]['passed']}), flush=True)
    if not args.skip_proofs:
        results['gym_checks'] = gym_checks(model)
        results['no_future_leak_and_reset'] = [prefix_proofs(model, mode) for mode in ('joint_targets', 'body_velocity')]
    results['source_unchanged'] = code == {name: file_hash(ROOT/name) for name in SOURCES}
    results['protected_after'] = verify_protected()
    results['all_repeats_identical'] = all(case['repeats_identical'] for case in results['cases'])
    results['all_interface_checks_passed'] = bool(all(case['passed'] for case in results['cases']) and
        results['source_unchanged'] and all(row['passed'] for row in results.get('no_future_leak_and_reset', [])))
    write_json(output/'results.json', results)
    print(json.dumps({'output': str(output), 'all_interface_checks_passed': results['all_interface_checks_passed']}), flush=True)
    if not results['all_interface_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
