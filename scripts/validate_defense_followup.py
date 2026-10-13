"""Freeze H3 paired move/stop and low-speed impact experiments after full H1."""
import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from k1_goalkeeper.defense_controller import DefensePulse
from k1_goalkeeper.defense_followup import run_followup_trial
from k1_goalkeeper.defense_validation import contact_state, _pairs_by_step
from k1_goalkeeper.impact_validation import ImpactCase, _launch
from k1_goalkeeper.velocity_validation import trajectory_hash
from scripts.validate_walk import verify_sources

H1 = ROOT/'reports/defense_validation'
OFFSETS = (('left_shoulder_roll_joint', .15), ('right_shoulder_roll_joint', -.15))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trial(path):
    summary = json.loads((path/'summary.json').read_text())
    with np.load(path/'trajectory.npz') as archive:
        trace = {key: archive[key] for key in archive.files}
    if trajectory_hash(trace) != summary['trajectory_sha256']:
        raise RuntimeError(f'Trajectory integrity failure: {path}')
    return summary, trace


def require_h1_gate():
    result = json.loads((H1/'results.json').read_text())
    if (not result['gate_passed'] or not result['zero_parity']['passed']
            or not result['original_sources_unchanged'] or result['protocol_sha256'] != sha(H1/'protocol.json')):
        raise RuntimeError('Full frozen H1 gate has not passed')
    needed = [s for s in result['trials'] if s['pulse'] and
              abs(s['pulse']['joint_offsets'][0][1]) in (.05, .15, .30)]
    if len(needed) != 24 or not all(s['passed'] and s['repeat_identical'] for s in needed):
        raise RuntimeError('H1 requires every family/amplitude x2 before H3')
    return {'results_sha256': sha(H1/'results.json'), 'protocol_sha256': result['protocol_sha256'],
            'gate_passed': True, 'trial_count': len(needed)}


def freeze(output):
    output.mkdir(parents=True, exist_ok=True)
    h1_gate = require_h1_gate()
    model = mujoco.MjModel.from_xml_path(str(H1/'scene_free.xml'))
    references = [load_trial(H1/f'both_roll_0.15_r{repeat}') for repeat in range(2)]
    if not all(summary['passed'] for summary, _ in references) or references[0][0]['trajectory_sha256'] != references[1][0]['trajectory_sha256']:
        raise RuntimeError('Native aim needs verified identical both_roll .15 H1 references')
    reference, trace = references[0]
    index = int(np.argmin(np.abs(trace['time']-6.)))
    if abs(trace['time'][index]-6.) > 1e-9:
        raise RuntimeError('H1 reference is missing exact t6 qpos')
    data = mujoco.MjData(model)
    data.qpos[:] = trace['qpos'][index]
    mujoco.mj_kinematics(model, data)
    launches = {}
    for speed in (1., 2.):
        aim_case = ImpactCase(f'freeze_native_arm_{speed:g}', target_body='left_elbow_yaw_link', speed=speed)
        launches[str(speed)] = _launch(model, data, aim_case, model.geom('ball_collision').id)
    protocol = {'stage': 'H3', 'repeats': 2, 'seed': 0, 'native_collision_only': True,
        'H4': 'not_implemented', 'H1_gate': h1_gate,
        'physics_dt_s': .001, 'policy_dt_s': .02,
        'selected_action': {'family': 'both_roll', 'amplitude_rad': .15, 'offsets': OFFSETS},
        'pulse_curve': 'unchanged DefensePulse quintic, peak slew .140625 rad/s',
        'motion': {'directions_body_vy_m_s': [.2, -.2], 'end_s': 23.,
            'pulse': asdict(DefensePulse(OFFSETS, start_s=6.)),
            'body_segments': {'lateral': [2., 3.], 'zero_before_pulse': [3., 6.],
                'pulse_and_recovery': [6., 17.], 'forward_vx_0_4': [17., 18.], 'final_zero': [18., 23.]},
            'recovery_windows': [[12., 17.], [18., 23.]],
            'matched_baseline': 'same entire body command timeline without arm intervention'},
        'impact': {'speeds_m_s': [1., 2.], 'launch_time_s': 6., 'end_s': 14.,
            'moving_duration_s': 4., 'pulse': asdict(DefensePulse(OFFSETS)),
            'target_body': 'left_elbow_yaw_link', 'recovery_windows': [[9., 14.]],
            'launches': launches, 'same_launch_for_baseline_and_arms': True,
            'reference': {'path': 'reports/defense_validation/both_roll_0.15_r0',
                'trajectory_sha256': reference['trajectory_sha256'], 'archive_index': index,
                'archive_time_s': float(trace['time'][index]),
                'summary_file_sha256': sha(H1/'both_roll_0.15_r0/summary.json'),
                'trajectory_file_sha256': sha(H1/'both_roll_0.15_r0/trajectory.npz')},
            'construction': 'independent static FK of already passed H1 t6 qpos then unchanged E _launch; predeclared fixed scene input, not an online predictor',
            'escalation': '1m/s arms must pass safety+response+recovery+native target contact x2 before 2m/s; miss is unverifiable, not unsafe'},
        'guards': {'start_s': 2., 'minimum_base_z_m': .42, 'maximum_tilt_degrees': 15.,
            'selected_target_margin_rad': .05, 'selected_actual_q': 'inside XML range',
            'continuous_arm_raw_saturation_s': .10, 'control_torque_limit_tolerance_nm': 1e-5,
            'new_self_contact': 'stop against same-step matched baseline',
            'new_nonfoot_ground_contact': 'stop against same-step matched baseline',
            'robot_ball': 'allowed only in impact', 'robot_goal_or_other_external': 'stop',
            'nonfinite': 'true numerical failure; deliberate contact/balance/target guard is censored'},
        'response': 'every selected signed median actual E qpos delta vs same-time baseline in final1s hold >=50% amplitude',
        'recorder': 'unchanged E full recorder; add only policy-tick nominal/delivered/offset plus active self/nonfoot/external pair rows',
        'implementation_sha256': {str(path.relative_to(ROOT)): sha(path) for path in
            (ROOT/'k1_goalkeeper/defense_followup.py', Path(__file__), ROOT/'k1_goalkeeper/defense_controller.py')},
        'H1_helper_sha256': {name: hashlib.sha256(inspect.getsource(helper).encode()).hexdigest()
                           for name, helper in (('contact_state', contact_state), ('_pairs_by_step', _pairs_by_step))},
        'scene_sha256': sha(H1/'scene_free.xml'),
        'original_protected_manifest_sha256': sha(H1/'protected_manifest.json')}
    path = output/'protocol.json'
    payload = json.dumps(protocol, indent=2, allow_nan=False)+'\n'
    if path.exists() and path.read_text() != payload:
        raise RuntimeError('Frozen H3 protocol differs; do not overwrite formal protocol')
    if not path.exists():
        path.write_text(payload)
    return protocol


def run(output, phase='all'):
    output = Path(output)
    protocol = freeze(output)
    verify_sources()
    protected = json.loads((H1/'protected_manifest.json').read_text())
    if not all(sha(ROOT/path) == expected for path, expected in protected.items()):
        raise RuntimeError('Protected A-G/source evidence changed')
    protocol_hash = sha(output/'protocol.json')
    model = mujoco.MjModel.from_xml_path(str(H1/'scene_free.xml'))
    result_path = output/'results.json'
    results = json.loads(result_path.read_text()) if result_path.exists() else {
        'stage': 'H3', 'protocol_sha256': protocol_hash, 'groups': [],
        'motion_gate_passed': False, 'low_speed_impact_gate_passed': False,
        'impact_gate_passed': False, 'gate_passed': False, 'H4': 'root_decides_after_H3'}
    if results['protocol_sha256'] != protocol_hash:
        raise RuntimeError('H3 result/protocol fingerprint mismatch')

    def save():
        verify_sources()
        unchanged = all(sha(ROOT/path) == expected for path, expected in protected.items())
        implementation = all(sha(ROOT/path) == expected for path, expected in protocol['implementation_sha256'].items())
        helper_unchanged = all(hashlib.sha256(inspect.getsource(helper).encode()).hexdigest() == protocol['H1_helper_sha256'][name]
                              for name, helper in (('contact_state', contact_state), ('_pairs_by_step', _pairs_by_step)))
        if not unchanged or not implementation or not helper_unchanged or sha(output/'protocol.json') != protocol_hash:
            raise RuntimeError('Frozen H3/protected evidence changed during execution')
        results['original_protected_unchanged'] = unchanged
        results['all_executed_repeats_identical'] = all(g['repeat_identical'] for g in results['groups'])
        result_path.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')

    def paired_group(name, case, pulse, launch, segments, recoveries, impact=False):
        old = next((g for g in results['groups'] if g['name'] == name), None)
        if old is not None:
            for row in old['runs']:
                summary, _ = load_trial(output/row['name'])
                if summary['trajectory_sha256'] != row['trajectory_sha256']:
                    raise RuntimeError('Saved H3 result integrity failure')
            return old
        rows, baseline_traces, mode_hashes = [], [], {}
        for mode in ('baseline', 'arms'):
            hashes = []
            for repeat in range(2):
                run_name = f'{name}_{mode}_r{repeat}'
                run_case = ImpactCase(**{**asdict(case), 'name': run_name})
                summary, trace = run_followup_trial(model, run_case,
                    pulse=pulse if mode == 'arms' else None, launch=launch, segments=segments,
                    recovery_windows=recoveries, baseline=baseline_traces[repeat] if mode == 'arms' else None,
                    allow_ball_contact=impact, output=output/run_name)
                if mode == 'baseline':
                    baseline_traces.append(trace)
                hashes.append(summary['trajectory_sha256'])
                rows.append({key: value for key, value in summary.items() if key != 'e_summary'})
                print(run_name, 'stable', summary['stable'], 'response', summary['authority_passed'],
                      'recovered', summary['recovered'], 'contact', summary['contact_verification'],
                      'stop', summary['stop'], flush=True)
            mode_hashes[mode] = hashes
            if mode == 'baseline' and (hashes[0] != hashes[1] or not all(row['passed'] for row in rows)):
                break
        identical = all(len(values) == 2 and values[0] == values[1] for values in mode_hashes.values())
        group = {'name': name, 'runs': rows, 'hashes_by_mode': mode_hashes,
                 'repeat_identical': identical, 'passed': bool(identical and len(rows) == 4 and all(row['passed'] for row in rows))}
        results['groups'].append(group)
        save()
        return group

    if phase in ('motion', 'all'):
        for vy in protocol['motion']['directions_body_vy_m_s']:
            group = paired_group(f'move_stop_vy_{vy:+.2f}', ImpactCase('motion', duration=21.),
                DefensePulse(OFFSETS, start_s=6.),
                {'position': [7., 0., .111], 'velocity': [0., 0., 0.], 'angular_velocity': [0., 0., 0.]},
                [(2., 3., (0., vy, 0.)), (17., 18., (.4, 0., 0.))], [(12., 17.), (18., 23.)])
            if not group['passed']:
                results['blocked_reason'] = group['name']+' failed motion gate'
                save()
                return results
        results['motion_gate_passed'] = True
        save()
    if phase in ('impact', 'all'):
        if not results['motion_gate_passed']:
            raise RuntimeError('H3 motion gate must pass before impacts')
        for speed in protocol['impact']['speeds_m_s']:
            group = paired_group(f'native_arm_impact_{speed:.1f}',
                ImpactCase('impact', target_body='left_elbow_yaw_link', speed=speed,
                           moving_duration=4., duration=8.), DefensePulse(OFFSETS),
                protocol['impact']['launches'][str(speed)], [], [(9., 14.)], impact=True)
            if not group['passed']:
                results['blocked_reason'] = group['name']+' failed or unverifiable; no higher speed'
                save()
                return results
            if speed == 1.:
                results['low_speed_impact_gate_passed'] = True
                save()
        results['impact_gate_passed'] = True
        results['gate_passed'] = results['motion_gate_passed'] and results['impact_gate_passed']
        save()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=H1/'followup')
    parser.add_argument('--phase', choices=('all', 'motion', 'impact'), default='all')
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    if args.freeze_only:
        freeze(args.output)
        print('H3 protocol frozen; no dynamics executed')
    else:
        report = run(args.output, args.phase)
        print(json.dumps({key: report[key] for key in ('motion_gate_passed', 'low_speed_impact_gate_passed', 'gate_passed')}))
