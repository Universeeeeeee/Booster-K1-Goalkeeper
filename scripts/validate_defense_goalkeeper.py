"""H4 four-way repeated comparison, unlocked only by H1 and H3 gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco

from k1_goalkeeper.ball_prediction import build_ball_floor_model
from k1_goalkeeper.defense_goalkeeper import run_h_trial
from k1_goalkeeper.positioning_validation import run_trial, save_trial
from k1_goalkeeper.predictive_goalkeeper import run_g_trial
from k1_goalkeeper.scene import build_scene


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, h3_evidence):
    hroot = ROOT/'reports/defense_validation'
    h1 = json.loads((hroot/'results.json').read_text())
    h3 = json.loads(h3_evidence.read_text())
    if not h1['gate_passed'] or not h3['gate_passed']:
        raise RuntimeError('H4 requires completed H1 and H3 gates')
    output.mkdir(parents=True, exist_ok=True)
    protected = json.loads((hroot/'protected_manifest.json').read_text())
    if any(digest(ROOT/p) != value for p, value in protected.items()):
        raise RuntimeError('A-G protected evidence changed')
    original = json.loads((ROOT/'reports/prediction_validation/protocol.json').read_text())
    names = ('air_high', 'air_wide', 'air_center', 'air_low_right', 'air_high_left')
    shots = {shot['name']: shot for group in ('regression', 'holdout') for shot in original[group]}
    code = ['k1_goalkeeper/defense_goalkeeper.py', 'k1_goalkeeper/defense_controller.py',
            'k1_goalkeeper/defense_validation.py', 'scripts/validate_defense_goalkeeper.py']
    protocol = {'stage': 'H4', 'repeats': 2, 'shots': [shots[n] for n in names],
                'modes': ['absent', 'passive', 'g_active', 'h_stop_then_arm'],
                'h1_sha256': digest(hroot/'results.json'), 'h3_sha256': digest(h3_evidence),
                'code_sha256': {p: digest(ROOT/p) for p in code},
                'action': 'native both shoulder roll +0.15/-0.15; 2s quintic rise,2s hold,2s return',
                'activation': 'current valid in-aperture G forecast and positive conservative robot-plane deadline; actual robot slow>=1s; shorter-than-2s rise budget labeled late partial experiment, not claimed feasible save; no future actual trajectory',
                'motion': 'suspend G translation from arm request through single-shot episode end; no untested simultaneous gait or return to stale G positioning state',
                'timing': 'offline simulation; synchronous wall latency measured in H0, not injected here',
                'success': 'absent goes right goal, robot contact, completed non-goal trajectory; contact alone not success',
                'native_only': True}
    path = output/'protocol.json'
    payload = json.dumps(protocol, indent=2)+'\n'
    if path.exists() and path.read_text() != payload:
        raise RuntimeError('existing frozen H4 protocol differs')
    path.write_text(payload)
    models = {mode: mujoco.MjModel.from_xml_path(str(build_scene(output/f'scene_{mode}.xml', robot_mode=mode)))
              for mode in ('free', 'absent')}
    predictor = build_ball_floor_model(output/'scene_absent.xml')
    report = {'stage': 'H4', 'protocol_sha256': digest(path), 'shots': [], 'all_repeats_identical': True}
    for name in names:
        results = {}
        for mode in protocol['modes']:
            runs = []
            for repeat in range(2):
                trial_name = f'{name}_{mode}_r{repeat}'
                launch = shots[name]['launch']
                if mode == 'h_stop_then_arm':
                    summary, trace = run_h_trial(models['free'], predictor, trial_name, launch)
                elif mode == 'g_active':
                    summary, trace = run_g_trial(models['free'], predictor, trial_name, launch)
                else:
                    summary, trace = run_trial(models['absent' if mode == 'absent' else 'free'], trial_name, mode, launch)
                save_trial(output/trial_name, summary, trace)
                runs.append(summary)
            repeat_ok = runs[0]['trajectory_sha256'] == runs[1]['trajectory_sha256']
            report['all_repeats_identical'] &= repeat_ok
            s = runs[0]
            robot_contacts = [contact for contact in s['contacts'] if contact['kind'] == 'robot']
            results[mode] = {'paths': [str((output/f'{name}_{mode}_r{r}').relative_to(ROOT)) for r in range(2)],
                'trajectory_hashes': [row['trajectory_sha256'] for row in runs], 'repeat_identical': repeat_ok,
                'completed': s['completed'], 'contact': s['effective_contact'], 'right_goal': s['right_goal'],
                'robot_contacts': robot_contacts, 'fall_events': s['fall_events'], 'numerical_error': s['numerical_error'],
                'policy_stop': s['policy_stop'], 'unexpected_contacts': s['unexpected_contacts'],
                'h_guard': s.get('h_guard'), 'action_timing': s.get('action_timing'),
                'minimum_base_height_m': s.get('minimum_base_height_m'),
                'maximum_tilt_degrees': s.get('maximum_tilt_degrees'),
                'arm_return_recovery': s.get('arm_return_recovery'),
                'final_stop_recovery': s.get('final_stop_recovery')}
            print(name, mode, 'contact', s['effective_contact'], 'goal', s['right_goal'], 'repeat', repeat_ok, flush=True)
        absent_goal = results['absent']['right_goal']
        for mode, metrics in results.items():
            metrics['blocked'] = bool(metrics['contact'] and metrics['right_goal'] is False) if absent_goal and metrics['completed'] else None
        g, h = results['g_active']['blocked'], results['h_stop_then_arm']['blocked']
        report['shots'].append({'name': name, 'launch': shots[name]['launch'], 'modes': results,
            'added_over_G': g is False and h is True, 'lost_G_success': g is True and h is False})
        (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        h = results['h_stop_then_arm']
        if not report['all_repeats_identical'] or h['h_guard'] or h['fall_events'] or h['numerical_error'] or h['policy_stop']:
            report['stopped_after'] = name
            break
    report['protected_files_unchanged'] = all(digest(ROOT/p) == value for p, value in protected.items())
    report['code_unchanged'] = all(digest(ROOT/p) == value for p, value in protocol['code_sha256'].items())
    report['summary'] = {mode: sum(bool(shot['modes'][mode]['blocked']) for shot in report['shots'])
                         for mode in protocol['modes']}
    report['summary'].update(added=sum(shot['added_over_G'] for shot in report['shots']),
                             lost=sum(shot['lost_G_success'] for shot in report['shots']))
    (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'reports/defense_validation/goalkeeper')
    parser.add_argument('--h3-evidence', type=Path, default=ROOT/'reports/defense_validation/followup/results.json')
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.h3_evidence)['summary'], indent=2))
