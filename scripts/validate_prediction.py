"""Stage G: frozen protocol, no-ball gate, four-way paired physical validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch

from k1_goalkeeper.ball_prediction import build_ball_floor_model
from k1_goalkeeper.positioning_validation import run_trial, save_trial, prediction_errors
from k1_goalkeeper.predictive_goalkeeper import run_g_trial
from k1_goalkeeper.scene import build_scene
from k1_goalkeeper.velocity_validation import trajectory_hash
from scripts.validate_impacts import compare
from scripts.validate_walk import verify_sources

PROTOCOL = ROOT/'reports/prediction_validation/protocol.json'
F_DATA = ROOT/'reports/positioning_validation'
G_CODE = ['k1_goalkeeper/ball_prediction.py', 'k1_goalkeeper/predictive_goalkeeper.py',
          'scripts/validate_prediction.py']


def hashes(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def load_trial(path):
    summary = json.loads((path/'summary.json').read_text())
    with np.load(path/'trajectory.npz') as archive:
        trace = {k: archive[k] for k in archive.files}
    if trajectory_hash(trace) != summary['trajectory_sha256']:
        raise RuntimeError(f'trajectory integrity failure: {path}')
    return summary, trace


def paired_outcome_change(f_blocked, g_blocked):
    return {'added_over_F': f_blocked is False and g_blocked is True,
            'lost_F_success': f_blocked is True and g_blocked is False,
            'comparison_censored': f_blocked is None or g_blocked is None}


def prepare_case(output, name, mode, algorithm, launch, repeats, fingerprint):
    """Independent process/data per physical case; repetitions remain sequential."""
    path = output/name
    if (path/'summary.json').exists():
        old = json.loads((path/'summary.json').read_text())
        if old.get('g_execution_fingerprint') == fingerprint and len(old['reproducibility']['hashes']) == repeats:
            return
    model = mujoco.MjModel.from_xml_path(str(output/f"scene_{'absent' if mode == 'absent' else 'free'}.xml"))
    predictor = build_ball_floor_model(output/'scene_absent.xml') if algorithm == 'G' else None
    first, trace, repeated = None, None, []
    for _ in range(repeats):
        s, t = run_g_trial(model, predictor, name, launch) if algorithm == 'G' else run_trial(model, name, mode, launch)
        repeated.append(s['trajectory_sha256'])
        if first is None:
            first, trace = s, t
    first['reproducibility'] = {'verified': len(set(repeated)) == 1, 'hashes': repeated}
    first['g_execution_fingerprint'] = fingerprint
    save_trial(path, first, trace)
    print('prepared', name, 'repeat', first['reproducibility']['verified'], flush=True)


def run(output, repeats=2, phase='all'):
    if repeats < 2:
        raise ValueError('Stage G acceptance requires at least two repetitions')
    protocol = json.loads(PROTOCOL.read_text())
    sources = verify_sources()
    f_code = json.loads((F_DATA/'results.json').read_text())['code_sha256']
    if hashes([ROOT/p for p in f_code]) != f_code:
        raise RuntimeError('Stage F source no longer matches the archived baseline')
    protected_paths = [p for p in (ROOT/'reports').rglob('*')
                       if p.is_file() and 'prediction_validation' not in p.parts]
    protected_paths += [p for directory in ('k1_goalkeeper', 'scripts', 'tests', 'docs', 'models', 'config', 'assets')
                        for p in (ROOT/directory).rglob('*') if p.is_file()
                        and p.suffix in ('.py', '.xml', '.urdf', '.json', '.pt', '.md')
                        and not any(word in p.name for word in ('prediction', 'predictive', 'audit_goalkeeper_failures', 'audit_ball_prediction'))]
    protected = hashes(protected_paths)
    code = hashes([ROOT/p for p in G_CODE])
    output.mkdir(parents=True, exist_ok=True)
    models = {mode: mujoco.MjModel.from_xml_path(str(build_scene(output/f'scene_{mode}.xml', robot_mode=mode)))
              for mode in ('free', 'absent')}
    predictor = build_ball_floor_model(output/'scene_absent.xml')
    scene_hashes = hashes([output/'scene_free.xml', output/'scene_absent.xml'])
    fingerprint = hashlib.sha256(json.dumps([code, f_code, sources, scene_hashes, protocol,
        mujoco.__version__, np.__version__, torch.__version__], sort_keys=True).encode()).hexdigest()
    report = {'stage': 'G', 'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'code_sha256': code, 'sources': sources, 'protected_sha256': protected,
              'F_code_matches_archive': True,
              'compiled_scene_sha256': scene_hashes,
              'versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__, 'torch': torch.__version__},
              'repeats': repeats, 'results': [], 'shots': [], 'positioning': [], 'position_gate_passed': False,
              'baseline_note': 'Original ten absent/passive/F-active and seven F positions referenced byte-for-byte; new holdout modes and all G modes freshly executed.'}

    def persist():
        report['all_repeats_identical'] = bool(report['results']) and all(s['reproducibility']['verified'] for s in report['results'])
        (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def execute(name, mode, launch, algorithm='F', target=0., historical=False):
        path = F_DATA/name if historical else output/name
        cached = path/'summary.json'
        # Missing/corrupt historical evidence must fail read-only, never regenerate F_DATA.
        loaded = load_trial(path) if historical else None
        if not historical and cached.exists():
            candidate = json.loads(cached.read_text())
            if (candidate.get('g_execution_fingerprint') == fingerprint and
                    len(candidate.get('reproducibility', {}).get('hashes', [])) == repeats):
                loaded = load_trial(path)
        if loaded:
            first, trace = loaded
        else:
            model = models['absent' if mode == 'absent' else 'free']
            def trial():
                if algorithm == 'G':
                    return run_g_trial(model, predictor, name, launch, mode=mode, target=target)
                return run_trial(model, name, mode, launch, target=target)
            first, trace = trial()
            repeated = [first['trajectory_sha256']]
            for _ in range(repeats-1):
                second, _ = trial()
                repeated.append(second['trajectory_sha256'])
            first['reproducibility'] = {'verified': len(set(repeated)) == 1, 'hashes': repeated}
            first['g_execution_fingerprint'] = fingerprint
            save_trial(path, first, trace)
        first['g_reference'] = {'path': str(path.relative_to(ROOT)), 'historical': historical, 'algorithm': algorithm}
        report['results'].append({k: v for k, v in first.items() if k != 'high_level_records'})
        print(name, 'reuse' if loaded else 'executed', 'fall', bool(first['fall_events']),
              'contact', first['effective_contact'], 'right_goal', first['right_goal'],
              'repeat', first['reproducibility']['verified'], flush=True)
        return first, trace

    parked = {'position': [0., 0., .111], 'velocity': [0., 0., 0.], 'angular_velocity': [0., 0., 0.]}
    for target in protocol['position_targets']:
        original = target in (0., .2, -.2, .4, -.4, .6, -.6)
        f, _ = execute(f'position_{target:+.2f}' if original else f'position_f_{target:+.2f}',
                       'position', parked, target=target, historical=original)
        g, _ = execute(f'position_g_{target:+.2f}', 'position', parked, algorithm='G', target=target)
        report['positioning'].append({'target': target, 'F': f['positioning'], 'G': g['positioning'],
                                      'F_recovery': f['final_stop_recovery'], 'G_recovery': g['final_stop_recovery']})
        persist()
    report['position_gate_passed'] = all(p['G']['passed'] for p in report['positioning'])
    persist()
    if not report['position_gate_passed'] or phase == 'position':
        report['protected_unchanged'] = protected == hashes(protected_paths)
        persist()
        return report

    for split in ('regression', 'holdout'):
        for shot in protocol[split]:
            outcomes, traces = {}, {}
            # Two isolated workers share no controller globals or MuJoCo data.
            with ProcessPoolExecutor(max_workers=2) as pool:
                pending = []
                for label, mode, algorithm in [('absent', 'absent', 'F'), ('passive', 'passive', 'F'),
                                                ('f_active', 'active', 'F'), ('g_active', 'active', 'G')]:
                    if split == 'regression' and algorithm == 'F':
                        continue
                    pending.append(pool.submit(prepare_case, output, shot['name']+'_'+label, mode,
                                               algorithm, shot['launch'], repeats, fingerprint))
                for future in pending:
                    future.result()
            for label, mode, algorithm in [('absent', 'absent', 'F'), ('passive', 'passive', 'F'),
                                            ('f_active', 'active', 'F'), ('g_active', 'active', 'G')]:
                historical = split == 'regression' and algorithm == 'F'
                name = shot['name']+'_'+('active' if historical and label == 'f_active' else label)
                outcomes[label], traces[label] = execute(name, mode, shot['launch'], algorithm, historical=historical)
            absent = outcomes['absent']
            metrics = {}
            for label in ('passive', 'f_active', 'g_active'):
                s, trace = outcomes[label], traces[label]
                cutoff = min((c['start_time'] for c in s['contacts'] if c['kind'] == 'robot'), default=float(trace['time'][-1]))
                records = [r for r in s['high_level_records'] if 2. < r['time'] < cutoff]
                reaches = [r for r in records if r.get('reachability')]
                feasible = [r for r in records if r.get('intercept_feasible') is not None]
                blocked = bool(s['effective_contact'] and s['right_goal'] is False) if absent['right_goal'] and s['completed'] else None
                metrics[label] = {'contact': s['effective_contact'], 'right_goal': s['right_goal'], 'blocked': blocked,
                    'fall': bool(s['fall_events']), 'recovery': s['final_stop_recovery'],
                    'physical_comparison': compare(s, trace, absent, traces['absent']),
                    'prediction_errors': prediction_errors(records, absent['plane_crossings']),
                    'first_station': reaches[0]['reachability'] if reaches else None,
                    'station_positive_updates': sum(r['reachability']['reachable'] for r in reaches),
                    'interception_positive_updates': sum(r['intercept_feasible'] for r in feasible),
                    'first_interception_decision': {k: feasible[0].get(k) for k in ('time', 'station_reachable', 'current_body_coverage', 'target_body_coverage', 'intercept_feasible')} if feasible else None,
                    'trajectory_sha256': s['trajectory_sha256'],
                    'launch_identical': all(s['launch'][k] == absent['launch'][k] for k in ('position', 'velocity', 'angular_velocity'))}
            row = {'name': shot['name'], 'split': split, 'kind': shot['kind'], 'launch': shot['launch'],
                   'absent_right_goal': absent['right_goal'], 'modes': metrics,
                   **paired_outcome_change(metrics['f_active']['blocked'], metrics['g_active']['blocked'])}
            report['shots'].append(row)
            print('paired', shot['name'], {m: v['blocked'] for m, v in metrics.items()}, flush=True)
            persist()
    report['protected_unchanged'] = protected == hashes(protected_paths)
    report['all_repeats_identical'] = all(r['reproducibility']['verified'] for r in report['results'])
    report['new_cases'] = sum(not r['g_reference']['historical'] for r in report['results'])
    report['historical_cases'] = sum(r['g_reference']['historical'] for r in report['results'])
    report['fall_cases'] = [r['case']['name'] for r in report['results'] if r['fall_events']]
    report['numerical_error_cases'] = [r['case']['name'] for r in report['results'] if r['numerical_error']]
    report['unexpected_contact_cases'] = [r['case']['name'] for r in report['results'] if r['unexpected_contacts']]
    report['summary'] = {split: {**{mode: sum(bool(s['modes'][mode]['blocked']) for s in report['shots'] if s['split'] == split)
                                  for mode in ('passive', 'f_active', 'g_active')},
                                'added': sum(s['added_over_F'] for s in report['shots'] if s['split'] == split),
                                'lost': sum(s['lost_F_success'] for s in report['shots'] if s['split'] == split)}
                         for split in ('regression', 'holdout')}
    persist()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/prediction_validation')
    parser.add_argument('--phase', choices=('position', 'all'), default='all')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    result = run(args.output, args.repeats, args.phase)
    if (not result['position_gate_passed'] or not result['protected_unchanged'] or
            not result['all_repeats_identical'] or result.get('numerical_error_cases')):
        raise SystemExit(1)
