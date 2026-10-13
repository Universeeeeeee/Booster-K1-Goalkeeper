"""Independent saved-result audit of Stage G decisions and matched Stage F references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from k1_goalkeeper.velocity_validation import trajectory_hash


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossing(trace, cutoff):
    """Actual ball center crossing the actual moving robot center plane, pre-contact."""
    time = trace['time']
    delta = trace['ball_position'][:, 0]-trace['base_position'][:, 0]
    hits = np.flatnonzero((time[1:] > 2.) & (time[1:] < cutoff) & (delta[:-1] <= 0.) & (delta[1:] > 0.))
    if not len(hits):
        return None
    i = int(hits[0]); fraction = float(-delta[i]/(delta[i+1]-delta[i]))
    ball = trace['ball_position'][i]+fraction*(trace['ball_position'][i+1]-trace['ball_position'][i])
    robot = trace['base_position'][i]+fraction*(trace['base_position'][i+1]-trace['base_position'][i])
    return {'time_s': float(time[i]+fraction*(time[i+1]-time[i])), 'ball_position_world_m': ball.tolist(),
            'robot_position_world_m': robot.tolist(), 'robot_minus_ball_lateral_m': float(robot[1]-ball[1]),
            'aligned_with_actual_ball_within_0_1_m': bool(abs(robot[1]-ball[1]) <= .1)}


def bursts(trace):
    time, command = trace['time'], trace['command']
    cuts = np.r_[0, np.flatnonzero(np.any(command[1:] != command[:-1], axis=1))+1, len(time)]
    result = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        if not np.any(command[start]):
            continue
        stop = min(end, len(time)-1)
        result.append({'start_s': float(time[start]), 'stop_s': float(time[stop]),
                       'duration_s': float(time[stop]-time[start]), 'command': command[start].tolist(),
                       'lateral_displacement_m': float(trace['base_position'][stop, 1]-trace['base_position'][start, 1])})
    return result


def decision(row):
    if row is None:
        return None
    return {k: row.get(k) for k in ('time', 'world_position', 'reachability', 'station_reachable',
                                   'current_body_coverage', 'target_body_coverage', 'decision_body_coverage',
                                   'coverage_observed_time', 'coverage_decision_lateral_m',
                                   'intercept_feasible', 'interception_prediction', 'command')}


def station_check(first_positive, trace, contact_time, plane_crossing):
    if first_positive is None:
        return {'status': 'no positive center-station decision', 'declared_target_entry_before_plane': None,
                'actual_ball_alignment_at_plane': None}
    target = float(first_positive['reachability']['target_lateral_m'])
    decision_time = first_positive['time']
    deadline = plane_crossing['time_s'] if plane_crossing else float(trace['time'][-1])
    observed_end = min(contact_time, deadline)
    selected = np.flatnonzero((trace['time'] >= decision_time) & (trace['time'] < observed_end))
    errors = np.abs(trace['base_position'][selected, 1]-target)
    entry = selected[errors <= .1]
    censored = plane_crossing is None
    return {'status': 'censored: robot contact before center-plane arrival' if censored and np.isfinite(contact_time)
            else 'censored: center-plane arrival unobserved' if censored else 'observable before robot contact',
            'first_positive_time_s': decision_time, 'first_declared_station_target_m': target,
            'actual_precontact_robot_plane_crossing': plane_crossing,
            'precontact_observation_end_s': observed_end,
            'precontact_minimum_declared_target_error_m': float(errors.min()) if len(errors) else None,
            'precontact_declared_target_entry_time_s': float(trace['time'][entry[0]]) if len(entry) else None,
            # A pre-contact entry is observed support even if later arrival is censored.
            'declared_target_entry_before_plane': None if censored else bool(len(entry)),
            'actual_ball_alignment_at_plane': None if censored else plane_crossing['aligned_with_actual_ball_within_0_1_m'],
            'actual_ball_lateral_error_at_plane_m': None if censored else plane_crossing['robot_minus_ball_lateral_m'],
            'interpretation': 'First declared center target entry and actual ball alignment are separate; transient entry is not a hold or a body-contact promise.'}


def analyze(source, output):
    results_path, protocol_path = source/'results.json', source/'protocol.json'
    results = json.loads(results_path.read_text()); protocol = json.loads(protocol_path.read_text())
    files = {results_path, protocol_path, Path(__file__).resolve()}
    evidence_checks = []
    cache = {}

    def load(row):
        path = ROOT/row['g_reference']['path']
        if str(path) not in cache:
            summary_path, trace_path = path/'summary.json', path/'trajectory.npz'
            summary = json.loads(summary_path.read_text())
            with np.load(trace_path) as archive:
                trace = {k: archive[k] for k in archive.files}
            checks = {'case': row['case']['name'], 'path': str(path.relative_to(ROOT)),
                      'trajectory_hash_matches_summary': trajectory_hash(trace) == summary['trajectory_sha256'],
                      'trajectory_hash_matches_runner_reference': summary['trajectory_sha256'] == row['trajectory_sha256'],
                      'decision_bytes_match_summary': json.loads(trace['high_level_json_utf8'].tobytes()) == summary['high_level_records']
                          if 'high_level_json_utf8' in trace else not summary['high_level_records']}
            assert all(v for k, v in checks.items() if k.endswith(('summary', 'reference')))
            evidence_checks.append(checks); files.update((summary_path, trace_path))
            cache[str(path)] = summary, trace
        return cache[str(path)]

    rows = {r['case']['name']: r for r in results['results']}
    positions = {}
    for row in results['results']:
        if row['mode'] != 'position':
            continue
        summary, trace = load(row)
        target = float(summary['positioning']['requested_goal_lateral_m'])
        algorithm = row['g_reference']['algorithm']
        positions.setdefault(str(target), {})[algorithm] = {
            'case': row['case']['name'], 'historical_reference': row['g_reference']['historical'],
            **summary['positioning'], 'delivered_control_bursts': bursts(trace),
            'lateral_command_duration_s': sum(b['duration_s'] for b in bursts(trace) if b['command'][1]),
            'final_stop_recovery_verified': summary['final_stop_recovery']['verified'],
            'falls': len(summary['fall_events']), 'numerical_error': summary['numerical_error']}
    for pair in positions.values():
        if 'F' in pair and 'G' in pair:
            f, g = pair['F'], pair['G']
            pair['G_minus_F'] = {'arrival_and_hold_time_s': g['arrival_and_hold_time_s']-f['arrival_and_hold_time_s']
                                if g['arrival_and_hold_time_s'] is not None and f['arrival_and_hold_time_s'] is not None else None,
                                'final_absolute_error_m': abs(g['final_error_m'])-abs(f['final_error_m']),
                                'final_stop_command_time_s': g['stop_command_time']-f['stop_command_time'],
                                'control_burst_count': len(g['delivered_control_bursts'])-len(f['delivered_control_bursts'])}

    shots = []
    for paired in results['shots']:
        name, split = paired['name'], paired['split']
        modes = {}
        for label in ('absent', 'passive', 'f_active', 'g_active'):
            case_name = name+'_'+('active' if split == 'regression' and label == 'f_active' else label)
            reference = rows[case_name]
            summary, trace = load(reference)
            contacts = [c['start_time'] for c in summary['contacts'] if c['kind'] == 'robot']
            contact_time = min(contacts, default=float('inf'))
            observed = [r for r in summary['high_level_records'] if 2. < r['time'] < contact_time]
            stations = [r for r in observed if r.get('reachability')]
            station_positive = [r for r in stations if r['reachability']['reachable']]
            intercepts = [r for r in observed if r.get('intercept_feasible') is not None]
            intercept_positive = [r for r in intercepts if r['intercept_feasible']]
            first_station = station_positive[0] if station_positive else None
            first_intercept = intercept_positive[0] if intercept_positive else None
            completed = summary['completed']
            absent_goal = paired['absent_right_goal']
            blocked = bool(summary['effective_contact'] and summary['right_goal'] is False) if completed and absent_goal else None
            plane = crossing(trace, contact_time) if label != 'absent' else None
            modes[label] = {'source': reference['g_reference'], 'completed': completed,
                            'right_goal': summary['right_goal'], 'effective_contact': summary['effective_contact'],
                            'blocked': blocked, 'falls': len(summary['fall_events']),
                            'numerical_error': summary['numerical_error'], 'balance_maintained': summary['balance_maintained'],
                            'recovery_verified': summary['final_stop_recovery']['verified'] if label != 'absent' else None,
                            'first_robot_contact_time_s': contact_time if contacts else None,
                            'first_station_decision': decision(stations[0] if stations else None),
                            'first_positive_station': decision(first_station),
                            'station_positive_update_count': len(station_positive),
                            'station_observation_audit': station_check(first_station, trace, contact_time, plane),
                            'first_interception_decision': decision(intercepts[0] if intercepts else None),
                            'first_positive_interception': decision(first_intercept),
                            'interception_positive_update_count': len(intercept_positive),
                            'operational_interception_contact_false_positive': bool(first_intercept and not summary['effective_contact']) if label == 'g_active' and completed else None,
                            'operational_interception_block_false_positive': bool(first_intercept and not blocked) if label == 'g_active' and completed and absent_goal else None,
                            'station_positive_without_contact': bool(first_station and not summary['effective_contact']) if completed else None,
                            'station_positive_without_block': bool(first_station and not blocked) if completed and absent_goal else None,
                            'actual_precontact_robot_plane_crossing': plane,
                            'delivered_control_bursts': bursts(trace),
                            'label_semantics': 'F records center-station feasibility only; no historical F interception claim.' if label == 'f_active' else
                                               'G intercept_feasible is center reachability AND native decision-pose coverage AND aperture; frozen pose hypothesis.' if label == 'g_active' else 'Physical control baseline; no active feasibility claim.'}
        shots.append({'name': name, 'split': split, 'kind': paired['kind'], 'modes': modes,
                      'added_over_F': paired['added_over_F'], 'lost_F_success': paired['lost_F_success']})
    aggregate = {}
    for split in ('regression', 'holdout', 'all'):
        selected = [s for s in shots if split == 'all' or s['split'] == split]
        aggregate[split] = {'shot_count': len(selected), 'modes': {}}
        for mode in ('absent', 'passive', 'f_active', 'g_active'):
            values = [s['modes'][mode] for s in selected]
            auditable = [v['station_observation_audit'] for v in values
                         if v['first_positive_station'] and v['station_observation_audit']['status'] == 'observable before robot contact']
            aggregate[split]['modes'][mode] = {
                'right_goals': sum(v['right_goal'] is True for v in values),
                'effective_contacts': sum(v['effective_contact'] for v in values),
                'blocks': sum(v['blocked'] is True for v in values),
                'station_positive_cases': sum(v['first_positive_station'] is not None for v in values),
                'station_positive_censored_cases': sum(v['first_positive_station'] is not None and
                                                     v['station_observation_audit']['status'].startswith('censored') for v in values),
                'station_observable_cases': len(auditable),
                'declared_target_entry_misses_before_actual_plane': sum(v['declared_target_entry_before_plane'] is False for v in auditable),
                'actual_ball_alignment_misses_at_plane': sum(v['actual_ball_alignment_at_plane'] is False for v in auditable),
                'interception_positive_cases': sum(v['first_positive_interception'] is not None for v in values),
                'operational_interception_contact_false_positives': sum(v['operational_interception_contact_false_positive'] is True for v in values),
                'operational_interception_block_false_positives': sum(v['operational_interception_block_false_positive'] is True for v in values),
                'station_positive_without_contact': sum(v['station_positive_without_contact'] is True for v in values),
                'station_positive_without_block': sum(v['station_positive_without_block'] is True for v in values)}
        aggregate[split]['added_over_F'] = sum(s['added_over_F'] for s in selected)
        aggregate[split]['lost_F_success'] = sum(s['lost_F_success'] for s in selected)
    expected = len(protocol['regression'])+len(protocol['holdout'])
    complete = len(shots) == expected and len(positions) == len(protocol['position_targets'])
    report = {'stage': 'G', 'complete_protocol_observed': complete,
              'expected_shots': expected, 'expected_splits': {k: len(protocol[k]) for k in ('regression', 'holdout')},
              'definitions': {
                  'station': 'First positive reachability.reachable; center target only, within 0.10 m.',
                  'arrival': 'First actual saved ball center crossing moving actual robot center X plane, using linear interpolation between 1 ms samples.',
                  'censor': 'If physical robot contact occurs before center-plane crossing, no post-contact robot pose is combined with an absent-ball future. Station endpoint and full-before-deadline entry scores are null; pre-contact entry evidence remains descriptive.',
                  'body_coverage': 'Native current/translated frozen pose overlap hypothesis, not future gait shape or an implemented reach/jump/dive.',
                  'interception': 'G intercept_feasible is distinct from station_reachable and current/target/decision_body_coverage.',
                  'operational_false_positive': 'At least one pre-contact intercept_feasible=True, followed by completed contact miss or completed block miss against an absent right goal, counted separately.',
                  'F_comparison': 'F station-positive/contact or block misses are reported separately and never relabeled as a historical interception promise.',
                  'position': 'F/G permanent tolerance hold is the saved F metric; stop times are last moving raw sample, bursts start one post-step sample after decisions.'},
              'positioning': positions, 'shots': shots, 'aggregate': aggregate,
              'verification': {'case_checks': evidence_checks, 'all_loaded_hashes_and_decisions_match': True,
                               'runner_protected_unchanged': results.get('protected_unchanged'),
                               'runner_all_repeats_identical': results.get('all_repeats_identical')},
              'minimal_recommendations': [
                  'Keep center-station feasibility, native body coverage and physical interception/block outcomes as separate reported quantities.',
                  'Use uncensored real robot-plane deadlines to audit center arrival; mark contacted trajectories censored instead of scoring invented post-contact counterfactual poses.',
                  'Treat operational false positives as empirical calibration observations for this declared protocol, not universal success-rate estimates.'],
              'input_manifest': {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}}
    # Detect a runner update while reading, rather than emitting a mixed snapshot.
    if json.loads(results_path.read_text()) != results:
        raise RuntimeError('Runner results changed during analysis; rerun after the current case batch.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'complete_protocol_observed': complete,
                      'shot_counts': {k: v['shot_count'] for k, v in aggregate.items()},
                      'all_modes': aggregate['all']['modes']}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'reports/prediction_validation')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/prediction_validation/decision_audit.json')
    args = parser.parse_args()
    analyze(args.input.resolve(), args.output.resolve())
