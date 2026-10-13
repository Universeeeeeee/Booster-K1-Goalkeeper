"""Audit frozen Stage F failures using saved observations and native collision geometry.

This performs kinematics on saved qpos, never a simulation step or policy call.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from scripts.audit_collisions import inventory
from k1_goalkeeper.velocity_validation import trajectory_hash

SHOTS = ('wide_left', 'wide_right', 'air_wide', 'air_high', 'bouncing')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(paths):
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()}


def load_case(source, name):
    folder = source/name
    summary = json.loads((folder/'summary.json').read_text())
    trace = dict(np.load(folder/'trajectory.npz'))
    return summary, trace


def segments(trace):
    commands, time = trace['command'], trace['time']
    cuts = np.r_[0, np.flatnonzero(np.any(commands[1:] != commands[:-1], axis=1))+1, len(time)]
    rows = []
    for first, end in zip(cuts[:-1], cuts[1:]):
        last = min(end, len(time)-1)
        rows.append({'start_s': float(time[first]), 'end_s': float(time[last]),
                     'duration_s': float(time[last]-time[first]), 'command': commands[first].tolist(),
                     'start_lateral_m': float(trace['base_position'][first, 1]),
                     'end_lateral_m': float(trace['base_position'][last, 1])})
    return rows


def position_audit(source, name):
    s, t = load_case(source, name)
    target = s['positioning']['requested_goal_lateral_m']
    rows = [r for r in s['high_level_records'] if r.get('tracking')]
    switches = []
    old = None
    for row in s['high_level_records']:
        if row['command'] != old:
            switches.append({k: row.get(k) for k in ('time', 'command', 'world_position', 'yaw', 'world_velocity', 'tracking')})
            old = row['command']
    stable = (t['time'] >= 2.) & (np.abs(t['base_position'][:, 1]-target) <= .1)
    changes = np.flatnonzero(stable[1:] != stable[:-1])+1
    last_outside = np.flatnonzero(~stable)[-1]
    arrival = float(t['time'][last_outside+1]-2.)
    bursts = [r for r in segments(t) if r['command'][1]]
    first_end = bursts[0]['end_s']
    next_start = bursts[1]['start_s'] if len(bursts) > 1 else float(t['time'][-1])
    after_first = (t['time'] >= first_end) & (t['time'] < next_start)
    first_stop = next(r for r in switches if r['time'] > 2. and not any(r['command']))
    stop_index = int(np.argmin(abs(t['time']-(first_stop['time']+.4))))
    result = {'case': name, 'positioning': s['positioning'],
              'raw_arrival_and_permanent_hold_s': arrival,
              'tolerance_transitions': [{'time_s': float(t['time'][i]), 'within_tolerance': bool(stable[i]),
                                         'lateral_m': float(t['base_position'][i, 1])} for i in changes],
              'delivered_command_segments': segments(t), 'high_level_command_switches': switches,
              'yaw_command_nonzero_samples': int(np.count_nonzero(t['command'][:, 2])),
              'maximum_tracking_yaw_error_rad': max(abs(r['tracking']['yaw_error_rad']) for r in rows),
              'first_stop_to_restart_lateral_range_m': [float(t['base_position'][after_first, 1].min()),
                                                       float(t['base_position'][after_first, 1].max())],
              'lateral_command_duration_s': sum(r['duration_s'] for r in bursts),
              'first_stop_prediction_check': {
                  'decision_time_s': first_stop['time'],
                  'predicted_stop_lateral_m': first_stop['tracking']['predicted_stop_m'],
                  'actual_lateral_after_0_4_s_m': float(t['base_position'][stop_index, 1]),
                  'actual_minus_predicted_m': float(t['base_position'][stop_index, 1]-first_stop['tracking']['predicted_stop_m'])},
              'safe_tracking_at_all_updates': all(r['tracking']['safe'] for r in rows),
              'falls': len(s['fall_events']), 'numerical_error': s['numerical_error'],
              'trajectory_hash_matches': trajectory_hash(t) == s['trajectory_sha256'],
              'high_level_records_match_raw_utf8': json.loads(t['high_level_json_utf8'].tobytes()) == s['high_level_records']}
    assert abs(arrival-s['positioning']['arrival_and_hold_time_s']) < 1e-8
    return result


def native_top(model, data, geom):
    """Exact axis support of collision primitive or compiled mesh convex hull."""
    axis = data.geom_xmat[geom].reshape(3, 3)[2]
    size, kind = model.geom_size[geom], model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        support = np.abs(axis)@size
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        support = size[0]*np.linalg.norm(axis[:2])+size[1]*abs(axis[2])
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        support = (model.mesh_vert[start:start+count]@axis).max()
    else:
        raise ValueError(f'Unexpected native collision type: {kind}')
    return float(data.geom_xpos[geom, 2]+support)


def geometry_audit(model, data, geoms, trace):
    """Measure native distances through the complete conservative body passage."""
    ball = model.geom('ball_collision').id
    radius = float(model.geom_size[ball, 0])
    best, best_row = float('inf'), None
    tops, bottoms, clearances, times = [], [], [], []
    indices = np.flatnonzero((trace['time'] > 2.) &
                             (trace['ball_position'][:, 0] <= trace['base_position'][:, 0]+.81))
    for i in indices:
        data.qpos[:] = trace['qpos'][i]
        mujoco.mj_kinematics(model, data)
        position = data.geom_xpos[ball]
        for geom in geoms:
            # Native enclosing radii give a safe lower bound for pruning only.
            lower = np.linalg.norm(position-data.geom_xpos[geom])-radius-model.geom_rbound[geom]
            if lower >= best:
                continue
            distance = float(mujoco.mj_geomDistance(model, data, ball, geom, 100., None))
            if distance < best:
                best = distance
                best_row = {'time_s': float(trace['time'][i]), 'geom': model.geom(geom).name,
                            'body': model.body(int(model.geom_bodyid[geom])).name,
                            'ball_center_world_m': trace['ball_position'][i].tolist(),
                            'robot_center_world_m': trace['base_position'][i].tolist()}
        # .81 is F's all-articulation robot extent plus ball radius rounded outward.
        if abs(position[0]-trace['base_position'][i, 0]) <= .81:
            top = max(native_top(model, data, geom) for geom in geoms)
            bottom = float(position[2]-radius)
            tops.append(top); bottoms.append(bottom); clearances.append(bottom-top); times.append(float(trace['time'][i]))
    return {'native_collision_geom_count': len(geoms), 'physics_samples_checked': len(indices),
            'minimum_native_ball_robot_surface_distance_m': best, 'nearest_state': best_row,
            'robot_passage_root_offset_bound_m': .81,
            'passage_sample_count': len(times), 'passage_time_range_s': [times[0], times[-1]],
            'maximum_native_robot_top_m': max(tops), 'minimum_ball_bottom_in_passage_m': min(bottoms),
            'minimum_ball_bottom_minus_native_robot_top_m': min(clearances),
            'ball_above_every_native_collision_geom_through_passage': min(clearances) > 0.}


def shot_audit(source, name, model, data, geoms):
    s, t = load_case(source, name+'_active')
    absent, absent_trace = load_case(source, name+'_absent')
    passive, _ = load_case(source, name+'_passive')
    rows = [r for r in s['high_level_records'] if 2.+1e-9 < r['time'] < absent['plane_crossings']['11.11']['time']]
    usable = [r for r in rows if r.get('reachability')]
    errors = {}
    for plane, value in s['prediction_errors_against_matched_absent'].items():
        samples = value['samples']
        modeled = [r for r in samples if not r['unmodeled_bounce']]
        errors[plane] = {k: value[k] for k in ('reference', 'first', 'maximum_absolute_errors')}
        errors[plane]['usable_model_maximum_absolute_errors'] = {key: max(abs(r[key]) for r in modeled) for key in
            ('time_error_s', 'lateral_error_m', 'height_error_m')} if modeled else None
        errors[plane]['unmodeled_bounce_samples'] = sum(r['unmodeled_bounce'] for r in samples)
    initial = rows[0]
    first_usable = usable[0] if usable else None
    comparison_keys = ('right_goal', 'effective_contact', 'completed', 'balance_maintained')
    roots = {
        'prediction_error': 'Large raw bounce errors are flagged and rejected; usable rolling estimates are evaluated separately.' if name == 'bouncing' else 'Small forecast errors; no evidence that prediction error caused this miss.',
        'insufficient_time': 'Already laterally aligned; insufficient lateral time does not explain the vertical miss.' if name == 'air_high' else 'Current conservative center travel interval never reaches the shot target.',
        'control_bias': 'Clipped targets stay within the 0.10 m startup deadband; zero movement. No left/right directional difference is demonstrated.' if name != 'air_high' else 'Zero command is appropriate for lateral center alignment; lateral tracking is not a vertical interception action.',
        'modeled_collision_coverage': 'Ball passes above every native collision geom during the entire robot passage.' if name == 'air_high' else 'Native geometry shows positive ball/body separation; no evidence specifically attributes the miss to missing local collision patches.',
        'missing_actions': 'F offers lateral walking only. Fast lateral movement, reach, jump and dive are not available or evaluated.'}
    return {'case': name, 'root_cause_assessment': roots,
            'matched_outcomes': {mode: {k: value[k] for k in comparison_keys}
                                 for mode, value in [('absent', absent), ('passive', passive), ('active', s)]},
            'identical_launch': s['launch_identical_to_absent'],
            'active_ball_trajectory_maximum_difference_from_absent_m': float(np.linalg.norm(t['ball_position']-absent_trace['ball_position'], axis=1).max()),
            'prediction_errors': errors,
            'first_observed_shot': {k: initial.get(k) for k in ('time', 'prediction', 'interception_prediction', 'reachability', 'command')},
            'first_usable_control': {k: first_usable.get(k) for k in ('time', 'interception_prediction', 'reachability', 'tracking', 'command')} if first_usable else None,
            'observation_to_first_usable_delay_s': first_usable['time']-initial['time'] if first_usable else None,
            'reachability_summary': s['reachability_summary'],
            'maximum_center_travel_budget_m': max(r['reachability']['travel_budget_m'] for r in usable) if usable else None,
            'maximum_decision_error_m': max(abs(r['tracking']['error_m']) for r in usable) if usable else None,
            'nonzero_delivered_command_samples': int(np.count_nonzero(np.any(t['command'] != 0., axis=1))),
            'native_geometry': geometry_audit(model, data, geoms, t),
            'coverage_limitations_from_F': s['coverage_limitations'],
            'trajectory_hash_matches': trajectory_hash(t) == s['trajectory_sha256'],
            'high_level_records_match_raw_utf8': json.loads(t['high_level_json_utf8'].tobytes()) == s['high_level_records']}


def markdown(report):
    p = report['positioning']
    high = report['shots']['air_high']['native_geometry']
    text = ['# Stage G: frozen Stage F failure audit', '',
            'Offline audit of saved 1 ms trajectories, delivered commands, 20 ms decisions and native MuJoCo collision geometry. No physics step, controller call, visual mesh collision assumption, or A–F evidence rewrite.', '',
            '## Why position −0.40 takes 8.624 s', '',
            'Both signs issue one 2.44 s lateral burst from t=2.00 to 4.44. The predictive stop uses filtered velocity ×0.4 s and stops with about 0.032–0.035 m predicted residual. The actual first burst leaves the robot near the ±0.30 m tolerance boundary, rather than near the predicted ±0.367 m stopping point.', '',
            'These times identify high-level decision boundaries. The raw post-step recorder first shows the updated commands 1 ms later (2.001 and 4.441); both layers are preserved in the JSON.', '',
            'Negative motion first enters tolerance at t=4.726, exits at 4.912, reenters at 4.974, then slow standing drift crosses out at 10.591. At the next 20 ms update (10.60), error exceeds 0.10 m and the controller restarts; permanent hold begins at 10.624, giving 8.624 s after enable. The positive case enters tolerance at 4.733 and stays inside to the trace end, giving 2.733 s. F’s arrival metric requires remaining inside until the end, in addition to at least 1 s observation.', '',
            'There are zero turn-command samples in both cases. Maximum tracked yaw error is 0.073716 rad negative and 0.037214 rad positive, below the 0.12 rad turn-entry threshold. Both remain safe and upright. This trace establishes asymmetric post-stop drift interacting with stop prediction and the 0.10 m deadband; it does not identify the policy’s internal cause of that drift.', '',
            '| Case | Permanent hold after enable (s) | Final lateral (m) | Lateral command duration (s) | Turn samples |',
            '|---|---:|---:|---:|---:|']
    for row in p.values():
        text.append(f"| {row['case']} | {row['raw_arrival_and_permanent_hold_s']:.3f} | {row['positioning']['final_goal_position'][0]:+.6f} | {row['lateral_command_duration_s']:.3f} | {row['yaw_command_nonzero_samples']} |")
    text += ['', '## Failed shots', '',
             'Every listed active shot has zero nonzero delivered command samples, no effective contact, and a right-goal outcome identical to the matched absent trajectory. Prediction correctness and center reachability are separate from physical collision coverage.', '',
             '| Case | First usable time (s) | Contact-bound time left (s) | Max center budget (m) | Closest native surface distance (m) | Attribution |',
             '|---|---:|---:|---:|---:|---|']
    labels = {'wide_left': 'insufficient modeled time + safe-center clamp + clipped-target deadband',
              'wide_right': 'insufficient modeled time + safe-center clamp + clipped-target deadband',
              'air_wide': 'short flight; zero modeled travel budget; missing fast lateral action',
              'air_high': 'center reachable; ball above native body; missing high interception action',
              'bouncing': 'initial unmodeled bounce abstention; later budget below movement deadband'}
    for name, row in report['shots'].items():
        usable = row['first_usable_control']
        text.append(f"| {name} | {usable['time']:.2f} | {usable['interception_prediction']['time_to_plane_s']:.4f} | {row['maximum_center_travel_budget_m']:.4f} | {row['native_geometry']['minimum_native_ball_robot_surface_distance_m']:.4f} | {labels[name]} |")
    text += ['',
             'Wide ±0.9 m: lateral forecast error is zero; whole-ball timing error ≤0.020842 s. Both targets exceed the ±0.65 m safe-center clamp; initial conservative travel is only 0.028084 m. The clipped decision target is inside the 0.10 m controller startup deadband, so it never starts. These are failures of the available action/time budget, compounded by how the conservative planner feeds the deadband; the trace does not prove physical impossibility for another action.', '',
             'Air-wide: ballistic whole-ball timing error is at numerical roundoff and height error ≤0.004263 m. The first robot-bound deadline has only 0.680314 s left, below the 1 s startup +1 s braking reserve; travel budget is zero. The lateral gap is about 0.599 m.', '',
             f"Air-high: the lateral center is already reachable. Height prediction error is only about 4.3 mm. Native surface distances and top supports are reconstructed from saved qpos for all 20 ball-eligible collision geoms, including compiled collision meshes. Disabled visual meshes are excluded. Through the complete ±0.81 m robot passage bound (t={high['passage_time_range_s'][0]:.3f}–{high['passage_time_range_s'][1]:.3f}), native body top is at most {high['maximum_native_robot_top_m']:.6f} m and ball bottom is at least {high['minimum_ball_bottom_in_passage_m']:.6f} m. Minimum simultaneous vertical clearance is {high['minimum_ball_bottom_minus_native_robot_top_m']:.6f} m; closest native surface separation is {high['minimum_native_ball_robot_surface_distance_m']:.6f} m. This establishes that this shot is above the modeled body, rather than passing through a missing local body patch.", '',
             'Bouncing: initial goal extrapolation is explicitly flagged unmodeled (first whole-ball time error −1.8282 s; height error −44.5408 m). The controller abstains until t=2.70, 0.68 s after the first observed launch state. Once near-ground estimates become usable, the maximum budget is only 0.033688 m, below the 0.10 m startup deadband, so commands remain zero. The large negative raw heights belong to rejected predictions, not accepted forecasts.', '',
             'Missing native collision regions around shoulder_pitch, hip_pitch and ankle_pitch remain a modeled coverage limitation. These traces show no contact and substantial lateral or vertical separation; they do not establish those missing patches as the cause of these particular failures. A jump, reach or dive was absent from F’s lateral-only action set and is not evaluated by this audit.', '',
             '## Reproduce and integrity', '',
             '```bash', '.venv/bin/python scripts/audit_goalkeeper_failures.py', '```', '',
             f"Historical reports unchanged during audit: {report['verification']['historical_reports_unchanged']}. Saved trajectory hashes and decision-byte parity all pass: {report['verification']['all_saved_evidence_checks_pass']}.", '',
             'JSON contains every command segment and switch, tolerance transition, first prediction and usable decision, prediction error bounds, closest collision state and exact native vertical support bounds. The geometry checks are discrete at the saved 1 ms resolution; they are evidence about these saved experiments, not a proof of success for other actions or unseen shots.', '']
    return '\n'.join(text)


def run(source, output):
    protected = [p for p in (ROOT/'reports').rglob('*') if p.is_file() and 'prediction_validation' not in p.parts]
    before = manifest(protected)
    inputs = [source/'scene_free.xml', source/'results.json', ROOT/'reports/collision_audit.json',
              ROOT/'k1_goalkeeper/positioning.py', ROOT/'k1_goalkeeper/positioning_validation.py',
              ROOT/'scripts/audit_collisions.py', ROOT/'scripts/audit_goalkeeper_failures.py']
    inputs += [source/n/file for n in ('position_-0.40', 'position_+0.40') for file in ('summary.json', 'trajectory.npz')]
    inputs += [source/(n+'_'+mode)/file for n in SHOTS for mode in ('active', 'passive', 'absent') for file in ('summary.json', 'trajectory.npz')]
    model = mujoco.MjModel.from_xml_path(str(source/'scene_free.xml'))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    records, _, _ = inventory(model, data)
    collision = [r for r in records if r['is_k1'] and r['ball_collision_eligible']]
    report = {'stage': 'G', 'audit': 'Frozen F failure attribution', 'source_directory': str(source.relative_to(ROOT)),
              'method': 'Saved qpos native kinematics and signed geom distances; no stepping or policy execution',
              'native_collision_inventory': [{k: r[k] for k in ('name', 'body', 'type', 'disabled_visual', 'ball_collision_eligible')} for r in collision],
              'positioning': {name: position_audit(source, name) for name in ('position_-0.40', 'position_+0.40')},
              'shots': {name: shot_audit(source, name, model, data, [r['geom_id'] for r in collision]) for name in SHOTS},
              'input_manifest': manifest(inputs), 'environment': {'mujoco': mujoco.__version__, 'numpy': np.__version__}}
    checks = [row['trajectory_hash_matches'] and row['high_level_records_match_raw_utf8'] for row in
              [*report['positioning'].values(), *report['shots'].values()]]
    report['verification'] = {'historical_reports_unchanged': manifest(protected) == before,
                              'historical_report_files_checked': len(before), 'all_saved_evidence_checks_pass': all(checks)}
    assert all(report['verification'][k] for k in ('historical_reports_unchanged', 'all_saved_evidence_checks_pass'))
    assert report['shots']['air_high']['native_geometry']['ball_above_every_native_collision_geom_through_passage']
    output.mkdir(parents=True, exist_ok=True)
    (output/'failure_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'failure_audit.md').write_text(markdown(report))
    print(json.dumps({'output': str(output), 'verification': report['verification'],
                      'native_minimum_distances_m': {k: v['native_geometry']['minimum_native_ball_robot_surface_distance_m'] for k, v in report['shots'].items()}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'reports/positioning_validation')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/prediction_validation')
    args = parser.parse_args()
    run(args.input.resolve(), args.output.resolve())
