"""H0: wall-clock audit of unchanged G prediction and official K1 control.

Replay saved current states on private MuJoCo data; never regenerate A--G trials.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch

from k1_goalkeeper.ball_prediction import build_ball_floor_model
from k1_goalkeeper.predictive_goalkeeper import PredictiveGoalkeeper
from k1_goalkeeper.velocity_validation import trajectory_hash
from k1_goalkeeper.walk_controller import K1WalkController
from scripts.validate_walk import verify_sources

CASES = [('long_rolling', 'reachable_left_g_active'),
         ('bouncing', 'bouncing_g_active'), ('airborne', 'air_center_g_active')]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    values = np.asarray(values)
    return {'samples': len(values), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'max_ms': float(values.max()),
            'samples_ms': values.tolist(),
            'exceeds_100ms_refresh_count': int(np.count_nonzero(values > 100)),
            'exceeds_20ms_policy_count': int(np.count_nonzero(values > 20))}


def measure(prepare, invoke, warmup, repeats, expected_delta=0.):
    durations, deltas = [], []
    for index in range(warmup+repeats):
        data = prepare()
        before = float(data.time)
        started = time.perf_counter_ns()
        invoke()
        elapsed = (time.perf_counter_ns()-started)/1e6
        delta = float(data.time)-before
        if not math.isclose(delta, expected_delta, abs_tol=1e-9):
            raise RuntimeError(f'Unexpected caller simulation time advance: {delta}')
        if index >= warmup:
            durations.append(elapsed)
            deltas.append(delta)
    return {**statistics(durations), 'caller_sim_time_delta_s': sorted(set(deltas))}


def run(output, warmup=3, repeats=20):
    if warmup < 1 or repeats < 2:
        raise ValueError('Use at least one warmup and two measured repetitions')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    source = ROOT/'reports/prediction_validation'
    # Hash all existing report evidence except new H outputs, plus relevant code/tests.
    protected_paths = [p for p in (ROOT/'reports').rglob('*') if p.is_file()
                       and 'defense_validation' not in p.parts]
    code_paths = [ROOT/p for p in ('k1_goalkeeper/ball_prediction.py',
        'k1_goalkeeper/predictive_goalkeeper.py', 'k1_goalkeeper/positioning_validation.py',
        'k1_goalkeeper/walk_controller.py', 'k1_goalkeeper/impact_validation.py')]
    protected_paths += code_paths + list((ROOT/'tests').glob('*.py'))
    protected = {str(p.relative_to(ROOT)): digest(p) for p in sorted(protected_paths)}
    official = verify_sources()
    model = mujoco.MjModel.from_xml_path(str(source/'scene_free.xml'))
    predictor_model = build_ball_floor_model(source/'scene_absent.xml')
    report = {'stage': 'H0', 'audit': 'wall-clock latency on archived G current states',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'environment': {'platform': platform.platform(), 'machine': platform.machine(),
            'python': sys.version, 'mujoco': mujoco.__version__, 'numpy': np.__version__,
            'torch': torch.__version__, 'torch_threads': torch.get_num_threads()},
        'method': {'clock': 'time.perf_counter_ns', 'warmup_per_metric': warmup,
            'measured_repetitions_per_metric': repeats, 'percentile': 'numpy linear p95',
            'sampling': 'first archived state at/after 2.02 s; independent of shot outcome',
            'timed_scope': 'calls only; scene/model/controller construction, state restore and file I/O excluded',
            'official_history': 'official policy history warms on repeated same current state; archived policy history is unavailable',
            'new_actions': 'only unchanged official gait on private clones; no new arm action or physical trial'},
        'budgets': {'predictor_refresh_period_ms': 100., 'official_policy_period_ms': 20.,
            'physics_step_ms': 1., 'predictor_horizon_s': 18.},
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in code_paths},
        'benchmark_script_sha256': digest(Path(__file__)),
        'compiled_scene_sha256': {str(p.relative_to(ROOT)): digest(p)
            for p in (source/'scene_free.xml', source/'scene_absent.xml')},
        'official_manifest': official, 'protected_sha256': protected, 'cases': []}

    for kind, name in CASES:
        path = source/name
        summary = json.loads((path/'summary.json').read_text())
        with np.load(path/'trajectory.npz') as archive:
            trace = {key: archive[key] for key in archive.files}
        if trajectory_hash(trace) != summary['trajectory_sha256']:
            raise RuntimeError(f'Archived trajectory integrity failure: {name}')
        index = int(np.flatnonzero(trace['time'] >= 2.02-1e-9)[0])
        saved_time = float(trace['time'][index])
        data = mujoco.MjData(model)
        base = model.joint('world_joint')
        bq, bv = int(base.qposadr[0]), int(base.dofadr[0])
        ball = model.joint('ball_free')
        q, v = int(ball.qposadr[0]), int(ball.dofadr[0])
        controller = K1WalkController(model, data)
        controller.initialize()

        def restore():
            mujoco.mj_resetData(model, data)
            data.qpos[:] = trace['qpos'][index]
            data.qvel[:] = trace['qvel'][index]
            data.ctrl[controller.actuator_ids] = trace['motor_controls'][index]
            data.time = saved_time
            mujoco.mj_forward(model, data)
            controller._expected_time = saved_time
            controller._warning_counts = controller._read_warning_counts()
            return data

        restore()
        saved_qpos, saved_qvel, saved_ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        sim = SimpleNamespace(model=model, data=data, physics_steps=round(saved_time/.001),
            controller=SimpleNamespace(data=data, base_qpos_address=bq, base_dof_address=bv))
        keeper = PredictiveGoalkeeper(model, predictor_model)
        keeper.command(sim)
        forecast, geometry = keeper.forecast, keeper.geometry
        position = copy.deepcopy(keeper.position)
        if geometry is None:
            raise RuntimeError(f'Benchmark snapshot cannot exercise full coverage: {name}')
        center = forecast['crossings'][str(float(keeper.planes[3]))]
        target = float(np.clip(center['lateral_m'], -.65, .65))
        decision = geometry['decision_lateral_m']

        def prepare_command(refresh=True, boundary=True):
            restore()
            keeper.records.clear()
            keeper.last_boundary = -1
            keeper.position = copy.deepcopy(position)
            keeper.forecast = None if refresh else forecast
            keeper.geometry = None if refresh else geometry
            keeper.forecast_time = saved_time-.02
            sim.physics_steps = round(saved_time/.001)+(0 if boundary else 1)
            return data

        def forecast_only():
            return keeper.predictor.forecast(saved_qpos[q:q+7],
                saved_qvel[v:v+6], keeper.planes)

        def coverage_only():
            return [keeper.coverage.evaluate(data, forecast['path'], lateral)
                    for lateral in (None, target, decision)]

        metrics = {}
        metrics['ball_predictor_forecast'] = measure(restore, forecast_only, warmup, repeats)
        metrics['coverage_three_evaluations'] = measure(restore, coverage_only, warmup, repeats)
        metrics['full_command_refresh_including_coverage'] = measure(
            prepare_command, lambda: keeper.command(sim), warmup, repeats)
        if keeper.refresh_count < warmup+repeats+1 or not keeper.records[-1]['forecast_refreshed']:
            raise RuntimeError('Full command benchmark did not exercise refresh')
        metrics['command_20ms_boundary_without_refresh'] = measure(
            lambda: prepare_command(False), lambda: keeper.command(sim), warmup, repeats)
        if keeper.records[-1]['forecast_refreshed']:
            raise RuntimeError('Non-refresh benchmark unexpectedly forecast')
        metrics['command_1ms_between_policy_boundaries'] = measure(
            lambda: prepare_command(False, False), lambda: keeper.command(sim), warmup, repeats)
        if keeper.records:
            raise RuntimeError('Between-boundary call unexpectedly made a decision')
        if not all(np.array_equal(actual, expected) for actual, expected in
                ((data.qpos, saved_qpos), (data.qvel, saved_qvel), (data.ctrl, saved_ctrl))):
            raise RuntimeError('G calls modified the caller simulation state')
        metrics['official_compute_targets_without_physics'] = measure(
            restore, controller.compute_targets, warmup, repeats)
        metrics['official_full_20ms_step_including_pd_and_physics'] = measure(
            restore, controller.step, warmup, repeats, expected_delta=.02)
        refresh_ms = metrics['full_command_refresh_including_coverage']['max_ms']
        official_ms = metrics['official_full_20ms_step_including_pd_and_physics']['max_ms']
        row = {'kind': kind, 'source': str(path.relative_to(ROOT)), 'archive_index': index,
            'archived_sim_time_s': saved_time, 'trajectory_sha256': summary['trajectory_sha256'],
            'current_ball_qpos7': saved_qpos[q:q+7].tolist(),
            'current_ball_qvel6': saved_qvel[v:v+6].tolist(),
            'prediction_planes_m': keeper.planes,
            'forecast': {'valid': forecast['valid'], 'simulated_future_duration_s': float(forecast['path'][-1, 0]),
                'path_points': len(forecast['path']), 'floor_contact_episodes': len(forecast['floor_contact_events']),
                'collision_robot_geoms': len(keeper.coverage.geoms)},
            'metrics': metrics, 'g_caller_state_unchanged': True,
            'estimated_serial_max_sum_ms': refresh_ms+official_ms,
            'serial_max_sum_exceeds_20ms': refresh_ms+official_ms > 20.}
        report['cases'].append(row)
        controller.stop()
        print(kind, json.dumps({key: {k: value[k] for k in ('median_ms', 'p95_ms', 'max_ms')}
                               for key, value in metrics.items()}), flush=True)

    report['protected_unchanged'] = protected == {
        str(p.relative_to(ROOT)): digest(p) for p in sorted(protected_paths)}
    verify_sources()
    report['runtime_limits'] = {
        'forecast_fixed_maximum_steps': 18000, 'forecast_wall_deadline_enforced': False,
        'forecast_async': False, 'forecast_cancelable': False,
        'refresh_uses': 'simulation time; expensive synchronous call holds the control-loop caller',
        'budget_met_all_measured_refresh_calls': all(c['metrics']['full_command_refresh_including_coverage']
            ['exceeds_100ms_refresh_count'] == 0 for c in report['cases']),
        'policy_budget_met_all_measured_refresh_calls': all(c['metrics']['full_command_refresh_including_coverage']
            ['exceeds_20ms_policy_count'] == 0 for c in report['cases']),
        'real_time_claim': False,
        'interpretation': 'Caller simulation time does not jump during prediction, but wall clock and physical observations age while the synchronous call blocks. Offline deterministic replay does not establish 50 Hz real-time control.',
        'limits': ['three saved initial shot states on one host, finite repetition count',
            'no recorded full policy history, viewer/render/network/sensor latency or scheduler jitter budget',
            'separate maximum sum is a conservative measured component estimate, not an end-to-end timing measurement',
            'horizon/no-crossing and later-state worst cases are not characterized by these three snapshots',
            '100 ms refresh interval does not grant a synchronous call 100 ms within a 20 ms policy deadline']}
    if not report['protected_unchanged']:
        raise RuntimeError('Protected existing source/test/report evidence changed during audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/defense_validation/predictor_latency.json')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    run(args.output, args.warmup, args.repeats)
