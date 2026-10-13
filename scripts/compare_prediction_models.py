"""Compare F and G forecasts from matched, current absent-ball states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from k1_goalkeeper.ball_prediction import BallPredictor, build_ball_floor_model
from k1_goalkeeper.positioning import predict_ball


def aggregate(samples, model_name):
    predictions = [sample[model_name] for sample in samples]
    valid = [prediction for prediction in predictions if prediction['valid']]
    supported = [prediction for prediction in valid if not prediction.get('future_floor_contact_unmodeled', False)]
    errors = [prediction['error'] for prediction in valid if prediction.get('error') is not None]
    supported_errors = [prediction['error'] for prediction in supported if prediction.get('error') is not None]

    def summarize(values):
        array = np.asarray(values, dtype=float)
        return {'count': len(array), 'rmse': np.sqrt(np.mean(array**2, axis=0)).tolist(),
                'maximum_absolute': np.abs(array).max(axis=0).tolist(),
                'mean_signed': array.mean(axis=0).tolist()} if len(array) else None

    return {'sample_count': len(predictions), 'valid_count': len(valid),
            'invalid_count': len(predictions)-len(valid),
            'invalid_rate': (len(predictions)-len(valid))/len(predictions) if predictions else None,
            'unmodeled_floor_count': len(valid)-len(supported),
            'supported_count': len(supported), 'errors_all_valid': summarize(errors),
            'errors_without_unmodeled_floor': summarize(supported_errors)}


def evaluate(protocol_path, regression_dir, holdout_dir, output, sample_period=.25):
    protocol = json.loads(protocol_path.read_text())
    if not np.isfinite(sample_period) or sample_period <= 0:
        raise ValueError('sample period must be finite and positive')
    report = {'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'method': 'F and G receive identical current absent qpos/qvel; actual crossings are used only after predictions for scoring',
              'sampling': {'period_s': sample_period, 'start': 'first recorded launched state',
                           'stop': 'observed current ball reaches the final requested plane, or trace ends',
                           'per_plane_eligibility': 'observed current ball x is before that plane'},
              'score_columns': ['arrival_time_error_s', 'lateral_error_m', 'height_error_m'],
              'interpretation': ['F valid predictions with future_floor_contact_unmodeled are separately counted and scored; their large ballistic errors are not operationally usable forecasts',
                                 'Prediction accuracy in known simulation physics is distinct from controller interception success',
                                 'No source launch, saved future positions, or crossing references are passed to either predictor'],
              'source_sha256': {}, 'cases': [], 'missing_cases': [], 'groups': {}}
    models = {}
    all_samples = []
    for split in ('regression', 'holdout'):
        directory = regression_dir if split == 'regression' else holdout_dir
        scene = directory/'scene_absent.xml'
        if not scene.exists():
            # G may write only its free scene; the builder removes its robot and goals.
            scene = directory/'scene_free.xml'
        for shot in protocol[split]:
            path = directory/(shot['name']+'_absent')
            files = [path/'summary.json', path/'trajectory.npz', scene]
            if not all(file.exists() for file in files):
                report['missing_cases'].append({'split': split, 'name': shot['name'],
                    'missing': [str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file) for file in files if not file.exists()]})
                continue
            for file in files:
                key = str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)
                if key not in report['source_sha256']:
                    report['source_sha256'][key] = hashlib.sha256(file.read_bytes()).hexdigest()
            if str(scene) not in models:
                models[str(scene)] = BallPredictor(build_ball_floor_model(scene))
            predictor = models[str(scene)]
            summary = json.loads(files[0].read_text())
            if summary['case']['robot_mode'] != 'absent':
                raise ValueError(f'{path} is not an absent reference')
            for key in ('position', 'velocity', 'angular_velocity'):
                if not np.allclose(summary['launch'][key], shot['launch'][key], rtol=0, atol=1e-12):
                    raise ValueError(f'{path} launch differs from predeclared protocol: {key}')
            trace = np.load(files[1])
            if trace['qpos'].shape[1] != 7 or trace['qvel'].shape[1] != 6:
                raise ValueError(f'{path} must record one absent free ball')
            case = {'name': shot['name'], 'split': split, 'kind': shot['kind'],
                    'reference_crossings': summary['plane_crossings'], 'samples': []}
            launch_time = float(summary['launch']['time'])
            next_sample = launch_time
            for index, current_time in enumerate(trace['time']):
                if current_time < next_sample-1e-9:
                    continue
                qpos, qvel = trace['qpos'][index].copy(), trace['qvel'][index].copy()
                if qpos[0] >= 11.11:
                    break
                next_sample = float(current_time)+sample_period
                physical = predictor.forecast(qpos, qvel, planes=(11., 11.11))
                for plane in (11., 11.11):
                    if qpos[0] >= plane:
                        continue
                    baseline = predict_ball(qpos[:3], qvel[:3], qvel[3:], plane_x=plane,
                                            radius=predictor.radius)
                    prediction = physical['crossings'][str(plane)]
                    sample = {'time': float(current_time), 'plane_x': plane,
                              'F': dict(baseline), 'G': dict(prediction)}
                    # This is the first use of actual future truth after both models ran.
                    reference = summary['plane_crossings'].get(str(plane))
                    for model_name in ('F', 'G'):
                        result = sample[model_name]
                        result['error'] = [float(current_time+result['time_to_plane_s']-reference['time']),
                                           float(result['lateral_m']-reference['position'][1]),
                                           float(result['height_m']-reference['position'][2])] if result['valid'] and reference else None
                    case['samples'].append(sample)
                    all_samples.append((split, shot['kind'], sample))
            case['planes'] = {str(plane): {name: aggregate([sample for sample in case['samples'] if sample['plane_x']==plane], name)
                                         for name in ('F', 'G')} for plane in (11., 11.11)}
            report['cases'].append(case)
            print(split, shot['name'], 'samples', len(case['samples']), flush=True)
    for split in ('regression', 'holdout'):
        report['groups'][split] = {}
        for kind in ('all', 'rolling', 'bouncing', 'airborne'):
            report['groups'][split][kind] = {str(plane): {name: aggregate([sample for row_split, row_kind, sample in all_samples
                if row_split==split and (kind=='all' or row_kind==kind) and sample['plane_x']==plane], name)
                for name in ('F', 'G')} for plane in (11., 11.11)}
    report['complete'] = not report['missing_cases']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'reports/prediction_validation/protocol.json')
    parser.add_argument('--regression-dir', type=Path, default=ROOT/'reports/positioning_validation')
    parser.add_argument('--holdout-dir', type=Path, default=ROOT/'reports/prediction_validation')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/prediction_validation/prediction_comparison.json')
    parser.add_argument('--sample-period', type=float, default=.25)
    parser.add_argument('--allow-missing', action='store_true', help='Write an explicitly incomplete development report')
    args = parser.parse_args()
    report = evaluate(args.protocol, args.regression_dir, args.holdout_dir, args.output, args.sample_period)
    if not report['complete'] and not args.allow_missing:
        raise SystemExit('Missing protocol cases; comparison report is incomplete')
