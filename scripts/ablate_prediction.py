"""Post-hoc diagnostic of the new held-out bounce success; not a tuning set."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
from k1_goalkeeper.ball_prediction import build_ball_floor_model
from k1_goalkeeper.positioning import LateralPositionController
from k1_goalkeeper.positioning_validation import save_trial
from k1_goalkeeper.predictive_goalkeeper import run_g_trial


def run():
    output = ROOT/'reports/prediction_validation'
    protocol = json.loads((output/'protocol.json').read_text())
    shot = next(s for s in protocol['holdout'] if s['name'] == 'bounce_early_right')
    model = mujoco.MjModel.from_xml_path(str(output/'scene_free.xml'))
    predictor = build_ball_floor_model(output/'scene_absent.xml')
    repeated = []
    for repeat in range(2):
        # Only this diagnostic uses the original F stop/restart controller under G predictions.
        with patch('k1_goalkeeper.predictive_goalkeeper.SettlingPositionController', LateralPositionController):
            summary, trace = run_g_trial(model, predictor, 'bounce_early_right_g_original_position', shot['launch'])
        repeated.append(summary['trajectory_sha256'])
        if repeat == 0:
            first, first_trace = summary, trace
    first['reproducibility'] = {'verified': len(set(repeated)) == 1, 'hashes': repeated}
    first['ablation'] = 'G predictor/geometry/decision with unmodified F position controller; post-hoc diagnostic, no parameter changes'
    save_trial(output/'ablation'/first['case']['name'], first, first_trace)
    absent = json.loads((output/'bounce_early_right_absent/summary.json').read_text())
    result = {'selection': 'Post-hoc selected new success; outside frozen primary protocol, not independent generalization evidence',
              'name': shot['name'], 'change': 'Replace only G settling-margin subclass with original F LateralPositionController',
              'reproducibility': first['reproducibility'], 'contact': first['effective_contact'],
              'right_goal': first['right_goal'], 'blocked': bool(first['effective_contact'] and not first['right_goal'])
              if absent['right_goal'] and first['completed'] else None,
              'fall': bool(first['fall_events']), 'numerical_error': first['numerical_error'],
              'recovery': first['final_stop_recovery'],
              'code_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                  ('scripts/ablate_prediction.py', 'k1_goalkeeper/ball_prediction.py', 'k1_goalkeeper/predictive_goalkeeper.py', 'k1_goalkeeper/positioning.py')}}
    (output/'ablation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
