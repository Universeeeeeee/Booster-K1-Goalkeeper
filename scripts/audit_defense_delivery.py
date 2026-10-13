"""Independent H4 delivered-state/PD and protected-model verification."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from k1_goalkeeper.scene_simulation import SceneSimulation
from k1_goalkeeper.velocity_validation import trajectory_hash


def run():
    directory = ROOT/'reports/defense_validation'
    results = json.loads((directory/'goalkeeper/results.json').read_text())
    protected = json.loads((directory/'protected_manifest.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in protected.items())
    model = mujoco.MjModel.from_xml_path(str(directory/'goalkeeper/scene_free.xml'))
    c = SceneSimulation(model, control_mode='k1_walk').controller
    original_model = json.loads((directory/'model_audit.json').read_text())
    model_equal = {key: hashlib.sha256(getattr(model, key).tobytes()).hexdigest() == value
                   for key, value in original_model['array_sha256'].items()}
    assert all(model_equal.values())
    audits = []
    launches = []
    archive_count = 0
    for shot in results['shots']:
        reference = None
        for mode, metrics in shot['modes'].items():
            for path in metrics['paths']:
                path = ROOT/path
                summary = json.loads((path/'summary.json').read_text())
                with np.load(path/'trajectory.npz') as archive:
                    trace = {key: archive[key] for key in archive.files}
                assert trajectory_hash(trace) == summary['trajectory_sha256']
                archive_count += 1
                launch = {key: summary['launch'][key] for key in ('position', 'velocity', 'angular_velocity', 'time')}
                if reference is None:
                    reference = launch
                assert launch == reference
                if mode != 'h_stop_then_arm':
                    continue
                h = trace['h_arm_policy']
                assert np.array_equal(np.rint(h[:, 0]/.02).astype(int), np.arange(len(h)))
                # The compact matrix stores float32 times; use discrete physics
                # indices, not an exact floating-time join, to select targets.
                ticks = np.arange(len(trace['time'])-1)//20
                nominal, targets, offset = h[:, 1:23], h[:, 23:45], h[:, 45:67]
                assert np.array_equal(targets, nominal+offset)
                q = trace['qpos'][:-1, c.qpos_addresses].astype(np.float32)
                dq = trace['qvel'][:-1, c.dof_addresses].astype(np.float32)
                raw = c.robot.joint_stiffness.numpy()*(targets[ticks]-q)-c.robot.joint_damping.numpy()*dq
                controls = np.clip(raw, -c.robot.effort_limit.numpy(), c.robot.effort_limit.numpy())
                assert np.array_equal(raw, trace['raw_torque'][1:])
                assert np.array_equal(controls, trace['motor_controls'][1:])
                assert np.array_equal(controls, trace['applied_torque'][1:])
                pulse = summary['arm_pulse']
                start = pulse['start_s']
                before = (trace['time'] >= start-1-1e-9) & (trace['time'] < start-1e-9)
                speed = np.abs(trace['body_linear_velocity'][before, :2]).max(axis=0)
                yaw = np.abs(trace['body_angular_velocity'][before, 2]).max()
                assert np.all(speed <= .05) and yaw <= .1
                assert np.all(trace['command'][trace['time'] >= start-1e-9] == 0)
                arm = [c.cfg.robot.joint_names.index(name) for name, _ in pulse['joint_offsets']]
                timestep_before_event = max(0, np.searchsorted(trace['time'], summary['action_timing']['event_time_s'])-1)
                tick = max(0, (timestep_before_event-1)//20)
                actual_q = trace['qpos'][timestep_before_event, c.qpos_addresses[arm]]
                audits.append({'path': str(path.relative_to(ROOT)), 'pd_raw_motor_applied_exact': True,
                    'policy_time_dtype': str(h.dtype),
                    'maximum_time_quantization_s': float(np.abs(h[:, 0].astype(float)-np.arange(len(h))*.02).max()),
                    'pre_action_1ms_maximum_abs_body_xy_velocity': speed.tolist(),
                    'pre_action_1ms_maximum_abs_yaw_rate': float(yaw),
                    'post_action_body_commands_zero': True,
                    'arm_joint_names': [c.cfg.robot.joint_names[i] for i in arm],
                    'last_pre_event_time_s': float(trace['time'][timestep_before_event]),
                    'actual_arm_q_at_pre_event_rad': actual_q.tolist(),
                    'delivered_arm_target_at_pre_event_rad': targets[tick, arm].tolist(),
                    'actual_target_error_at_pre_event_rad': (actual_q-targets[tick, arm]).tolist()})
        launches.append({'shot': shot['name'], 'all_eight_launch_states_identical': True, 'launch': reference})
    report = {'stage': 'H4 independent delivery audit', 'archives_verified': archive_count,
              'protected_file_count': len(protected), 'protected_files_unchanged': True,
              'native_model_arrays_match_H0': model_equal, 'paired_launches': launches,
              'method': 'reconstruct unchanged PD from prior actual qpos/qvel and delivered target at integer policy tick; compare raw demand, motor controls and applied actuator torque exactly',
              'time_note': 'compact H target matrix timestamps are float32 with <1us quantization; original E physical time remains float64 and governs control',
              'runs': audits}
    (directory/'goalkeeper/delivered_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Verified', archive_count, 'archives;', len(audits), 'H runs with exact delivered PD; protected', len(protected))
    return report


if __name__ == '__main__':
    run()
