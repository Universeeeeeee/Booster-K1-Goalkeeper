"""H3 hooks around the original E recorder; H1 control remains unchanged."""
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from .defense_controller import ARM_JOINTS, PostFilteredArmController
from .defense_validation import contact_state, _pairs_by_step
from .impact_validation import run_impact
from .velocity_validation import recovery_metrics, trajectory_hash


def run_followup_trial(model, case, *, pulse=None, launch=None, segments=(),
                       recovery_windows=(), baseline=None, allow_ball_contact=False,
                       guard_start_s=2., output=None):
    """Return E summary/trace plus policy-tick arm and force-bearing pair evidence.

    Body commands are explicit (start, end, velocity3) segments. A matched baseline
    supplies same-step contact comparisons and joint response from its E qpos.
    """
    policy = {key: [] for key in ('h_policy_time', 'h_nominal_targets',
                                  'h_delivered_targets', 'h_arm_offset')}
    contacts = {key: [] for key in ('self_contacts', 'nonfoot_ground_contacts',
                                    'external_robot_contacts')}
    context = {'installed': False, 'wrapper': None, 'stop': None, 'policy_count': -1,
               'saturation_duration': np.zeros(8), 'last_offset': None, 'last_tick': None}
    dt = float(model.opt.timestep)
    baseline_self = _pairs_by_step(baseline, 'self_contacts', dt)
    baseline_ground = _pairs_by_step(baseline, 'nonfoot_ground_contacts', dt)
    ball_geom = model.geom('ball_collision').id

    def reject(sim, reason, kind='invalid', before_delivery=False):
        context['stop'] = {'kind': kind, 'reason': reason, 'time': float(sim.data.time),
                           'before_delivery': before_delivery}
        sim.controller.stop()
        raise RuntimeError('H3 guard: '+reason)

    def command(sim):
        if not context['installed']:
            c = sim.controller
            context['arm_indices'] = [c.cfg.robot.joint_names.index(name) for name in ARM_JOINTS]
            context['joint_qpos_addresses'] = c.qpos_addresses.copy()
            if pulse is not None:
                wrapper = PostFilteredArmController(c, pulse)
                context['wrapper'] = wrapper

                def checked_targets():
                    targets = wrapper.compute_targets()
                    offset = wrapper.offset.numpy()
                    selected = np.flatnonzero(offset != 0)
                    if len(selected):
                        ranges = model.jnt_range[c.joint_ids[selected]]
                        candidate = targets.numpy()[selected]
                        if np.any(candidate < ranges[:, 0]+.05) or np.any(candidate > ranges[:, 1]-.05):
                            reject(sim, 'selected target margin rejected before PD', before_delivery=True)
                    now = float(sim.data.time)
                    amplitudes = {c.cfg.robot.joint_names.index(j): abs(a) for j, a in pulse.joint_offsets}
                    for index, amplitude in amplitudes.items():
                        if abs(offset[index]) > amplitude+1e-7:
                            reject(sim, 'residual exceeds frozen amplitude', before_delivery=True)
                        if context['last_offset'] is not None:
                            slew = abs(offset[index]-context['last_offset'][index])/(now-context['last_tick'])
                            if slew > 1.875*amplitude/min(pulse.rise_s, pulse.return_s)+1e-5:
                                reject(sim, 'residual slew exceeds frozen quintic bound', before_delivery=True)
                    context['last_offset'], context['last_tick'] = offset.copy(), now
                    return targets

                c.compute_targets = checked_targets
            context['installed'] = True
        # These current-state copies are only for the H1-equivalent raw-PD guard;
        # E owns the full qpos/qvel and torque records.
        context['pre_q'] = sim.data.qpos.astype(np.float32)[sim.controller.qpos_addresses].copy()
        context['pre_dq'] = sim.data.qvel.astype(np.float32)[sim.controller.dof_addresses].copy()
        now = float(sim.data.time)
        return next((tuple(velocity) for start, end, velocity in segments
                     if start-1e-9 <= now < end-1e-9), (0., 0., 0.))

    def observe(sim):
        c, data, now = sim.controller, sim.data, float(sim.data.time)
        wrapper = context['wrapper']
        delivered = sim.targets.detach().cpu().numpy()
        nominal = wrapper.nominal_targets.numpy() if wrapper else delivered
        offset = wrapper.offset.numpy() if wrapper else np.zeros_like(delivered)
        if sim.policy_steps != context['policy_count']:
            context['policy_count'] = sim.policy_steps
            for key, value in zip(policy, (now-dt, nominal.copy(), delivered.copy(), offset.copy())):
                policy[key].append(value)
        state = contact_state(model, data)
        for key, state_key in (('self_contacts', 'self_contact_pairs'),
                               ('nonfoot_ground_contacts', 'nonfoot_ground_pairs'),
                               ('external_robot_contacts', 'external_robot_pairs')):
            for pair in sorted(state[state_key]):
                contacts[key].append([now, *pair])
        if now < guard_start_s-1e-9:
            return
        raw = c.robot.joint_stiffness.numpy()*(delivered-context['pre_q'])-c.robot.joint_damping.numpy()*context['pre_dq']
        limits = c.robot.effort_limit.numpy()
        arm_indices = context['arm_indices']
        context['saturation_duration'] = np.where(np.abs(raw[arm_indices]) >= limits[arm_indices]-1e-6,
                                                 context['saturation_duration']+dt, 0.)
        quat = data.qpos[c.base_qpos_address+3:c.base_qpos_address+7]
        tilt = float(np.degrees(np.arccos(np.clip(1-2*(quat[1]**2+quat[2]**2), -1, 1))))
        reason, kind = None, 'invalid'
        if not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc, data.ctrl,
                                                          delivered, nominal, offset, raw, data.qfrc_actuator)):
            reason, kind = 'nonfinite state or control', 'numerical'
        elif data.qpos[c.base_qpos_address+2] < .42 or tilt > 15:
            reason, kind = 'balance guard: z<0.42 m or tilt>15 deg', 'instability'
        elif state['self_contact_pairs']-baseline_self.get(round(now/dt), set()):
            reason, kind = 'new active self-contact', 'instability'
        elif state['nonfoot_ground_pairs']-baseline_ground.get(round(now/dt), set()):
            reason, kind = 'new active non-foot ground contact', 'instability'
        elif any(not (allow_ball_contact and ball_geom in pair) for pair in state['external_robot_pairs']):
            reason, kind = 'unexpected active robot-goal/external contact', 'instability'
        elif np.any(context['saturation_duration'] > .10+1e-12):
            reason = 'continuous arm raw saturation exceeds 0.10 s'
        elif np.any(np.abs(data.ctrl[c.actuator_ids]) > limits+1e-5) or np.any(np.abs(data.qfrc_actuator[c.dof_addresses]) > limits+1e-5):
            reason = 'delivered control/torque exceeds official effort limit'
        elif not np.array_equal(delivered[offset == 0], nominal[offset == 0]):
            reason = 'unselected target changed'
        selected = np.flatnonzero(offset != 0)
        if reason is None and len(selected):
            ranges = model.jnt_range[c.joint_ids[selected]]
            if np.any(delivered[selected] < ranges[:, 0]+.05) or np.any(delivered[selected] > ranges[:, 1]-.05):
                reason = 'selected target violates XML 0.05 rad margin'
            elif np.any(data.qpos[c.qpos_addresses[selected]] < ranges[:, 0]) or np.any(data.qpos[c.qpos_addresses[selected]] > ranges[:, 1]):
                reason = 'selected actual joint outside XML range'
        if reason:
            reject(sim, reason, kind)

    e_summary, e_trace = run_impact(model, case, launch=launch,
                                   command_provider=command, step_observer=observe)
    trace = {**e_trace, **{key: np.asarray(value) for key, value in policy.items()},
             **{key: np.asarray(value, dtype=float).reshape(-1, 3) for key, value in contacts.items()}}
    stable = (context['stop'] is None and e_summary['completed'] and not e_summary['fall_events']
              and e_summary['numerical_error'] is None and e_summary['policy_stop'] is None)
    velocity = np.column_stack([trace['body_linear_velocity'][:, :2], trace['body_angular_velocity'][:, 2]])
    recoveries = [{'start_s': start, 'end_s': end,
                   **recovery_metrics(trace, velocity, start, end, stable and trace['time'][-1] >= end-1e-9)}
                  for start, end in recovery_windows]
    recovered = bool(recoveries and all(row['verified'] for row in recoveries))
    response = []
    if pulse is not None and baseline is not None:
        end = pulse.start_s+pulse.rise_s+pulse.hold_s
        mask = (trace['time'] >= end-1-1e-9) & (trace['time'] < end-1e-9)
        times = trace['time'][mask]
        for joint, amplitude in pulse.joint_offsets:
            if amplitude == 0:
                continue
            address = int(model.joint(joint).qposadr[0])
            fraction = None
            if len(times) == 1000 and baseline['time'][-1] >= end-1e-9:
                matched = np.interp(times, baseline['time'], baseline['qpos'][:, address])
                fraction = float(np.median(np.sign(amplitude)*(trace['qpos'][mask, address]-matched))/abs(amplitude))
            response.append({'joint': joint, 'amplitude_rad': amplitude,
                             'window_s': [end-1, end], 'median_signed_response_fraction': fraction,
                             'passed': fraction is not None and fraction >= .5})
    authority = all(row['passed'] for row in response) if pulse is not None else True
    if pulse is not None and any(a for _, a in pulse.joint_offsets) and not response:
        authority = False
    target_contacts = [event for event in e_summary['contacts']
                       if event['kind'] == 'robot' and event['body'] == case.target_body
                       and event['normal_impulse_ns'] > 0]
    contact_verified = bool(target_contacts) if allow_ball_contact else None
    impact_verified = bool(stable and recovered and contact_verified) if allow_ball_contact else None
    passed = bool(stable and recovered and authority
                  and (not allow_ball_contact or pulse is None or contact_verified))
    numerical = context['stop'] is None or context['stop']['kind'] == 'numerical'
    summary = {'name': case.name, 'stage': 'H3', 'pulse': asdict(pulse) if pulse else None,
        'body_command_segments': list(segments), 'allow_ball_contact': allow_ball_contact,
        'recorder': 'unchanged E run_impact; only policy-tick arm targets and active pair rows added',
        'joint_qpos_addresses': context.get('joint_qpos_addresses', np.empty(0, dtype=int)).tolist(),
        'e_summary': e_summary, 'stop': context['stop'], 'stable': bool(stable),
        'recovery_windows': recoveries, 'recovered': recovered, 'response': response,
        'authority_passed': bool(authority), 'target_native_contacts': target_contacts,
        'target_native_contact_verified': contact_verified,
        'contact_verification': ('verified' if contact_verified else 'unverifiable') if allow_ball_contact else 'not_applicable',
        'impact_validation_passed': impact_verified, 'passed': passed,
        'failure_class': 'instability_or_invalid' if not stable else 'recovery' if not recovered
                         else 'authority' if not authority else 'unverifiable_contact' if not passed else None,
        'numerical_error': e_summary['numerical_error'] if numerical else None,
        'censored_by_deliberate_guard': context['stop'] is not None and not numerical,
        'recorder_guard_diagnostic': e_summary['numerical_error'] if context['stop'] is not None and not numerical else None,
        'trajectory_sha256': trajectory_hash(trace)}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/'trajectory.npz', **trace)
        (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary, trace
