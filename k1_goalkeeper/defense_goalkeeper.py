"""H4 offline prototype: G forecast, stop, then the independently tested arm pulse."""
from dataclasses import asdict
from unittest.mock import patch

import numpy as np

from .defense_controller import ARM_JOINTS, DefensePulse, PostFilteredArmController
from .defense_validation import contact_state
from .positioning_validation import run_trial, save_trial
from .predictive_goalkeeper import PredictiveGoalkeeper
from .velocity_validation import body_state, recovery_metrics, trajectory_hash


class DefenseGoalkeeper(PredictiveGoalkeeper):
    """Explicit experiment; arm activation temporarily suspends lateral walking.

    Timing uses current observations only. Simulated time does not account for
    synchronous G wall-clock latency; this is not a real-time controller claim.
    """
    def __init__(self, model, predictor_model):
        super().__init__(model, predictor_model)
        self.wrapper = None
        self.pulse = None
        self.stationary_since = None
        self.arm_request = False
        self.guard_event = None
        self.policy_count = -1
        self.arm_rows = []
        self.contact_rows = []
        self.saturation_duration = np.zeros(8)

    def guard(self, sim, reason, numerical=False):
        self.guard_event = {'time': float(sim.data.time), 'reason': reason,
                            'numerical': numerical,
                            'delivered_state': {key: [float(v) if np.isfinite(v) else str(v)
                                                      for v in getattr(sim.data, key)]
                                                for key in ('qpos', 'qvel', 'ctrl', 'qfrc_actuator')},
                            'active_pairs': {key: sorted(value) for key, value
                                             in contact_state(sim.model, sim.data).items()}}
        sim.controller.stop()
        raise RuntimeError('H4 guard: '+reason)

    def install(self, sim):
        zero = DefensePulse((('left_shoulder_roll_joint', 0.),
                             ('right_shoulder_roll_joint', 0.)))
        self.wrapper = PostFilteredArmController(sim.controller, zero)

        def targets():
            result = self.wrapper.compute_targets()
            selected = np.flatnonzero(self.wrapper.offset.numpy())
            if len(selected):
                ranges = sim.model.jnt_range[sim.controller.joint_ids[selected]]
                values = result.numpy()[selected]
                if np.any(values < ranges[:, 0]+.05) or np.any(values > ranges[:, 1]-.05):
                    self.guard(sim, 'selected target outside 0.05 rad XML margin')
            return result

        sim.controller.compute_targets = targets

    def command(self, sim):
        if self.wrapper is None:
            self.install(sim)
        self.pre_q = sim.data.qpos.astype(np.float32)[sim.controller.qpos_addresses].copy()
        self.pre_dq = sim.data.qvel.astype(np.float32)[sim.controller.dof_addresses].copy()
        count = len(self.records)
        command = super().command(sim)
        if count == len(self.records):
            return command
        now = float(sim.data.time)
        row = self.records[-1]
        state = body_state(sim.controller)
        slow = (np.all(np.abs(state['body_linear_velocity'][:2]) <= .05)
                and abs(state['body_angular_velocity'][2]) <= .1
                and np.cos(state['rpy'][0])*np.cos(state['rpy'][1]) >= np.cos(np.deg2rad(10)))
        if not slow:
            self.stationary_since = None
        elif self.stationary_since is None:
            self.stationary_since = now
        prediction = row.get('prediction')
        early = row.get('interception_prediction')
        incoming = bool(now > 2. and prediction and prediction['valid'] and prediction['inside_aperture']
                        and early and early['valid'] and early['time_to_plane_s'] > 0
                        and row['ball_velocity'][0] > .05
                        and row['ball_position'][0] < row['world_position'][0]
                        and not self.contact_seen)
        if incoming and self.pulse is None:
            self.arm_request = True
            if slow and now-self.stationary_since >= 1.-1e-9:
                self.pulse = DefensePulse((('left_shoulder_roll_joint', .15),
                                           ('right_shoulder_roll_joint', -.15)), start_s=now)
                self.wrapper.pulse = self.pulse
        if self.arm_request and self.pulse is None and not incoming:
            self.arm_request = False
        # Preserve the nominal decision for audit. Do not resume its internal
        # positioning state after suppressing commands in this single-shot trial.
        row['g_nominal_command'] = list(command)
        if self.arm_request:
            command = [0., 0., 0.]
        self.last_command = command
        row['command'] = list(command)
        row['h_arm_request'] = self.arm_request
        row['h_pulse_start_s'] = self.pulse.start_s if self.pulse else None
        row['h_requested_pose_fraction'] = self.pulse.factor(now) if self.pulse else 0.
        row['h_full_requested_pose_ready_s'] = self.pulse.start_s+2. if self.pulse else None
        row['h_full_rise_fits_current_deadline'] = bool(incoming and early['time_to_plane_s'] >= 2.)
        row['h_late_partial_experiment'] = bool(incoming and early['time_to_plane_s'] < 2.)
        row['h_control_note'] = 'stop then native both-roll pulse; positive but short deadlines are labeled late partial experiments, not credible interception plans; hold zero walking command after action'
        return command

    def observe(self, sim):
        super().observe(sim)
        c, t = sim.controller, float(sim.data.time)
        if sim.policy_steps != self.policy_count:
            self.policy_count = sim.policy_steps
            self.arm_rows.append(np.r_[t-sim.model.opt.timestep,
                self.wrapper.nominal_targets.numpy(), self.wrapper.delivered_targets.numpy(),
                self.wrapper.offset.numpy()])
        state = contact_state(sim.model, sim.data)
        ball = sim.model.geom('ball_collision').id
        for kind, key in enumerate(('self_contact_pairs', 'nonfoot_ground_pairs', 'external_robot_pairs')):
            for pair in sorted(state[key]):
                self.contact_rows.append([t, kind, *pair])
        if t < 2.:
            return
        body = body_state(c)
        tilt = np.degrees(np.arccos(np.clip(np.cos(body['rpy'][0])*np.cos(body['rpy'][1]), -1, 1)))
        raw = c.robot.joint_stiffness.numpy()*(sim.targets.numpy()-self.pre_q)-c.robot.joint_damping.numpy()*self.pre_dq
        arm_ids = [c.cfg.robot.joint_names.index(name) for name in ARM_JOINTS]
        effort = c.robot.effort_limit.numpy()
        self.saturation_duration = np.where(np.abs(raw[arm_ids]) >= effort[arm_ids]-1e-6,
                                            self.saturation_duration+sim.model.opt.timestep, 0.)
        if not all(np.isfinite(value).all() for value in (raw, sim.data.qpos, sim.data.qvel, sim.data.qacc,
                                                         sim.data.ctrl, sim.data.qfrc_actuator)):
            self.guard(sim, 'nonfinite state or control', numerical=True)
        if body['base_position'][2] < .42 or tilt > 15:
            self.guard(sim, 'height/tilt conservative safety limit')
        if state['self_contact_pairs'] or state['nonfoot_ground_pairs']:
            self.guard(sim, 'active self-contact or nonfoot ground contact')
        if any(ball not in pair for pair in state['external_robot_pairs']):
            self.guard(sim, 'robot-goal contact')
        if np.any(self.saturation_duration > .10+1e-12):
            self.guard(sim, 'continuous arm torque saturation above 0.10s')
        if (np.any(np.abs(sim.data.ctrl[c.actuator_ids]) > effort+1e-5)
                or np.any(np.abs(sim.data.qfrc_actuator[c.dof_addresses]) > effort+1e-5)):
            self.guard(sim, 'delivered torque exceeds official limit')
        selected = np.flatnonzero(self.wrapper.offset.numpy())
        if len(selected):
            limits = sim.model.jnt_range[c.joint_ids[selected]]
            actual = sim.data.qpos[c.qpos_addresses[selected]]
            if np.any(actual < limits[:, 0]) or np.any(actual > limits[:, 1]):
                self.guard(sim, 'selected actual joint outside XML range')


def run_h_trial(model, predictor_model, name, launch, output=None):
    adapter = DefenseGoalkeeper(model, predictor_model)
    with patch('k1_goalkeeper.positioning_validation.TruthGoalkeeper', return_value=adapter):
        summary, trace = run_trial(model, name, 'active', launch)
    trace['h_arm_policy'] = np.asarray(adapter.arm_rows).reshape(-1, 67)
    trace['h_robot_contacts'] = np.asarray(adapter.contact_rows).reshape(-1, 4)
    summary.update(stage='H4', mode='experimental_stop_then_arm',
                   arm_pulse=asdict(adapter.pulse) if adapter.pulse else None,
                   h_guard=adapter.guard_event,
                   timing_scope='offline simulation; synchronous G wall latency not injected',
                   h_arm_policy_columns='time, nominal22, delivered22, offset22')
    if adapter.guard_event and not adapter.guard_event['numerical']:
        summary['recorder_guard_diagnostic'] = summary['numerical_error']
        summary['numerical_error'] = None
    contacts = [c for c in summary['contacts'] if c['kind'] == 'robot']
    times = [c['start_time'] for c in contacts]
    crossing = summary['plane_crossings'].get('11.0')
    events = times + ([crossing['time']] if crossing else [])
    deadline = min(events) if events else None
    pulse = adapter.pulse
    summary['action_timing'] = {
        'reference': 'earliest actual robot contact or goal center-plane crossing',
        'event_time_s': deadline,
        'full_requested_pose_time_s': pulse.start_s+2 if pulse else None,
        'full_requested_pose_before_event': bool(pulse and deadline and pulse.start_s+2 <= deadline),
        'full_requested_pose_held_at_event': bool(pulse and deadline and pulse.start_s+2 <= deadline <= pulse.start_s+4),
        'requested_fraction_at_event': pulse.factor(deadline) if pulse and deadline else None,
        'note': 'requested envelope is not proof of actual pose tracking; actual qpos and targets saved'}
    velocity = np.column_stack([trace['body_linear_velocity'][:, :2], trace['body_angular_velocity'][:, 2]])
    start = pulse.start_s+6. if pulse else 9.
    summary['arm_return_recovery'] = recovery_metrics(trace, velocity, start, 20.,
        summary['completed'] and not summary['fall_events'] and not summary['numerical_error'] and not adapter.guard_event)
    summary['trajectory_sha256'] = trajectory_hash(trace)
    if output:
        save_trial(output, summary, trace)
    return summary, trace
