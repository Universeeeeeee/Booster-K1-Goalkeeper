"""Stage G experiments; original Stage F controller and recorder stay intact."""
import copy
from unittest.mock import patch

import mujoco
import numpy as np

from .ball_prediction import BallPredictor
from .impact_validation import _robot_body
from .positioning import LateralPositionController, reachable_target
from .positioning_validation import TruthGoalkeeper, run_trial


class SettlingPositionController(LateralPositionController):
    """Keep F motion rules, with 2cm residual margin before the 10cm boundary."""
    def command(self, time, position, yaw, world_velocity, target):
        command, tracking = super().command(time, position, yaw, world_velocity, target)
        correction = (not self.direction and not self.turning and tracking['safe']
                      and abs(tracking['error_m']) > .08 and time-self.last_switch >= 1.2)
        if correction:
            self.direction = 1 if tracking['error_m'] > 0 else -1
            self.last_switch = time
            command = [0., float(np.sign(np.cos(yaw)*self.direction)*.2), 0.]
            tracking['direction'] = self.direction
        tracking['settling_correction_started'] = bool(correction)
        return command, tracking


class CollisionCoverage:
    """Native signed-distance queries on a private, frozen current robot pose.

    Future translation is a geometry hypothesis, not a simulated or guaranteed gait.
    Disabled visual meshes never participate. A 15mm overlap margin rejects grazes.
    """
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        self.ball = model.geom('ball_collision').id
        self.ball_q = int(model.joint('ball_free').qposadr[0])
        self.base_q = int(model.joint('world_joint').qposadr[0])
        self.geoms = [g for g in range(model.ngeom) if _robot_body(model, model.geom_bodyid[g])
                      and ((model.geom_contype[g] & model.geom_conaffinity[self.ball]) or
                           (model.geom_contype[self.ball] & model.geom_conaffinity[g]))]

    def evaluate(self, scene_data, path, target_lateral=None):
        self.data.qpos[:] = scene_data.qpos
        if target_lateral is not None:
            self.data.qpos[self.base_q+1] = target_lateral
        base_x = float(self.data.qpos[self.base_q])
        candidates = path[np.abs(path[:, 1]-base_x) <= .81] if len(path) else path
        best, witness = float('inf'), None
        for point in candidates:
            self.data.qpos[self.ball_q:self.ball_q+3] = point[1:4]
            mujoco.mj_kinematics(self.model, self.data)
            for geom in self.geoms:
                # Bounding spheres only reject distant pairs; positive coverage needs native exact distance.
                if np.linalg.norm(point[1:4]-self.data.geom_xpos[geom]) > self.model.geom_rbound[geom]+.12:
                    continue
                distance = mujoco.mj_geomDistance(self.model, self.data, self.ball, geom, .15, None)
                if distance < best:
                    best = float(distance)
                    witness = {'relative_time_s': float(point[0]), 'ball_position': point[1:4].tolist(),
                               'geom': mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                               'body': mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[geom]))}
                if best <= -.015:
                    return {'covered': True, 'signed_distance_m': best, 'witness': witness,
                            'required_overlap_m': .015, 'assumption': 'frozen current pose and candidate lateral translation'}
        return {'covered': False, 'signed_distance_m': best if np.isfinite(best) else None, 'witness': witness,
                'required_overlap_m': .015, 'assumption': 'frozen current pose; false can also mean sparse trajectory sampling'}


class PredictiveGoalkeeper(TruthGoalkeeper):
    def __init__(self, model, predictor_model, mode='active', target=0.):
        # Reuse F truth-state sampling and contact observer, without executing F active decisions.
        super().__init__('position' if mode == 'position' else 'passive', target=target)
        self.position = SettlingPositionController()
        self.g_mode = mode
        self.predictor = BallPredictor(predictor_model)
        self.coverage = CollisionCoverage(model)
        self.forecast = None
        self.forecast_time = -float('inf')
        self.geometry = None
        self.refresh_count = 0

    def command(self, sim):
        previous = self.last_boundary
        super().command(sim)
        if previous == self.last_boundary or self.g_mode != 'active':
            return self.last_command
        row = self.records[-1]
        now = row['time']
        row['f_prediction'] = row['prediction']
        row['f_whole_ball_prediction'] = row.get('whole_ball_prediction')
        row['station_reachable'] = None
        row['current_body_coverage'] = None
        row['target_body_coverage'] = None
        row['intercept_feasible'] = None
        command = [0., 0., 0.]
        if now >= 2.-1e-9 and not self.contact_seen:
            refresh = self.forecast is None or now-self.forecast_time >= .1-1e-9
            if refresh:
                ball = sim.model.joint('ball_free')
                q, v = int(ball.qposadr[0]), int(ball.dofadr[0])
                self.planes = [11., 11.11, row['world_position'][0]-.81, row['world_position'][0]]
                self.forecast = self.predictor.forecast(sim.data.qpos[q:q+7].copy(), sim.data.qvel[v:v+6].copy(), self.planes)
                self.forecast_time = now
                self.refresh_count += 1
                center = self.forecast['crossings'][str(float(self.planes[3]))]
                early_forecast = self.forecast['crossings'][str(float(self.planes[2]))]
                if center['valid'] and early_forecast['valid']:
                    target = float(np.clip(center['lateral_m'], -.65, .65))
                    reach_at_observation = reachable_target(center['lateral_m'], row['world_position'][1], self.position.filtered_velocity,
                                                            early_forecast['time_to_plane_s'], self.position.limits)
                    self.geometry = {'current': self.coverage.evaluate(sim.data, self.forecast['path']),
                                     'target': self.coverage.evaluate(sim.data, self.forecast['path'], target),
                                     'decision': self.coverage.evaluate(sim.data, self.forecast['path'], reach_at_observation['decision_lateral_m']),
                                     'decision_lateral_m': reach_at_observation['decision_lateral_m'], 'observed_time': now}
                else:
                    self.geometry = None
            elapsed = now-self.forecast_time
            predictions = copy.deepcopy(self.forecast['crossings'])
            for prediction in predictions.values():
                if prediction['valid']:
                    prediction['time_to_plane_s'] -= elapsed
                    if prediction['time_to_plane_s'] < 0:
                        prediction.update(valid=False, reason='forecast plane already passed')
            row['prediction'] = predictions['11.0']
            row['whole_ball_prediction'] = predictions['11.11']
            row['interception_prediction'] = predictions[str(float(self.planes[2]))]
            center = predictions[str(float(self.planes[3]))]
            early = row['interception_prediction']
            row['forecast_observed_time'] = self.forecast_time
            row['forecast_refreshed'] = bool(refresh)
            row['predicted_floor_events'] = [e for e in self.forecast['floor_contact_events'] if e['velocity_before'][2] < -.1][:8]
            if early['valid'] and center['valid'] and self.geometry is not None:
                reach = reachable_target(center['lateral_m'], row['world_position'][1], self.position.filtered_velocity,
                                         early['time_to_plane_s'], self.position.limits)
                row['reachability'] = reach
                row['station_reachable'] = reach['reachable']
                row['current_body_coverage'] = self.geometry['current']
                row['target_body_coverage'] = self.geometry['target']
                row['decision_body_coverage'] = self.geometry['decision']
                row['coverage_observed_time'] = self.geometry['observed_time']
                row['coverage_decision_lateral_m'] = self.geometry['decision_lateral_m']
                covered_station_still_reachable = reach['interval_m'][0] <= self.geometry['decision_lateral_m'] <= reach['interval_m'][1]
                row['intercept_feasible'] = bool(reach['reachable'] and covered_station_still_reachable and
                                                 self.geometry['decision']['covered'] and center['inside_aperture'])
                if self.geometry['target']['covered'] and center['inside_aperture']:
                    command, row['tracking'] = self.position.command(now, row['world_position'], row['yaw'],
                                                                     row['world_velocity'], reach['decision_lateral_m'])
                    row['planning_reason'] = 'separate center reachability and native modeled-body coverage; conservative partial positioning if unreachable'
                else:
                    row['planning_reason'] = 'no credible modeled-body coverage with existing lateral action'
            else:
                row['planning_reason'] = 'invalid or expired floor-only forecast; stop'
        self.last_command = command
        row['command'] = command
        return command


def run_g_trial(model, predictor_model, name, launch, mode='active', target=0., duration=18., output=None):
    adapter = PredictiveGoalkeeper(model, predictor_model, mode=mode, target=target)
    with patch('k1_goalkeeper.positioning_validation.TruthGoalkeeper', return_value=adapter):
        summary, trace = run_trial(model, name, mode, launch, target=target, duration=duration, output=None)
    summary.update(stage='G', predictor_refresh_count=adapter.refresh_count,
                   prediction_assumptions=['same known simulation floor/ball physics, not measured future',
                                           'forecast excludes robot and goal contact',
                                           'coverage is a frozen current-pose geometry hypothesis, not a new action'])
    if output:
        from .positioning_validation import save_trial
        save_trial(output, summary, trace)
    return summary, trace
