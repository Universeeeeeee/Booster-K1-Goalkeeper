"""Truth-state, lateral-only goalkeeper baseline above the official walking policy."""
from dataclasses import dataclass
import math

import numpy as np


def world_to_goal(position):
    x, y, z = position
    return np.array([y, 11. - x, z])


def predict_ball(position, velocity, angular_velocity=(0., 0., 0.), plane_x=11., radius=.11):
    """Current-state extrapolation; explicitly abstain on future floor bounces."""
    p, v, w = map(lambda value: np.asarray(value, dtype=float), (position, velocity, angular_velocity))
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (p, v, w)) or not np.isfinite(plane_x):
        raise ValueError("prediction requires finite three-component states and plane")
    if v[0] <= .05 or p[0] >= plane_x:
        return {"valid": False, "reason": "not approaching plane", "plane_x": float(plane_x)}
    t = float((plane_x-p[0])/v[0])
    rolling = bool(p[2] <= radius+.02 and abs(v[2]) < .3)
    y = float(p[1]+v[1]*t)
    z = radius if rolling else float(p[2]+v[2]*t-4.905*t*t)
    ground_time = float((v[2]+math.sqrt(max(0., v[2]**2+19.62*max(0., p[2]-radius))))/9.81)
    bounce = not rolling and ground_time < t
    return {"valid": True, "plane_x": float(plane_x), "time_to_plane_s": t,
            "lateral_m": y, "height_m": z, "model": "constant_speed_ground" if rolling else "gravity_ballistic",
            "future_floor_contact_unmodeled": bounce,
            "inside_aperture": None if bounce else bool(abs(y)+radius <= 1.5 and radius-.002 <= z <= 2.-radius),
            "spin_norm_rad_s": float(np.linalg.norm(w)),
            "limitations": "rolling deceleration, spin forces and bounce restitution are not predicted"}


@dataclass(frozen=True)
class ResponseLimits:
    speed_m_s: float = .10
    startup_s: float = 1.0
    stop_s: float = 1.0
    stop_distance_m: float = .09
    safe_lateral_m: float = .65


def reachable_target(target, lateral, velocity, time_left, limits=ResponseLimits()):
    values = [target, lateral, velocity, time_left]
    if not np.isfinite(values).all() or time_left < 0:
        raise ValueError("reachability requires finite values and nonnegative time")
    # No credit for gait-phase instantaneous velocity; reserve braking drift and restart.
    reserve = max(limits.stop_distance_m, abs(velocity)*limits.stop_s)
    travel = max(0., limits.speed_m_s*max(0., time_left-limits.startup_s-limits.stop_s)-reserve)
    low = max(-limits.safe_lateral_m, lateral-travel)
    high = min(limits.safe_lateral_m, lateral+travel)
    if low > high:  # Already outside safe range: stop and expose violation.
        decision = float(np.clip(lateral, -limits.safe_lateral_m, limits.safe_lateral_m))
        return {"reachable": False, "target_lateral_m": float(target), "decision_lateral_m": decision,
                "interval_m": [decision, decision], "reason": "outside safe range", "travel_budget_m": 0.}
    braking_possible = time_left >= limits.stop_s or abs(velocity) <= .05
    reachable = (low-.10 <= target <= high+.10 and abs(target) <= limits.safe_lateral_m
                 and braking_possible)
    return {"reachable": bool(reachable), "target_lateral_m": float(target),
            "decision_lateral_m": float(np.clip(target, low, high)), "interval_m": [float(low), float(high)],
            "travel_budget_m": float(travel), "braking_reserve_m": float(reserve),
            "position_tolerance_m": .10, "braking_time_available": bool(braking_possible),
            "reason": "empirical center-position estimate within 0.10m tolerance" if reachable else "insufficient time, braking time or post clearance"}


class LateralPositionController:
    """Quantized lateral motion, predictive stop and hysteresis; no policy resets."""
    def __init__(self, limits=ResponseLimits()):
        self.limits = limits
        self.reset()

    def reset(self):
        self.direction = 0
        self.last_switch = -math.inf
        self.filtered_velocity = 0.
        self.last_time = None
        self.turning = False

    def command(self, time, position, yaw, world_velocity, target):
        if not np.isfinite([time, *position, yaw, *world_velocity, target]).all():
            raise ValueError("position control requires finite state")
        if self.last_time is not None and time < self.last_time:
            raise ValueError("reset controller before rewinding time")
        dt = .02 if self.last_time is None else time-self.last_time
        self.last_time = time
        alpha = 1.-math.exp(-dt/.25)
        self.filtered_velocity += alpha*(world_velocity[1]-self.filtered_velocity)
        requested = float(target)
        target = float(np.clip(target, -self.limits.safe_lateral_m, self.limits.safe_lateral_m))
        error = target-position[1]
        predicted_stop = position[1]+self.filtered_velocity*.4
        yaw_error = math.atan2(math.sin(math.pi-yaw), math.cos(math.pi-yaw))
        safe = abs(position[1]) <= self.limits.safe_lateral_m+.03 and 10.05 <= position[0] <= 10.75
        self.turning = abs(yaw_error) > (.035 if self.turning else .12)
        if not safe or self.turning:
            self.direction = 0
            self.last_switch = time
        elif self.direction:
            if self.direction*(target-predicted_stop) <= .035 and time-self.last_switch >= .4:
                self.direction = 0
                self.last_switch = time
        elif abs(error) > .10 and time-self.last_switch >= 1.2:
            self.direction = 1 if error > 0 else -1
            self.last_switch = time
        # Rotate the desired world lateral velocity into current body coordinates.
        world_command = self.direction*.2
        body_y = math.cos(yaw)*world_command
        command = [0., float(np.sign(body_y)*.2) if self.direction else 0.,
                   float(np.sign(yaw_error)*.2) if self.turning and safe else 0.]
        return command, {"requested_target_m": requested, "target_m": target, "error_m": float(error),
                         "predicted_stop_m": float(predicted_stop), "filtered_world_vy": float(self.filtered_velocity),
                         "safe": bool(safe), "yaw_error_rad": float(yaw_error), "direction": self.direction,
                         "body_projection_uncommanded_vx": float(math.sin(yaw)*world_command)}
