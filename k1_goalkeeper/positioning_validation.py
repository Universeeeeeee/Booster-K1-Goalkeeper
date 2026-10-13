"""Stage F adapters around Stage E's physical contact and trajectory recorder."""
import json
from pathlib import Path

import mujoco
import numpy as np

from .impact_validation import ImpactCase, _robot_body, run_impact
from .positioning import LateralPositionController, ResponseLimits, predict_ball, reachable_target, world_to_goal
from .velocity_validation import body_state, recovery_metrics, trajectory_hash


class TruthGoalkeeper:
    def __init__(self, mode, target=0., control_end=14., limits=ResponseLimits()):
        if mode not in ("position", "active", "passive", "absent"):
            raise ValueError("unknown F control mode")
        self.mode, self.target, self.control_end = mode, target, control_end
        self.position = LateralPositionController(limits)
        self.records = []
        self.last_command = [0., 0., 0.]
        self.last_boundary = -1
        self.unexpected_contacts = []
        self.contact_seen = False
        self.crossings = {}
        self.previous_ball = None
        self.robot_geoms = None

    def command(self, sim):
        if sim.physics_steps == self.last_boundary or sim.physics_steps % 20:
            return self.last_command
        self.last_boundary = sim.physics_steps
        state = body_state(sim.controller)
        q, v = sim.controller.base_qpos_address, sim.controller.base_dof_address
        position, yaw = state["base_position"], state["rpy"][2]
        velocity = sim.data.qvel[v:v+3].copy()
        ball = sim.model.joint("ball_free")
        bq, bv = int(ball.qposadr[0]), int(ball.dofadr[0])
        p, bv6 = sim.data.qpos[bq:bq+3].copy(), sim.data.qvel[bv:bv+6].copy()
        row = {"time": float(sim.data.time), "goal_position": world_to_goal(position).tolist(),
               "world_position": position.tolist(), "yaw": float(yaw), "world_velocity": velocity.tolist(),
               "ball_position": p.tolist(), "ball_velocity": bv6[:3].tolist(), "prediction": None,
               "reachability": None, "tracking": None}
        command = [0., 0., 0.]
        if sim.data.time >= 2.-1e-9:
            row["prediction"] = predict_ball(p, bv6[:3], bv6[3:])
            row["whole_ball_prediction"] = predict_ball(p, bv6[:3], bv6[3:], plane_x=11.11)
            # All-articulation collision bound .693084m + ball radius .11m,
            # rounded outward; the deadline precedes possible body contact.
            interception = predict_ball(p, bv6[:3], bv6[3:], plane_x=float(position[0]-.81))
            row["interception_prediction"] = interception
            target = self.target
            enabled = self.mode == "position" and sim.data.time < self.control_end
            row["planning_reason"] = "position target" if self.mode == "position" else "waiting or stopped"
            prediction_usable = (interception["valid"] and not interception["future_floor_contact_unmodeled"]
                                 and interception["inside_aperture"] is True)
            if self.mode == "active" and not self.contact_seen and prediction_usable:
                reach = reachable_target(interception["lateral_m"], position[1], self.position.filtered_velocity,
                                         interception["time_to_plane_s"], self.position.limits)
                row["reachability"] = reach
                target = reach["decision_lateral_m"]
                row["planning_reason"] = "current-state lateral forecast; center reachability does not certify body coverage"
                # Stop when the ball has passed the robot or when prediction is unavailable.
                enabled = True
            elif self.mode == "active" and not self.contact_seen:
                row["planning_reason"] = "abstain: moving away, outside aperture, or future floor bounce unmodeled"
            if enabled:
                command, row["tracking"] = self.position.command(sim.data.time, position, yaw, velocity, target)
        self.last_command = command
        row["command"] = list(command)
        self.records.append(row)
        return command

    def observe(self, sim):
        model, data = sim.model, sim.data
        if self.robot_geoms is None:
            self.robot_geoms = {g for g in range(model.ngeom) if _robot_body(model, model.geom_bodyid[g])}
        ball_id = model.geom("ball_collision").id
        bq = int(model.joint("ball_free").qposadr[0])
        p, time = data.qpos[bq:bq+3].copy(), float(data.time)
        if time > 2.+1e-9:
            if self.previous_ball is not None:
                old_time, old = self.previous_ball
                for plane in (11., 11.11):
                    if str(plane) not in self.crossings and old[0] <= plane < p[0]:
                        fraction = (plane-old[0])/(p[0]-old[0])
                        self.crossings[str(plane)] = {"time": float(old_time+fraction*(time-old_time)),
                                                      "position": (old+fraction*(p-old)).tolist()}
            self.previous_ball = (time, p)
        for contact_index, contact in enumerate(data.contact[:data.ncon]):
            if contact.efc_address < 0 or contact.dist >= 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if (g1 == ball_id and r2) or (g2 == ball_id and r1):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                if force[0] > 1e-6:
                    self.contact_seen = True
            if r1 != r2:
                other = g2 if r1 else g1
                body = int(model.geom_bodyid[other])
                if body != 0 and other != ball_id and not self.unexpected_contacts:
                    self.unexpected_contacts.append({"time": time, "kind": "robot-goal", "geom_ids": [g1, g2]})


def prediction_errors(records, reference_crossings):
    result = {}
    for plane, key in (("11.0", "prediction"), ("11.11", "whole_ball_prediction")):
        crossing = reference_crossings.get(plane)
        errors = []
        if crossing:
            for row in records:
                prediction = row.get(key)
                if row["time"] > 2.+1e-9 and row["time"] < crossing["time"] and prediction and prediction["valid"]:
                    errors.append({"time": row["time"], "model": prediction["model"],
                                   "unmodeled_bounce": prediction["future_floor_contact_unmodeled"],
                                   "time_error_s": row["time"]+prediction["time_to_plane_s"]-crossing["time"],
                                   "lateral_error_m": prediction["lateral_m"]-crossing["position"][1],
                                   "height_error_m": prediction["height_m"]-crossing["position"][2]})
        result[plane] = {"reference": crossing, "samples": errors,
                         "first": errors[0] if errors else None,
                         "maximum_absolute_errors": {key: max(abs(e[key]) for e in errors) for key in
                             ("time_error_s", "lateral_error_m", "height_error_m")} if errors else None}
    return result


def run_trial(model, name, mode, launch, duration=18., target=0., output=None):
    adapter = TruthGoalkeeper(mode, target, control_end=14.)
    case = ImpactCase(name, robot_mode="absent" if mode == "absent" else "free", duration=duration)
    summary, trace = run_impact(model, case, launch=launch,
                                command_provider=adapter.command if mode != "absent" else None,
                                step_observer=adapter.observe)
    summary.update(stage="F", mode=mode, high_level_records=adapter.records,
                   plane_crossings=adapter.crossings, unexpected_contacts=adapter.unexpected_contacts,
                   coverage_limitations=["shoulder_pitch, hip_pitch and ankle_pitch have visual-only geometry",
                                         "no complete body silhouette; a no-contact miss alone does not identify a controller failure"])
    event = summary["exit_event"]
    summary["right_goal"] = bool(event and event["outcome"] == "Goal" and event["goal"] == "right") if summary["completed"] or event else None
    if mode != "absent":
        t = trace["time"]
        after = t >= 2.
        moving = np.any(np.abs(trace["command"]) > 0, axis=1)
        stop = float(t[np.flatnonzero(moving)[-1]]) if moving.any() else 2.
        velocities = np.column_stack([trace["body_linear_velocity"][:, :2], trace["body_angular_velocity"][:, 2]])
        summary["final_stop_recovery"] = recovery_metrics(trace, velocities, stop, float(t[-1]),
                                                          summary["completed"] and not summary["fall_events"] and not summary["numerical_error"])
        goal_position = np.column_stack([trace["base_position"][:, 1], 11.-trace["base_position"][:, 0], trace["base_position"][:, 2]])
        trace["goal_position"] = goal_position
        summary["maximum_lateral_center_m"] = float(np.abs(goal_position[:, 0]).max())
        summary["world_x_range_m"] = [float(trace["base_position"][:, 0].min()), float(trace["base_position"][:, 0].max())]
        if mode == "position":
            error = goal_position[:, 0]-target
            stable = after & (np.abs(error) <= .10)
            arrival = next((float(t[i]-2.) for i in np.flatnonzero(stable) if np.all(stable[i:]) and t[-1]-t[i] >= 1.), None)
            direction = np.sign(target-goal_position[0, 0])
            summary["positioning"] = {"requested_goal_lateral_m": target,
                "commanded_goal_lateral_m": float(np.clip(target, -adapter.position.limits.safe_lateral_m, adapter.position.limits.safe_lateral_m)),
                "final_error_m": float(error[-1]), "maximum_overshoot_m": float(max(0., np.max(direction*error[after]))) if direction else float(np.abs(error[after]).max()),
                "arrival_and_hold_time_s": arrival, "tolerance_m": .10,
                "final_goal_position": goal_position[-1].tolist(), "stop_command_time": stop,
                "passed": bool(abs(error[-1]) <= .10 and arrival is not None and summary["final_stop_recovery"]["verified"]
                               and summary["balance_maintained"] and summary["maximum_lateral_center_m"] <= adapter.position.limits.safe_lateral_m+.03
                               and summary["world_x_range_m"][0] >= 10.05 and summary["world_x_range_m"][1] <= 10.75
                               and not summary["fall_events"] and not summary["numerical_error"] and not adapter.unexpected_contacts)}
    # Include delivered high-level commands, decisions and current observations in repetition verification.
    trace["high_level_json_utf8"] = np.frombuffer(json.dumps(adapter.records, sort_keys=True, allow_nan=False).encode(), dtype=np.uint8)
    summary["trajectory_sha256"] = trajectory_hash(trace)
    if output:
        save_trial(output, summary, trace)
    return summary, trace


def save_trial(output, summary, trace):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path/"trajectory.npz", **trace)
    (path/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
