"""H1 measurements through the unchanged E recorder and scene physics loop."""
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from .defense_controller import PostFilteredArmController
from .impact_validation import ImpactCase, _robot_body, run_impact
from .velocity_validation import recovery_metrics, trajectory_hash


def contact_state(model, data):
    """Only solver-active contacts carrying actual positive normal force."""
    self_pairs, ground_pairs, external_pairs = set(), set(), set()
    for index, contact in enumerate(data.contact[:data.ncon]):
        if contact.efc_address < 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        if force[0] <= 1e-6:
            continue
        first, second = int(contact.geom1), int(contact.geom2)
        bodies = [int(model.geom_bodyid[g]) for g in (first, second)]
        robot = [_robot_body(model, b) for b in bodies]
        pair = tuple(sorted((first, second)))
        if all(robot):
            self_pairs.add(pair)
        elif any(robot) and 0 in bodies:
            body = bodies[0] if robot[0] else bodies[1]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if name not in ("left_ankle_roll_link", "right_ankle_roll_link"):
                ground_pairs.add(pair)
        elif any(robot):
            external_pairs.add(pair)
    return {"self_contact_pairs": self_pairs, "nonfoot_ground_pairs": ground_pairs,
            "external_robot_pairs": external_pairs}


def _pairs_by_step(trace, key, dt):
    result = {}
    if trace is not None:
        for row in trace.get(key, np.empty((0, 3))):
            result.setdefault(round(float(row[0])/dt), set()).add(tuple(map(int, row[1:])))
    return result


def run_defense_trial(model, pulse, name, seed=0, output=None, baseline=None, guards=True):
    """A pulse is an arm intervention; pulse=None preserves original E control."""
    rows = {key: [] for key in ("h_time", "nominal_targets", "delivered_targets", "arm_offset",
                                "actual_joint_pos", "actual_joint_vel", "h_base_height", "h_tilt_degrees",
                                "h_raw_torque", "h_motor_controls", "h_applied_torque")}
    policy = {key: [] for key in ("policy_time", "last_action", "obs_history", "filtered_targets")}
    contacts = {key: [] for key in ("self_contacts", "nonfoot_ground_contacts", "external_robot_contacts")}
    context = {"wrapper": None, "installed": False, "stop": None, "policy_count": -1,
               "saturation_duration": np.zeros(8)}
    baseline_self = _pairs_by_step(baseline, "self_contacts", model.opt.timestep)
    baseline_ground = _pairs_by_step(baseline, "nonfoot_ground_contacts", model.opt.timestep)
    start = pulse.start_s if pulse is not None else 3.
    end = (pulse.start_s + pulse.rise_s + pulse.hold_s + pulse.return_s + 5.) if pulse else 14.

    def command(sim):
        if not context["installed"]:
            if pulse is not None:
                context["wrapper"] = PostFilteredArmController(sim.controller, pulse)

                def checked_targets():
                    wrapper = context["wrapper"]
                    targets = wrapper.compute_targets()
                    selected = np.flatnonzero(wrapper.offset.numpy() != 0)
                    if guards and len(selected):
                        ranges = model.jnt_range[sim.controller.joint_ids[selected]]
                        candidate = targets.numpy()[selected]
                        if np.any(candidate < ranges[:, 0]+.05) or np.any(candidate > ranges[:, 1]-.05):
                            context["stop"] = {"kind": "invalid", "reason": "selected target margin rejected before PD",
                                               "time": float(sim.data.time), "before_delivery": True,
                                               "nominal_targets": wrapper.nominal_targets.numpy().tolist(),
                                               "rejected_targets": targets.numpy().tolist(), "offset": wrapper.offset.numpy().tolist()}
                            sim.controller.stop()
                            raise RuntimeError("H1 guard: selected target margin rejected before PD")
                    return targets

                sim.controller.compute_targets = checked_targets
            context["installed"] = True
        context["pre_q"] = sim.data.qpos.astype(np.float32)[sim.controller.qpos_addresses].copy()
        context["pre_dq"] = sim.data.qvel.astype(np.float32)[sim.controller.dof_addresses].copy()
        return (0., 0., 0.)

    def observe(sim):
        c, data, t = sim.controller, sim.data, float(sim.data.time)
        wrapper = context["wrapper"]
        delivered = sim.targets.detach().cpu().numpy().copy()
        nominal = wrapper.nominal_targets.cpu().numpy().copy() if wrapper else delivered.copy()
        offset = wrapper.offset.cpu().numpy().copy() if wrapper else np.zeros(22, dtype=np.float32)
        quat = data.qpos[c.base_qpos_address+3:c.base_qpos_address+7]
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2*(quat[1]**2+quat[2]**2), -1, 1))))
        height = float(data.qpos[c.base_qpos_address+2])
        raw = c.robot.joint_stiffness.numpy()*(delivered-context["pre_q"])-c.robot.joint_damping.numpy()*context["pre_dq"]
        controls = data.ctrl[c.actuator_ids].copy()
        applied = data.qfrc_actuator[c.dof_addresses].copy()
        values = (t, nominal, delivered, offset, data.qpos[c.qpos_addresses].copy(),
                  data.qvel[c.dof_addresses].copy(), height, tilt, raw.copy(), controls, applied)
        for key, value in zip(rows, values):
            rows[key].append(value)
        if sim.policy_steps != context["policy_count"]:
            context["policy_count"] = sim.policy_steps
            for key, value in zip(policy, (t-model.opt.timestep, c.policy.last_action.numpy().copy(),
                                           c.policy.obs_history.numpy().copy(), c.policy.filtered_dof_target.numpy().copy())):
                policy[key].append(value)
        state = contact_state(model, data)
        for key, state_key in (("self_contacts", "self_contact_pairs"), ("nonfoot_ground_contacts", "nonfoot_ground_pairs"),
                               ("external_robot_contacts", "external_robot_pairs")):
            for pair in sorted(state[state_key]):
                contacts[key].append([t, *pair])
        if not guards or t < start - 1e-9:
            return
        reason = None
        limits = c.robot.effort_limit.numpy()
        context["saturation_duration"] = np.where(np.abs(raw[2:10]) >= limits[2:10]-1e-6,
                                                    context["saturation_duration"]+model.opt.timestep, 0.)
        if not all(np.isfinite(v).all() for v in (delivered, nominal, offset, raw, applied, data.qpos, data.qvel, data.qacc, data.ctrl)):
            reason = "nonfinite state or control"
        elif height < .42 or tilt > 15:
            reason = "balance guard: z<0.42 m or tilt>15 deg"
        elif state["self_contact_pairs"] - baseline_self.get(round(t/model.opt.timestep), set()):
            reason = "new active self-contact"
        elif state["nonfoot_ground_pairs"] - baseline_ground.get(round(t/model.opt.timestep), set()):
            reason = "new active non-foot ground contact"
        elif state["external_robot_pairs"]:
            reason = "unexpected active robot-ball/goal contact"
        elif np.any(context["saturation_duration"] > .10+1e-12):
            reason = "continuous arm raw demand saturation exceeds 0.10 s"
        elif np.any(np.abs(controls) > limits+1e-5) or np.any(np.abs(applied) > limits+1e-5):
            reason = "delivered control/torque exceeds official effort limit"
        elif not np.array_equal(delivered[offset == 0], nominal[offset == 0]):
            reason = "unselected joint target changed"
        selected = np.flatnonzero(offset != 0)
        if reason is None and len(selected):
            ranges = model.jnt_range[c.joint_ids[selected]]
            if np.any(delivered[selected] < ranges[:, 0]+.05) or np.any(delivered[selected] > ranges[:, 1]-.05):
                reason = "selected nonzero-residual target violates XML 0.05 rad margin"
            elif np.any(data.qpos[c.qpos_addresses[selected]] < ranges[:, 0]) or np.any(data.qpos[c.qpos_addresses[selected]] > ranges[:, 1]):
                reason = "selected actual joint outside XML range"
        if reason:
            context["stop"] = {"kind": "instability" if reason.startswith("balance") or "contact" in reason else "invalid",
                               "reason": reason, "time": t, "height_m": height, "tilt_degrees": tilt,
                               "self_contact_pairs": sorted(state["self_contact_pairs"]),
                               "nonfoot_ground_pairs": sorted(state["nonfoot_ground_pairs"]),
                               "external_robot_pairs": sorted(state["external_robot_pairs"])}
            c.stop()
            raise RuntimeError("H1 guard: " + reason)

    # The E recorder launches a zero-speed ball far from K1 at t=2; it never
    # contacts the robot. Launch/reset/physics ownership remain original E code.
    launch = {"position": [7., 0., .111], "velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.]}
    e_summary, e_trace = run_impact(model, ImpactCase(name, duration=end-2, seed=seed),
                                    launch=launch, command_provider=command, step_observer=observe)
    trace = {**e_trace, **{key: np.asarray(value) for key, value in rows.items()},
             **{key: np.asarray(value) for key, value in policy.items()},
             **{key: np.asarray(value, dtype=float).reshape(-1, 3) for key, value in contacts.items()}}
    response = []
    if pulse is not None and baseline is not None:
        lo = pulse.start_s+pulse.rise_s+pulse.hold_s-1.
        hi = pulse.start_s+pulse.rise_s+pulse.hold_s
        selected = (trace["h_time"] >= lo-1e-9) & (trace["h_time"] < hi-1e-9)
        for joint, amplitude in pulse.joint_offsets:
            if amplitude == 0:
                continue
            index = e_summary["joint_names"].index(joint)
            times = trace["h_time"][selected]
            score = None
            if len(times) == 1000 and len(baseline["h_time"]) and baseline["h_time"][-1] >= hi-1e-9:
                matched = np.interp(times, baseline["h_time"], baseline["actual_joint_pos"][:, index])
                score = float(np.median(np.sign(amplitude)*(trace["actual_joint_pos"][selected, index]-matched))/abs(amplitude))
            response.append({"joint": joint, "amplitude_rad": amplitude, "median_signed_response_fraction": score,
                             "passed": score is not None and score >= .5})
    stable = context["stop"] is None and e_summary["completed"] and not e_summary["fall_events"] and e_summary["numerical_error"] is None and e_summary["policy_stop"] is None
    recovery_start = pulse.start_s+pulse.rise_s+pulse.hold_s+pulse.return_s if pulse else 9.
    velocity = np.column_stack([e_trace["body_linear_velocity"][:, :2], e_trace["body_angular_velocity"][:, 2]])
    recovery = recovery_metrics(e_trace, velocity, recovery_start, end, stable)
    recovered = bool(recovery["verified"])
    response_passed = bool(response and all(row["passed"] for row in response)) if pulse is not None and any(v for _, v in pulse.joint_offsets) else True
    numerical_guard = context["stop"] is not None and context["stop"]["reason"] == "nonfinite state or control"
    deliberate_guard = context["stop"] is not None and not numerical_guard
    summary = {"name": name, "seed": seed, "pulse": asdict(pulse) if pulse else None,
               "recorder": "unchanged E run_impact with per-instance command/observer hooks", "e_summary": e_summary,
               "stop": context["stop"], "stable": bool(stable), "authority_passed": response_passed,
               "recovery": recovery, "recovery_start_s": recovery_start, "recovered": recovered,
               "passed": bool(stable and response_passed and recovered), "response": response,
               "failure_class": "numerical" if numerical_guard else "instability_or_invalid" if not stable else "authority" if not response_passed else "recovery" if not recovered else None,
               "censored_by_deliberate_guard": deliberate_guard,
               "numerical_error": None if deliberate_guard else e_summary["numerical_error"],
               "recorder_guard_diagnostic": e_summary["numerical_error"] if deliberate_guard else None,
               "last_delivered_step_time": float(trace["h_time"][-1]) if len(trace["h_time"]) else None,
               "minimum_height_m": float(trace["h_base_height"].min()) if len(trace["h_time"]) else None,
               "maximum_tilt_degrees": float(trace["h_tilt_degrees"].max()) if len(trace["h_time"]) else None,
               "post_start_minimum_height_m": float(trace["h_base_height"][trace["h_time"] >= start-1e-9].min()) if np.any(trace["h_time"] >= start-1e-9) else None,
               "post_start_maximum_tilt_degrees": float(trace["h_tilt_degrees"][trace["h_time"] >= start-1e-9].max()) if np.any(trace["h_time"] >= start-1e-9) else None,
               "trajectory_sha256": trajectory_hash(trace),
               "baseline_contact_comparison": "new force-bearing pairs at each identical physical step versus matched zero baseline"}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/"trajectory.npz", **trace)
        (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary, trace
