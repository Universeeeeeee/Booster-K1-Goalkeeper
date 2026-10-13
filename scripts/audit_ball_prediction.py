"""Readonly Stage F ball-dynamics audit; forecast functions receive current/past state only."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from k1_goalkeeper.positioning import predict_ball


def ball_floor_model(scene):
    source = ET.parse(scene).getroot()
    root = ET.Element("mujoco", model="causal_ball_floor_audit")
    for name in ("option", "default"):
        node = source.find(name)
        if node is not None:
            root.append(deepcopy(node))
    world = ET.SubElement(root, "worldbody")
    world.append(deepcopy(source.find("worldbody/geom[@name='ground']")))
    ball = deepcopy(source.find("worldbody/body[@name='ball']"))
    for geom in ball.iter("geom"):
        geom.attrib.pop("material", None)
    world.append(ball)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def physical_forecast(model, qpos, qvel, planes=(11., 11.11), horizon=18.):
    """Clone current ball only; no future trajectory, launch metadata or measured fit."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    result = {}
    for _ in range(int(np.ceil(horizon/model.opt.timestep))):
        before, before_time = data.qpos[:3].copy(), float(data.time)
        mujoco.mj_step(model, data)
        p = data.qpos[:3]
        for plane in planes:
            if str(plane) not in result and before[0] <= plane < p[0]:
                fraction = float((plane-before[0])/(p[0]-before[0]))
                result[str(plane)] = {"time_to_plane_s": before_time+fraction*model.opt.timestep,
                                      "position": (before+fraction*(p-before)).tolist()}
        if len(result) == len(planes):
            break
    return result


def history_forecast(p, v, old_v, history_dt, plane=11.):
    """Constant horizontal acceleration estimated solely from earlier observation."""
    if p[2] > .13 or abs(v[2]) >= .3 or v[0] <= .05 or p[0] >= plane:
        return None
    acceleration = (v[:2]-old_v[:2])/history_dt
    distance = plane-p[0]
    ax = acceleration[0]
    if abs(ax) < 1e-9:
        arrival = distance/v[0]
    else:
        discriminant = v[0]**2+2*ax*distance
        if discriminant < 0:
            return None
        arrival = 2*distance/(v[0]+np.sqrt(discriminant))
    return {"time_to_plane_s": float(arrival),
            "position": [plane, float(p[1]+v[1]*arrival+.5*acceleration[1]*arrival**2), .11]}


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "minimum": float(values.min()), "maximum": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std())} if len(values) else None


def error(prediction, current_time, reference):
    return [current_time+prediction["time_to_plane_s"]-reference["time"],
            prediction["position"][1]-reference["position"][1],
            prediction["position"][2]-reference["position"][2]]


def errors_summary(values):
    values = np.asarray(values)
    return {"count": len(values), "maximum_absolute": np.abs(values).max(axis=0).tolist(),
            "rmse": np.sqrt(np.mean(values**2, axis=0)).tolist(),
            "columns": ["time_s", "lateral_m", "height_m"]} if len(values) else None


def world_spin(qpos, angular_velocity):
    result = np.zeros(3)
    mujoco.mju_rotVecQuat(result, angular_velocity, qpos[3:7])
    return result


def run(source, output):
    scene = source/"scene_absent.xml"
    model = ball_floor_model(scene)
    full_model = mujoco.MjModel.from_xml_path(str(scene))
    ball_geom = model.geom("ball_collision").id
    ground_geom = model.geom("ground").id
    body = model.body("ball").id
    provenance_files = [scene, *sorted(source.glob("*_absent/trajectory.npz")), *sorted(source.glob("*_absent/summary.json"))]
    before_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_files}
    report = {"purpose": "Stage G model selection evidence; F cases are development data, not independent holdouts",
              "causality": "Each forecast receives only current qpos/qvel or current and past velocities; future reference is read exclusively by evaluator after forecast",
              "limitations": ["Floor-only forecast excludes goal/robot contact; evaluation must stop before such physical contact",
                              "qpos quaternion is required because freejoint angular velocity is in the body frame",
                              "Warmstart is reset; measured error includes this approximation",
                              "Known simulated contact parameters do not certify real-ball fidelity"],
              "source_sha256": before_hashes,
              "parameters": {"mujoco_version": mujoco.__version__, "timestep": float(model.opt.timestep),
                  "gravity": model.opt.gravity.tolist(), "ball_mass": float(model.body_mass[body]),
                  "ball_inertia": model.body_inertia[body].tolist(), "ball_friction": model.geom_friction[ball_geom].tolist(),
                  "ball_solref": model.geom_solref[ball_geom].tolist(), "ball_solimp": model.geom_solimp[ball_geom].tolist(),
                  "ball_priority": int(model.geom_priority[ball_geom]), "ball_condim": int(model.geom_condim[ball_geom]),
                  "ground_friction": model.geom_friction[ground_geom].tolist(), "ground_condim": int(model.geom_condim[ground_geom])},
              "cases": []}
    aggregate = {name: [] for name in ("physical", "naive", "history")}
    runtime, full_runtime = [], []
    for path in sorted(source.glob("*_absent")):
        if not (path/"trajectory.npz").exists():
            continue
        trace = np.load(path/"trajectory.npz")
        summary = json.loads((path/"summary.json").read_text())
        crossing = summary["plane_crossings"].get("11.0")
        if not crossing:
            continue
        t, p, v = trace["time"], trace["ball_position"], trace["ball_velocity"]
        launch_time = summary["launch"]["time"]
        samples = np.flatnonzero((t >= launch_time+.02-1e-9)&(t < crossing["time"]))[::500]
        case = {"name": path.name, "reference": crossing, "samples": [], "bounce_episodes": [], "rolling_acceleration": {}}
        case_errors = {name: [] for name in aggregate}
        for index in samples:
            started = time.perf_counter()
            physical = physical_forecast(model, trace["qpos"][index], trace["qvel"][index])
            elapsed = time.perf_counter()-started
            runtime.append(elapsed)
            row = {"time": float(t[index]), "physical_wall_s": elapsed}
            physical11 = physical.get("11.0")
            if physical11:
                row["physical_error"] = error(physical11, t[index], crossing)
                case_errors["physical"].append(row["physical_error"])
            naive = predict_ball(p[index], v[index], trace["ball_angular_velocity_body"][index])
            if naive["valid"]:
                n = {"time_to_plane_s": naive["time_to_plane_s"],
                     "position": [11., naive["lateral_m"], naive["height_m"]]}
                row["naive_error"] = error(n, t[index], crossing)
                row["naive_unmodeled_floor_contact"] = naive["future_floor_contact_unmodeled"]
                case_errors["naive"].append(row["naive_error"])
            # A real history window exists only after 0.5 s from launch.
            old_index = index-500
            if t[old_index] >= launch_time-1e-9:
                historical = history_forecast(p[index], v[index], v[old_index], t[index]-t[old_index])
                if historical:
                    row["history_error"] = error(historical, t[index], crossing)
                    case_errors["history"].append(row["history_error"])
            case["samples"].append(row)
        for contact in summary["contacts"]:
            before = np.asarray(contact["ball_velocity_before"])
            after = np.asarray(contact["ball_velocity_after"])
            if contact["kind"] != "ground" or before[2] >= -.1 or contact["start_time"] >= crossing["time"]:
                continue
            i = max(0, int(np.searchsorted(t, contact["start_time"])))
            j = min(len(t)-1, int(np.searchsorted(t, contact["end_time"])))
            w0 = world_spin(trace["qpos"][i], trace["ball_angular_velocity_body"][i])
            w1 = world_spin(trace["qpos"][j], trace["ball_angular_velocity_body"][j])
            slip0 = before+np.cross(w0, [0., 0., -.11])
            slip1 = after+np.cross(w1, [0., 0., -.11])
            case["bounce_episodes"].append({"start_time": contact["start_time"], "duration_s": contact["duration_s"],
                "effective_restitution": float(after[2]/-before[2]), "velocity_before": before.tolist(),
                "velocity_after": after.tolist(), "world_spin_before": w0.tolist(), "world_spin_after": w1.tolist(),
                "horizontal_contact_slip_before": slip0[:2].tolist(), "horizontal_contact_slip_after": slip1[:2].tolist()})
        rolling_acc = []
        for i in np.flatnonzero((t>=launch_time+.65)&(t<crossing["time"]))[::500]:
            old = i-500
            window = slice(old, i+1)
            if np.all(p[window, 2] < .13) and np.all(np.abs(v[window, 2]) < .05):
                rolling_acc.append((v[i, :2]-v[old, :2])/(t[i]-t[old]))
        for axis, column in (("x_m_s2",0),("y_m_s2",1)):
            case["rolling_acceleration"][axis] = stats([a[column] for a in rolling_acc])
        for name, values in case_errors.items():
            case[name+"_errors"] = errors_summary(values)
            aggregate[name].extend(values)
        # Independent full-scene versus reduced-model check at same current state.
        index = samples[0]
        started = time.perf_counter()
        full_forecast = physical_forecast(full_model, trace["qpos"][index], trace["qvel"][index])
        full_runtime.append(time.perf_counter()-started)
        case["full_scene_first_forecast_error"] = error(full_forecast["11.0"], t[index], crossing)
        report["cases"].append(case)
        print(path.name, case["physical_errors"], flush=True)
    report["aggregate_errors"] = {name: errors_summary(values) for name, values in aggregate.items()}
    report["forecast_wall_seconds"] = {"reduced_model": stats(runtime), "full_absent_model_first_forecasts": stats(full_runtime)}
    report["source_unchanged"] = before_hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_files}
    report["recommendation"] = "Choose fixed-parameter MuJoCo floor+ball current-state clone for bounce and rolling; retain cheap ballistic/linear baseline as benchmark. Evaluate distinct held-out launches before controller integration."
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"reports/positioning_validation")
    parser.add_argument("--output", type=Path, default=ROOT/"reports/prediction_validation/ball_model_audit.json")
    args = parser.parse_args()
    run(args.source, args.output)
