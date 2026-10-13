"""H2: native geometry from recorded H1 qpos; FK only, no dynamics or policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_defense_geometry import region


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_hashes(model, fields):
    return {field: hashlib.sha256(np.ascontiguousarray(getattr(model, field)).tobytes()).hexdigest()
            for field in fields}


def shape_bounds(model, data, geom, rotation=None, origin=None):
    """Exact per-axis convex support bounds, not a filled bounding box."""
    center = data.geom_xpos[geom].copy()
    basis = data.geom_xmat[geom].reshape(3, 3)
    if rotation is not None:
        center = rotation.T @ (center - origin)
        basis = rotation.T @ basis
    kind = int(model.geom_type[geom])
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(basis) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[0] * np.sqrt(np.sum(basis[:, :2] ** 2, axis=1)) + size[1] * np.abs(basis[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        vertices = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh])]
        points = vertices @ basis.T + center
        return points.min(axis=0), points.max(axis=0)
    else:
        raise ValueError(f"Unsupported native upper-body shape {kind}")
    return center - extent, center + extent


def sample_geometry(model, data, qpos, geoms):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    trunk = model.body("trunk").id
    rotation, origin = data.xmat[trunk].reshape(3, 3), data.xpos[trunk]
    measured = {"trunk_origin_world_z_m": float(origin[2])}
    for name, ids in geoms.items():
        local = [shape_bounds(model, data, geom, rotation, origin) for geom in ids]
        world = [shape_bounds(model, data, geom) for geom in ids]
        minimum = np.asarray([bounds[0] for bounds in local]).min(axis=0)
        maximum = np.asarray([bounds[1] for bounds in local]).max(axis=0)
        measured.update({f"{name}.world_z_max_m": float(max(bounds[1][2] for bounds in world)),
                         f"{name}.local_y_min_m": float(minimum[1]),
                         f"{name}.local_y_max_m": float(maximum[1]),
                         f"{name}.local_z_min_m": float(minimum[2]),
                         f"{name}.local_z_max_m": float(maximum[2])})
    measured["arms_span_local_y_m"] = (max(measured[f"{side}_arm.local_y_max_m"] for side in ("left", "right"))
                                        - min(measured[f"{side}_arm.local_y_min_m"] for side in ("left", "right")))
    measured["left_arm_outside_trunk_lateral_extent_m"] = measured["left_arm.local_y_max_m"] - measured["trunk.local_y_max_m"]
    measured["right_arm_outside_trunk_lateral_extent_m"] = measured["trunk.local_y_min_m"] - measured["right_arm.local_y_min_m"]
    measured["arms_world_z_max_m"] = max(measured[f"{side}_arm.world_z_max_m"] for side in ("left", "right"))
    return measured


def summarize(rows):
    return {key: {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}
            for key in rows[0] for values in [np.asarray([row[key] for row in rows])]}


def analyze(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    results_path, audit_path = directory / "results.json", directory / "model_audit.json"
    scene_path, protocol_path = directory / "scene_free.xml", directory / "protocol.json"
    results = json.loads(results_path.read_text())
    audit = json.loads(audit_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if results["protocol_sha256"] != digest(protocol_path):
        raise RuntimeError("H1 protocol hash does not match results")
    protected = protocol["original_source_sha256"]
    protected_before = {path: digest(ROOT / path) for path in protected}
    if protected_before != protected:
        raise RuntimeError("Protected original H1 sources changed")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    fields = audit["array_sha256"]
    before_hashes = model_hashes(model, fields)
    if before_hashes != fields or not all(np.array_equal(getattr(model, key), value)
                                         for key, value in audit["arrays"].items()):
        raise RuntimeError("Compiled model differs from H0 model_audit mass/inertia/geometry")
    geoms = {key: [] for key in ("trunk", "left_arm", "right_arm")}
    for geom in range(model.ngeom):
        name = region(model.body(int(model.geom_bodyid[geom])).name)
        if name in geoms and (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            geoms[name].append(geom)
    if [len(geoms[key]) for key in geoms] != [2, 3, 3]:
        raise RuntimeError("Expected unchanged native torso and arm geometry")
    data, baseline_data = mujoco.MjData(model), mujoco.MjData(model)
    baselines = {}
    paths = [results_path, audit_path, scene_path, protocol_path]
    for summary in results["baseline"]:
        path = directory / summary["name"] / "trajectory.npz"
        with np.load(path) as trace:
            baselines[summary["name"]] = {key: trace[key].copy() for key in ("time", "qpos")}
        paths.append(path)
    trials = []
    for summary in results["trials"]:
        name = summary["name"]
        path = directory / name / "trajectory.npz"
        baseline_name = f"zero_original_r{name.rsplit('_r', 1)[1]}"
        baseline = baselines[baseline_name]
        pulse = summary["pulse"]
        hold_end = pulse["start_s"] + pulse["rise_s"] + pulse["hold_s"]
        desired_times = np.arange(hold_end - 1., hold_end - 1e-9, .02)
        paths.append(path)
        with np.load(path) as trace:
            times, positions = trace["time"], trace["qpos"]
            indices = np.rint(desired_times / model.opt.timestep).astype(int)
            available = indices < len(times)
            indices, requested_times = indices[available], desired_times[available]
            if not len(indices):
                trials.append({"name": name, "status": "no_hold_samples", "H1_passed": summary["passed"]})
                continue
            if (positions.shape[1] != model.nq or np.any(indices >= len(baseline["time"]))
                    or not np.allclose(times[indices], requested_times, rtol=0, atol=1e-8)
                    or not np.allclose(baseline["time"][indices], times[indices], rtol=0, atol=1e-8)):
                raise RuntimeError(f"{name}: mismatched H1 qpos/time/baseline")
            actual, reference, deltas = [], [], []
            for index in indices:
                measured = sample_geometry(model, data, positions[index], geoms)
                matched = sample_geometry(model, baseline_data, baseline["qpos"][index], geoms)
                actual.append(measured)
                reference.append(matched)
                deltas.append({key: measured[key] - matched[key] for key in measured})
            trials.append({"name": name, "pulse": pulse, "H1_passed": summary["passed"],
                           "H1_stable": summary["stable"], "H1_recovered": summary["recovered"],
                           "matched_baseline": baseline_name, "source_trace": str(path.relative_to(ROOT)),
                           "hold_window_s": [hold_end - 1., hold_end],
                           "requested_sample_times_s": requested_times.tolist(),
                           "actual_sample_times_s": times[indices].tolist(),
                           "trace_qpos_indices": indices.tolist(), "matched_baseline_indices": indices.tolist(),
                           "sample_count": len(indices), "complete_hold_window": len(indices) == 50,
                           "actual": summarize(actual), "matched_original": summarize(reference),
                           "paired_delta": summarize(deltas),
                           "samples": [{"time_s": float(times[index]), "actual": measured,
                                        "matched_original": matched, "delta": delta}
                                       for index, measured, matched, delta in zip(indices, actual, reference, deltas)]})
    after_hashes = model_hashes(model, fields)
    protected_after = {path: digest(ROOT / path) for path in protected}
    if protected_after != protected_before:
        raise RuntimeError("Protected sources changed during FK analysis")
    report = {"stage": "H2", "mode": "official_native_no_extensions",
              "generation": ".venv/bin/python scripts/analyze_defense_coverage.py",
              "H1_gate_passed": results["gate_passed"], "H1_trial_count": len(results["trials"]),
              "method": {"state_source": "actual full-scene recorded qpos, never requested target angles",
                         "static_calls": ["mj_kinematics", "mj_comPos"], "physics_steps": 0,
                         "policy_inferences": 0, "collision_geometry_added": 0,
                         "sampling": "50 samples every20ms in last1s hold, identical original baseline physical times",
                         "bounds": "exact box/cylinder axial support and compiled convex-mesh vertex extrema",
                         "local_frame": "actual trunk rotation and origin at each recorded sample",
                         "ball_radius_added_to_geometry": False,
                         "scope": "extrema/bounding ranges only; bounds are not continuous coverage, contact certificates, interception probability, or successful saves; H1 contains no ball-robot shot"},
              "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in paths},
              "model_array_sha256_before": before_hashes, "model_array_sha256_after": after_hashes,
              "model_audit_hashes_match": before_hashes == fields,
              "model_parameters_unchanged": before_hashes == after_hashes,
              "protected_original_source_sha256": protected_before,
              "protected_original_sources_unchanged": protected_before == protected_after,
              "native_geoms": {key: [model.geom(geom).name for geom in ids] for key, ids in geoms.items()},
              "trials": trials}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/defense_validation")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/defense_validation/dynamic_coverage.json")
    args = parser.parse_args()
    report = analyze(args.input, args.output)
    print(json.dumps({"trials": len(report["trials"]), "H1_gate_passed": report["H1_gate_passed"],
                      "model_audit_hashes_match": report["model_audit_hashes_match"], "physics_steps": 0}))
