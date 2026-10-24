#!/usr/bin/env python3
"""Compare frozen G1 MuJoCo and PhysX recordings offline; never run either engine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT / "results/g1/physics_reference_20261023T173052_077419Z"
CASES = ("free_fall", "ground_bounce", "center_low", "post", "crossbar")
# PhysX float32 state delivery: spin near 72.7 rad/s rounds by about 2e-6.
INITIAL_ATOL = {"position": 1e-6, "velocity": 1e-6, "angular_velocity_body": 1e-5}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def trajectory_arrays(record):
    samples = record["trajectory"]
    time = np.asarray([sample["time"] for sample in samples], dtype=float)
    fields = {key: np.asarray([sample[key] for sample in samples], dtype=float)
              for key in ("position", "velocity", "angular_velocity_body")}
    if len(time) < 2 or not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
        raise ValueError("trajectory needs at least two finite, strictly increasing times")
    for key, array in fields.items():
        if array.shape != (len(time), 3) or not np.isfinite(array).all():
            raise ValueError(f"invalid trajectory {key}")
    return time, fields


def validate_initial(record, shot):
    """Check requested and delivered initial state independently of summaries."""
    if record["shot"] != shot:
        raise ValueError(f"shot protocol mismatch: {shot['name']}")
    times, fields = trajectory_arrays(record)
    if abs(times[0]) > 1e-12 or abs(record["initial_state"]["time"]) > 1e-12:
        raise ValueError("initial time must be zero")
    errors = {}
    for key, shot_key in (("position", "position"), ("velocity", "velocity"),
                          ("angular_velocity_body", "angular_velocity")):
        expected = np.asarray(shot[shot_key], dtype=float)
        initial = np.asarray(record["initial_state"][key], dtype=float)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError(f"invalid initial {key}")
        residual = max(float(np.max(np.abs(initial - expected))),
                       float(np.max(np.abs(fields[key][0] - expected))))
        if residual > INITIAL_ATOL[key]:
            raise ValueError(f"initial {key} mismatch: {residual} > {INITIAL_ATOL[key]}")
        errors[key] = residual
    return {"passed": True, "absolute_tolerance_by_field": INITIAL_ATOL,
            "max_component_residuals": errors}


def aligned_errors(source, target):
    """Linear interpolation on both original sample grids, without extrapolation."""
    st, sf = trajectory_arrays(source)
    tt, tf = trajectory_arrays(target)
    start, end = max(st[0], tt[0]), min(st[-1], tt[-1])
    if end <= start:
        raise ValueError("trajectories have no positive time overlap")
    grid = np.unique(np.concatenate((st[(st >= start) & (st <= end)],
                                     tt[(tt >= start) & (tt <= end)])))
    result = {"method": "linear interpolation on union of original timestamps within overlap; no extrapolation",
              "overlap_start_s": float(start), "overlap_end_s": float(end),
              "aligned_samples": len(grid), "source_end_s": float(st[-1]),
              "target_end_s": float(tt[-1])}
    for key, units in (("position", "m"), ("velocity", "m_s")):
        a = np.column_stack([np.interp(grid, st, sf[key][:, axis]) for axis in range(3)])
        b = np.column_stack([np.interp(grid, tt, tf[key][:, axis]) for axis in range(3)])
        norms = np.linalg.norm(b - a, axis=1)
        # Time weighting avoids giving one engine's denser grid greater weight.
        squared_integral = np.sum(np.diff(grid) * (norms[:-1] ** 2 + norms[1:] ** 2) / 2)
        result[key] = {f"time_weighted_rms_{units}": float(np.sqrt(squared_integral / (end - start))),
                       f"max_{units}": float(norms.max()), f"at_overlap_end_{units}": float(norms[-1])}
    return result


def energies(record, protocol):
    _, fields = trajectory_arrays(record)
    mass = float(protocol["ball_mass_kg"])
    inertia = np.asarray(protocol["ball_inertia_kg_m2"], dtype=float)
    # Body-frame angular velocity and principal inertia are compatible here. G1 ball is isotropic.
    if inertia.shape != (3,) or not np.allclose(inertia, inertia[0], rtol=1e-10, atol=1e-12):
        raise ValueError("energy comparison currently requires the frozen isotropic G1 ball inertia")
    kinetic = (0.5 * mass * np.sum(fields["velocity"] ** 2, axis=1)
               + 0.5 * np.sum(inertia * fields["angular_velocity_body"] ** 2, axis=1))
    potential = -mass * (fields["position"] @ np.asarray(protocol["gravity_m_s2"]))
    return kinetic + potential


def observations(record, protocol):
    times, fields = trajectory_arrays(record)
    energy = energies(record, protocol)
    onsets = sorted(record["contacts"], key=lambda event: event["time"])
    first = {}
    for event in onsets:
        first.setdefault(event["geom"], {"time_s": event["time"], "kind": event["kind"],
                                         "observation": event.get("onset_observation", "source contact-pair onset")})
    grounds = [event for event in onsets if event["geom"] == "ground"]
    rebounds = []
    for index, event in enumerate(grounds):
        if event["ball_velocity_before"][2] >= 0:
            continue
        end = grounds[index + 1]["time"] if index + 1 < len(grounds) else float("inf")
        indices = np.flatnonzero((times > event["time"]) & (times < end))
        upward = indices[fields["velocity"][indices, 2] > 0.05]
        if not len(upward):
            continue
        first_up = upward[0]
        # The apex is a sampled maximum after reversal, bounded by next observed ground onset.
        flight = indices[indices >= first_up]
        apex = flight[np.argmax(fields["position"][flight, 2])]
        pre = max(0, int(np.searchsorted(times, event["time"], side="right")) - 1)
        rebounds.append({"ground_contact_onset_s": event["time"],
                         "first_upward_sample_s": float(times[first_up]),
                         "sampled_apex_time_s": float(times[apex]),
                         "sampled_apex_center_height_m": float(fields["position"][apex, 2]),
                         "sampled_apex_clearance_m": float(fields["position"][apex, 2] - protocol["ball_radius_m"]),
                         "apex_bracketed_by_descent": bool(np.any(fields["velocity"][flight, 2] < 0)),
                         "energy_before_onset_sample_j": float(energy[pre]),
                         "energy_at_apex_sample_j": float(energy[apex]),
                         "energy_change_to_apex_j": float(energy[apex] - energy[pre]),
                         "energy_fraction_retained_to_apex": float(energy[apex] / energy[pre]) if energy[pre] else None})
    return {"outcome": record["outcome"], "exit_event": record["exit_event"],
            "first_contact_by_geom": first, "contact_onset_count": len(onsets),
            "ground_rebounds": rebounds, "sample_time_resolution_s": record["timestep"],
            "contact_duration_s": None,
            "contact_duration_note": "Source stores onsets only; neither exact contact duration nor force history is observable.",
            "energy": {"definition": "0.5*m*|v|^2 + 0.5*sum(I*w_body^2) - m*g dot position; datum z=0",
                       "initial_j": float(energy[0]), "final_j": float(energy[-1]),
                       "change_j": float(energy[-1] - energy[0])},
            "minimum_center_z_m": float(fields["position"][:, 2].min())}


def validate_free_fall(record, protocol):
    """Reject mislabeled clocks using this short, collision-free ballistic case."""
    times, fields = trajectory_arrays(record)
    initial = fields["position"][0]
    velocity = fields["velocity"][0]
    gravity = np.asarray(protocol["gravity_m_s2"], dtype=float)
    dt = protocol["timestep_s"]
    expected_velocity = velocity + times[:, None] * gravity
    # Frozen MuJoCo and PhysX use semi-implicit Euler for this case.
    expected_position = initial + times[:, None] * velocity + 0.5 * (times * (times + dt))[:, None] * gravity
    position_error = float(np.max(np.abs(fields["position"] - expected_position)))
    velocity_error = float(np.max(np.abs(fields["velocity"] - expected_velocity)))
    tolerance = 1e-4  # Accumulated float32 delivery error over the 0.1 s case.
    if record["contacts"] or position_error > tolerance or velocity_error > tolerance:
        raise ValueError("invalid free_fall clock/physics evidence: contact-free 0.1 s ballistic sanity failed "
                         f"(position error={position_error}, velocity error={velocity_error})")
    # Isaac callbacks deliver dt as float32; 100 * float32(.001) differs by 4.75 ns.
    duration_tolerance = max(1e-9, 2 * np.finfo(np.float32).eps * record["shot"]["duration"])
    if abs(times[-1] - record["shot"]["duration"]) > duration_tolerance:
        raise ValueError("invalid free_fall clock/physics evidence: incomplete observation duration")
    return {"passed": True, "integration": "semi-implicit Euler, gravity from protocol",
            "max_component_position_error_m": position_error,
            "max_component_velocity_error_m_s": velocity_error,
            "absolute_tolerance": tolerance, "duration_tolerance_s": float(duration_tolerance)}


def compare_records(source, target, source_protocol, target_protocol, shot):
    for record, protocol in ((source, source_protocol), (target, target_protocol)):
        if record["timestep"] != protocol["timestep_s"]:
            raise ValueError("record timestep differs from its protocol")
    initial = {"source": validate_initial(source, shot), "target": validate_initial(target, shot)}
    sanity = ({"source": validate_free_fall(source, source_protocol),
               "target": validate_free_fall(target, target_protocol)} if shot["name"] == "free_fall" else None)
    so, to = observations(source, source_protocol), observations(target, target_protocol)
    contacts = {}
    for geom in sorted(set(so["first_contact_by_geom"]) | set(to["first_contact_by_geom"])):
        s = so["first_contact_by_geom"].get(geom)
        t = to["first_contact_by_geom"].get(geom)
        contacts[geom] = {"source": s, "target": t,
                          "target_minus_source_s": t["time_s"] - s["time_s"] if s and t else None}
    paired = [{"index": index, "apex_height_target_minus_source_m": t["sampled_apex_center_height_m"] - s["sampled_apex_center_height_m"],
               "energy_change_target_minus_source_j": t["energy_change_to_apex_j"] - s["energy_change_to_apex_j"]}
              for index, (s, t) in enumerate(zip(so["ground_rebounds"], to["ground_rebounds"]))]
    return {"initial_conditions": initial, "free_fall_sanity": sanity, "trajectory_errors": aligned_errors(source, target),
            "source": so, "target": to, "first_contact_comparison": contacts,
            "outcome_equal": so["outcome"] == to["outcome"], "paired_rebounds_by_ordinal": paired,
            "rebound_pairing_note": "Ordinal association only; unmatched rebounds remain in each engine observations."}


def source_parameters(reference):
    """Read authored settings from the frozen XML without compiling or stepping it."""
    scene = reference / "source_scene_absent.xml"
    if not scene.exists():
        return {"note": "Frozen scene unavailable; compiled source parameter parity is not established."}
    root = ET.parse(scene).getroot()
    selected = {"ground", "ball_collision"}
    return {"provenance": "Authored frozen XML attributes, not newly compiled solver readback; omitted attributes use MuJoCo defaults.",
            "option": root.find("option").attrib,
            "default_geom": root.find("default/geom").attrib,
            "collision_geoms": [geom.attrib for geom in root.iter("geom") if geom.get("name") in selected or geom.get("name", "").startswith("goal_") and geom.get("name", "").endswith("_collision")],
            "contact_pairs": [pair.attrib for pair in root.findall("contact/pair")]}


def compare_directories(reference, target):
    sp = json.loads((reference / "protocol.json").read_text())
    tp = json.loads((target / "protocol.json").read_text())
    if sp["cases"] != tp["cases"] or sp["repeats"] != 2 or tp["repeats"] != 2:
        raise ValueError("protocol requires identical cases and two repeats")
    if tuple(shot["name"] for shot in sp["cases"]) != CASES:
        raise ValueError("protocol must contain all five frozen G1 cases in order")
    for key in ("robot_mode", "seed", "ball_radius_m", "ball_mass_kg", "ball_inertia_kg_m2", "gravity_m_s2", "timestep_s"):
        if sp[key] != tp[key]:
            raise ValueError(f"physical protocol mismatch: {key}")
    cases, inputs = [], []
    for directory in (reference, target):
        for name in ("protocol.json", "manifest.json"):
            path = directory / name
            inputs.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    scene = reference / "source_scene_absent.xml"
    if scene.exists():
        inputs.append({"path": str(scene.resolve()), "sha256": hashlib.sha256(scene.read_bytes()).hexdigest()})
    for shot in sp["cases"]:
        repeats, source_hashes, target_hashes = [], [], []
        for repeat in range(2):
            records = []
            for directory, hashes in ((reference, source_hashes), (target, target_hashes)):
                path = directory / f"{shot['name']}_repeat_{repeat}.json"
                data = path.read_bytes()
                inputs.append({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()})
                record = json.loads(data)
                hashes.append(digest(record))
                records.append(record)
            repeats.append(compare_records(*records, sp, tp, shot))
        cases.append({"case": shot["name"], "repeats": repeats,
                      "source_repeat_full_record_sha256": source_hashes,
                      "target_repeat_full_record_sha256": target_hashes,
                      "source_repeat_identical": source_hashes[0] == source_hashes[1],
                      "target_repeat_identical": target_hashes[0] == target_hashes[1]})
    return {"stage": "G1 offline cross-engine observations", "gate_passed": False,
            "gate_note": "Metrics only: no acceptance tolerances or whole-robot/USD equivalence inferred.",
            "source_contact_time_semantics": sp.get("contact_time_semantics"),
            "target_contact_time_semantics": tp.get("contact_time_semantics", "unspecified"),
            "contact_comparison_note": "Source onset is pre-step clock with post-step after-velocity; timestep resolution is not exact impact timing. Target CONTACT_PERSISTS denotes first observed existing contact, not necessarily a new impact. Contact offsets can report separated surfaces.",
            "source_engine_parameters": source_parameters(reference),
            "target_engine_parameters": tp.get("engine_parameters", "UNSPECIFIED: inspect target manifest; parameter parity is not established."),
            "parameter_note": "Solver/contact models differ. MuJoCo solref/solimp are not PhysX restitution. Matching mass, inertia, gravity, timestep and initial conditions does not establish material or solver equivalence.",
            "source_manifest": json.loads((reference / "manifest.json").read_text()),
            "target_manifest": json.loads((target / "manifest.json").read_text()),
            "inputs": inputs, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/g1" /
                        datetime.now(timezone.utc).strftime("physics_comparison_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    report = compare_directories(args.reference_dir.resolve(), args.target_dir.resolve())
    output = args.output_dir.resolve()
    if not output.is_relative_to(PROJECT / "results"):
        parser.error("output-dir must be a new directory under Robot-Perception-Sim/results")
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output / "comparison.json")
    for case in report["cases"]:
        first = case["repeats"][0]
        print(f"{case['case']}: {first['source']['outcome']} / {first['target']['outcome']}; "
              f"position max={first['trajectory_errors']['position']['max_m']:.6g} m; "
              f"target repeat identical={case['target_repeat_identical']}")


if __name__ == "__main__":
    main()
