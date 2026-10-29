#!/usr/bin/env python3
"""Export G4 evidence; runs the unchanged G1 PhysX runner as a separate command."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trajectory.events import classify_record
from trajectory.shot_catalog import load_catalog, measured_csv_samples, procedural_samples


def frame_crossing_audit(record):
    """Check center crossings at either frame plane against native frame evidence.

    Conservative swept check of the frontal posts/bar. This does not claim a
    full USD narrow-phase overlap or validation of back-of-goal geometry.
    """
    suspicious = []
    for before, after in zip(record["trajectory"], record["trajectory"][1:]):
        a, b = before["position"], after["position"]
        for plane in (-11, 11):
            if a[0] == b[0] or not min(a[0], b[0]) <= plane <= max(a[0], b[0]):
                continue
            f = (plane-a[0]) / (b[0]-a[0])
            y, z = [a[i]+f*(b[i]-a[i]) for i in (1, 2)]
            intersects = ((abs(abs(y)-1.54) < .15 and -.11 < z < 2.19)
                          or (abs(z-2.04) < .15 and abs(y) < 1.65))
            if intersects:
                goal = "goal_right" if plane > 0 else "goal_left"
                implicated = set()
                if abs(abs(y)-1.54) < .15 and -.11 < z < 2.19:
                    side = "right" if y * plane > 0 else "left"
                    implicated.add(f"{goal}_{side}_post_collision")
                if abs(z-2.04) < .15 and abs(y) < 1.65:
                    implicated.add(f"{goal}_crossbar_collision")
                native = any(any(e["other_collider"].endswith("_" + geom) for geom in implicated)
                             and e["type"] in ("CONTACT_FOUND", "CONTACT_PERSISTS")
                             and abs(e["time_after_step"]-after["time"]) <= .0031
                             for e in record.get("raw_contact_events", []))
                if not native:
                    suspicious.append({"time_s": after["time"], "position_yz": [y,z]})
    return suspicious


def summarize(directory, catalog, detector_class, override_dirs=()):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    cases = []
    supplemental = {str(d): json.loads((Path(d) / "manifest.json").read_text()) for d in override_dirs}
    for shot in catalog["cases"]:
        records, hashes = [], []
        for repeat in range(catalog["repeats"]):
            filename = f'{shot["name"]}_repeat_{repeat}.json'
            path = directory / filename
            for override in override_dirs:
                candidate = Path(override) / filename
                if candidate.exists():
                    path = candidate
            record = json.loads(path.read_text())
            if record["shot"] != shot:
                raise ValueError("Run initial conditions differ from catalog")
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            records.append(record)
        ballistic = []
        record = records[0]
        first_contact = min((c["time"] for c in record["contacts"]), default=float("inf"))
        for sample in record["trajectory"]:
            t = sample["time"]
            if t > min(.01, first_contact):
                break
            expected_vz = shot["velocity"][2] - 9.81*t
            ballistic.append(abs(sample["velocity"][2] - expected_vz))
        ballistic_error = max(ballistic, default=0.)
        if ballistic_error > .001:
            raise ValueError("Native state velocity disagrees with independent gravity/time check")
        classification = classify_record(records[0], detector_class)
        cases.append({"case": shot["name"], **classification,
                      "record_directory": str(path.parent),
                      "repeat_identical": len(set(hashes)) == 1,
                      "first_10ms_max_ballistic_vz_error_m_s": ballistic_error,
                      "repeat_file_sha256": hashes,
                      "unreported_frame_crossings": frame_crossing_audit(records[0]),
                      "robot_mode": catalog["robot_mode"],
                      "initial_position_world_m": records[0]["initial_state"]["position"],
                      "initial_angular_velocity_world_rad_s": shot["angular_velocity"]})
    return {"cases": cases, "case_count": len(cases),
            "outcome_counts": dict(Counter(c["outcome"] for c in cases)),
            "classification_counts": dict(Counter(c["classification"] for c in cases)),
            "repeat_identical": all(c["repeat_identical"] for c in cases),
            "unreported_frame_crossing_count": sum(len(c["unreported_frame_crossings"]) for c in cases),
            "native_manifest": manifest, "supplemental_native_manifests": supplemental,
            "limitations": ["Camera visibility/occlusion is unobserved by physical runs; use separate G5 image evidence",
                            "Procedural and CSV replay do not model collision response",
                            "No measured real-ball CSV supplied; ingestion verified with synthetic fixture only",
                            "No aerodynamic Magnus model; spin affects contacts only",
                            "Frame audit is frontal swept geometry, not full USD narrow phase"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "config/scenarios/g4_golden.json")
    parser.add_argument("--source-root", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--physical-dir", type=Path)
    sources.add_argument("--measured-csv", type=Path)
    sources.add_argument("--procedural-case")
    parser.add_argument("--override-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    from k1_goalkeeper.detection import GoalDetector
    catalog = load_catalog(args.catalog)
    if args.physical_dir:
        result = summarize(args.physical_dir, catalog, GoalDetector, args.override_dir)
    elif args.measured_csv:
        samples, provenance = measured_csv_samples(args.measured_csv)
        result = {**provenance, "trajectory": samples,
                  "labels": classify_record({"trajectory": samples, **provenance}, GoalDetector)}
    elif args.procedural_case:
        shot = next(s for s in catalog["cases"] if s["name"] == args.procedural_case)
        samples = list(procedural_samples(shot, catalog["timestep_s"]))
        result = {"source": "procedural_kinematic", "physical_collision_response": False,
                  "trajectory": samples, "labels": classify_record({"trajectory": samples,
                  "source": "procedural_kinematic"}, GoalDetector)}
    else:
        parser.error("Choose physical directory, measured CSV, or procedural case")
    result["exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["catalog_sha256"] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    result["detector_sha256"] = hashlib.sha256((args.source_root / "k1_goalkeeper/detection.py").read_bytes()).hexdigest()
    if args.physical_dir:
        manifests = [result["native_manifest"], *result["supplemental_native_manifests"].values()]
        if any(m["source_detector_sha256"] != result["detector_sha256"] for m in manifests):
            raise ValueError("Detector source differs from the actual native runs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
