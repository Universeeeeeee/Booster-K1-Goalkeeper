"""Stage F: no-shot positioning gate, followed by matched goalkeeper comparisons."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch

from k1_goalkeeper.positioning import ResponseLimits
from k1_goalkeeper.positioning_validation import prediction_errors, run_trial, save_trial
from k1_goalkeeper.scene import build_scene
from scripts.validate_walk import verify_sources
from scripts.validate_impacts import compare


def shots():
    result = []
    def add(name, position, velocity, spin=(0., 0., 0.), kind="rolling"):
        result.append({"name": name, "kind": kind, "launch": {"position": list(position),
                       "velocity": list(velocity), "angular_velocity": list(spin)}})
    for name, y, x, speed in [("center", 0., -1., 1.8), ("reachable_left", -.35, -3., 1.5),
                             ("reachable_right", .35, -3., 1.5), ("wide_left", -.9, 0., 3.),
                             ("wide_right", .9, 0., 3.)]:
        add(name, (x, y, .111), (speed, 0., 0.), (0., speed/.11, 0.))
    add("angled", (-3., -.5, .111), (1.5, .09, 0.), (-.09/.11, 1.5/.11, 0.))
    for name, y, height in [("air_center", 0., .55), ("air_wide", .6, .55), ("air_high", 0., 1.4)]:
        flight = (10.4-4.)/8.
        vz = (height-.2+4.905*flight*flight)/flight
        add(name, (4., y, .2), (8., 0., vz), kind="airborne")
    add("bouncing", (2., -.35, .8), (3., 0., 0.), kind="bouncing")
    return result


def protected_files():
    files = [ROOT/p for p in ["k1_goalkeeper/walk_controller.py", "k1_goalkeeper/scene_simulation.py",
              "k1_goalkeeper/velocity_validation.py", "k1_goalkeeper/scene.py", "k1_goalkeeper/simulation.py",
              "k1_goalkeeper/detection.py", "config/simulation.json", "models/futsal_goal_3x2_reference.urdf",
              "assets/booster_assets/robots/K1/K1_22dof.xml"]]
    files += list((ROOT/"tests").glob("test_*.py"))
    files = [p for p in files if p.name != "test_positioning.py"]
    for directory in ("walk_validation", "walk_validation_1ms", "velocity_validation", "goalkeeper_validation", "impact_validation"):
        files += list((ROOT/"reports"/directory).rglob("*.json"))
    files += list((ROOT/"assets/booster_deploy").rglob("*.pt"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run(output, phase="all", repeats=2):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output.mkdir(parents=True, exist_ok=True)
    protected = protected_files()
    sources = verify_sources()
    models = {mode: mujoco.MjModel.from_xml_path(str(build_scene(output/f"scene_{mode}.xml", robot_mode=mode))) for mode in ("free", "absent")}
    report = {"stage": "F", "phase": phase, "repeats": repeats, "limits": asdict(ResponseLimits()),
              "sources": sources, "protected_sha256": protected,
              "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__, "torch": torch.__version__},
              "positioning": [], "shots": [], "results": [], "f1_passed": False}

    def execute(name, mode, launch, target=0., duration=18.):
        model = models["absent" if mode == "absent" else "free"]
        first, trace = run_trial(model, name, mode, launch, target=target, duration=duration)
        hashes = [first["trajectory_sha256"]]
        for _ in range(repeats-1):
            second, _ = run_trial(model, name, mode, launch, target=target, duration=duration)
            hashes.append(second["trajectory_sha256"])
        first["reproducibility"] = {"verified": repeats >= 2 and len(set(hashes)) == 1, "hashes": hashes}
        save_trial(output/name, first, trace)
        report["results"].append({key: value for key, value in first.items() if key != "high_level_records"})
        print(name, "fall", bool(first["fall_events"]), "contact", first["effective_contact"],
              "right_goal", first["right_goal"], "repeat", first["reproducibility"]["verified"], flush=True)
        return first, trace

    def persist():
        report["protected_unchanged"] = protected_files() == protected
        report["all_repeats_identical"] = bool(report["results"]) and all(s["reproducibility"]["verified"] for s in report["results"])
        report["code_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ("k1_goalkeeper/positioning.py", "k1_goalkeeper/positioning_validation.py", "k1_goalkeeper/impact_validation.py", "scripts/validate_positioning.py")}
        (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    parked = {"position": [0., 0., .111], "velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.]}
    for target in [0., .2, -.2, .4, -.4, .6, -.6]:
        summary, trace = execute(f"position_{target:+.2f}", "position", parked, target=target)
        report["positioning"].append(summary["positioning"])
        persist()
    report["f1_passed"] = all(s["passed"] for s in report["positioning"])
    persist()
    if not report["f1_passed"] or phase == "position":
        return report
    # F1 is an explicit physical gate; no shot simulations run before it succeeds.
    for shot in shots():
        outcomes, traces = {}, {}
        for mode in ("absent", "passive", "active"):
            outcomes[mode], traces[mode] = execute(shot["name"]+"_"+mode, mode, shot["launch"])
        absent = outcomes["absent"]
        for mode in ("passive", "active"):
            summary, trace = outcomes[mode], traces[mode]
            contacts = [c for c in summary["contacts"] if c["kind"] == "robot"]
            first_contact = min((c["start_time"] for c in contacts), default=float(trace["time"][-1]))
            records = [r for r in summary["high_level_records"] if r["time"] < first_contact]
            summary["prediction_errors_against_matched_absent"] = prediction_errors(records, absent["plane_crossings"])
            reaches = [r["reachability"] for r in records if r["reachability"]]
            tracking = [r["tracking"] for r in records if r["tracking"]]
            summary["reachability_summary"] = {"first": reaches[0] if reaches else None,
                "reachable_updates": sum(r["reachable"] for r in reaches),
                "unreachable_updates": sum(not r["reachable"] for r in reaches),
                "interpretation": "center positioning feasibility, not a promised interception or complete collision silhouette"}
            summary["tracking_rmse_m"] = float(np.sqrt(np.mean([r["error_m"]**2 for r in tracking]))) if tracking else None
            summary["blocked"] = bool(summary["effective_contact"] and summary["right_goal"] is False) if absent["right_goal"] and summary["completed"] else None
            summary["physical_comparison"] = compare(summary, trace, absent, traces["absent"])
            summary["launch_identical_to_absent"] = all(summary["launch"][k] == absent["launch"][k] for k in ("position", "velocity", "angular_velocity"))
            save_trial(output/summary["case"]["name"], summary, trace)
            for index, old in enumerate(report["results"]):
                if old["case"]["name"] == summary["case"]["name"]:
                    report["results"][index] = {key: value for key, value in summary.items() if key != "high_level_records"}
        passive, active = outcomes["passive"], outcomes["active"]
        row = {**shot, "absent_right_goal": absent["right_goal"], "passive_right_goal": passive["right_goal"],
               "active_right_goal": active["right_goal"], "passive_contact": passive["effective_contact"],
               "active_contact": active["effective_contact"], "active_blocked": active["blocked"],
               "added_success": bool(absent["right_goal"] and passive["right_goal"] and active["blocked"]),
               "lost_passive_success": bool(absent["right_goal"] and passive["blocked"] and active["right_goal"]),
               "active_first_reachability": active["reachability_summary"]["first"]}
        report["shots"].append(row)
        print("comparison", row, flush=True)
        persist()
    report["added_success_count"] = sum(s["added_success"] for s in report["shots"])
    report["lost_passive_success_count"] = sum(s["lost_passive_success"] for s in report["shots"])
    report["fall_case_count"] = sum(bool(s["fall_events"]) for s in report["results"])
    report["numerical_error_count"] = sum(s["numerical_error"] is not None for s in report["results"])
    report["unexpected_contact_count"] = sum(bool(s["unexpected_contacts"]) for s in report["results"])
    persist()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/positioning_validation")
    parser.add_argument("--phase", choices=("position", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    result = run(args.output, args.phase, args.repeats)
    if not result["f1_passed"] or not result["protected_unchanged"]:
        raise SystemExit(1)
