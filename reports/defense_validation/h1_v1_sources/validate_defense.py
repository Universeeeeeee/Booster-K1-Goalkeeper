"""Freeze and execute H1 gates; larger residuals require every earlier gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np

from k1_goalkeeper.defense_controller import DefensePulse
from k1_goalkeeper.defense_validation import run_defense_trial
from k1_goalkeeper.scene import build_scene
from scripts.validate_walk import verify_sources


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    verify_sources()
    manifest_path = ROOT/"reports/defense_validation/protected_manifest.json"
    source_hashes = json.loads(manifest_path.read_text())
    if not all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in source_hashes.items()):
        raise RuntimeError("protected source differs from root H0 manifest before protocol freeze")
    protocol = {"stage": "H1", "repeats": 2, "amplitudes_rad": [.05, .15, .30],
                "settle_s": 3., "rise_s": 2., "hold_s": 2., "return_s": 2., "recovery_s": 5.,
                "ramp": "quintic 10u^3-15u^4+6u^5; peak derivative 1.875*amplitude/2s",
                "cases": {"left_roll": [["left_shoulder_roll_joint", 1]],
                          "right_roll": [["right_shoulder_roll_joint", -1]],
                          "both_roll": [["left_shoulder_roll_joint", 1], ["right_shoulder_roll_joint", -1]],
                          "both_pitch": [["aaleft_shoulder_pitch_joint", -1], ["aaright_shoulder_pitch_joint", -1]]},
                "guards": {"minimum_base_z_m": .42, "maximum_tilt_degrees": 15., "selected_target_margin_rad": .05,
                           "new_self_contact": "stop", "new_nonfoot_ground_contact": "stop", "any_robot_ball_goal_contact": "stop",
                           "continuous_arm_raw_saturation_s": .10, "actual_selected_q_inside_XML_range": True,
                           "unselected_targets_identical_to_same_tick_nominal": True, "delivered_torque_limit_tolerance_nm": 1e-5},
                "contact_reference": "same physical timestep in matched zero baseline; no union across future baseline contacts",
                "authority": "every selected joint signed median delta-q versus matched zero baseline in last1s hold >=50% amplitude",
                "recovery": "existing recovery_metrics over t9..14: full1s slow bins, last2s mean velocity within0.05/0.05/0.1, z>=0.45, tilt<=10deg",
                "guard_classification": "deliberate observer stops are censored guard events; original E numerical_error diagnostic retained separately, not counted as numerical instability",
                "gate": "exact zero-residual parity x2 first; identical whole-trace repeats and stable+authority+recovered for every case before larger stage; authority-only failure permits remaining same-amplitude families, blocks larger stages/H3/H4; safety/recovery failure repeats identical case then stops all other cases",
                "mutation": "instance compute_targets hook only; original action/history/filter/PD/files unchanged",
                "history": "official nominal actor last_action retained; applied targets logged separately",
                "implementation_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (ROOT/"k1_goalkeeper/defense_controller.py", ROOT/"k1_goalkeeper/defense_validation.py", ROOT/"scripts/validate_defense.py")},
                "original_source_sha256": source_hashes}
    path = output/"protocol.json"
    payload = json.dumps(protocol, indent=2, allow_nan=False)+"\n"
    if path.exists() and path.read_text() != payload:
        raise RuntimeError("frozen H1 protocol exists with different content")
    path.write_text(payload)
    protocol_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    model = mujoco.MjModel.from_xml_path(str(build_scene(output/"scene_free.xml")))
    results = {"stage": "H1", "protocol_sha256": protocol_hash, "zero_parity": None,
               "baseline": [], "trials": [], "gate_passed": False, "H3": "blocked_by_H1_gate", "H4": "blocked_by_H1_gate"}

    def save():
        verify_sources()
        unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in source_hashes.items())
        results["original_sources_unchanged"] = unchanged
        implementation_unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest
                                       for p, digest in protocol["implementation_sha256"].items())
        if not unchanged or not implementation_unchanged or hashlib.sha256(path.read_bytes()).hexdigest() != protocol_hash:
            raise RuntimeError("protected source or frozen protocol drift")
        (output/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")

    baselines = []
    for repeat in range(2):
        summary, trace = run_defense_trial(model, None, f"zero_original_r{repeat}", output=output/f"zero_original_r{repeat}")
        results["baseline"].append(summary)
        baselines.append(trace)
        save()
        if not summary["passed"]:
            results["blocked_reason"] = "zero baseline fails frozen stability/recovery guard"
            save()
            return results
    zero = DefensePulse((("left_shoulder_roll_joint", 0.), ("right_shoulder_roll_joint", 0.)))
    wrappers, wrapper_traces, differences = [], [], []
    for repeat in range(2):
        summary, trace = run_defense_trial(model, zero, f"zero_wrapper_r{repeat}", output=output/f"zero_wrapper_r{repeat}", baseline=baselines[repeat])
        wrappers.append(summary)
        wrapper_traces.append(trace)
        differences.append([key for key in baselines[repeat] if not np.array_equal(baselines[repeat][key], trace[key])])
    repeat_differences = [key for key in baselines[0] if not np.array_equal(baselines[0][key], baselines[1][key])]
    wrapper_repeat_differences = [key for key in wrapper_traces[0] if not np.array_equal(wrapper_traces[0][key], wrapper_traces[1][key])]
    results["zero_parity"] = {"passed": not any(differences) and not repeat_differences and not wrapper_repeat_differences,
                              "wrapper_differences": differences, "baseline_repeat_differences": repeat_differences,
                              "wrapper_repeat_differences": wrapper_repeat_differences,
                              "compared_arrays": list(baselines[0]), "wrapper_summaries": wrappers}
    save()
    if not results["zero_parity"]["passed"]:
        results["blocked_reason"] = "zero residual or repeat parity failed"
        save()
        return results
    for amplitude in protocol["amplitudes_rad"]:
        stage_failed = False
        for action, offsets in protocol["cases"].items():
            case_summaries, case_traces = [], []
            for repeat in range(2):
                pulse = DefensePulse(tuple((name, sign*amplitude) for name, sign in offsets))
                name = f"{action}_{amplitude:.2f}_r{repeat}"
                summary, trace = run_defense_trial(model, pulse, name, output=output/name, baseline=baselines[repeat])
                results["trials"].append(summary)
                case_summaries.append(summary)
                case_traces.append(trace)
                print(f"{name}: stable={summary['stable']} recovered={summary['recovered']} authority={summary['authority_passed']} response={summary['response']}", flush=True)
                save()
                if not summary["passed"]:
                    stage_failed = True
                    results["blocked_reason"] = f"amplitude {amplitude:g}: {name} failed {summary['failure_class']}"
            repeat_differences = [key for key in case_traces[0] if not np.array_equal(case_traces[0][key], case_traces[1][key])]
            for summary in case_summaries:
                summary["repeat_identical"] = not repeat_differences
                summary["repeat_differences"] = repeat_differences
            if repeat_differences:
                stage_failed = True
                results["blocked_reason"] = f"{action}_{amplitude:.2f} repeat trace mismatch"
            unsafe = any(not summary["stable"] or not summary["recovered"] for summary in case_summaries)
            save()
            if unsafe or repeat_differences:
                break
        if stage_failed:
            save()
            return results
    results["gate_passed"] = True
    results["H3"] = results["H4"] = "not_executed_H1_only"
    save()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"reports/defense_validation")
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"gate_passed": report["gate_passed"], "blocked_reason": report.get("blocked_reason")}, indent=2))
