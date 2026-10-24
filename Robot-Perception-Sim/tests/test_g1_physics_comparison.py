"""Offline G1 checks; no simulator execution or reference mutation.

Integration checks require the existing gitignored frozen archive at comparison.REFERENCE.
Restore that archive before running; never regenerate references to satisfy these tests.
"""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "compare_g1_physics", Path(__file__).resolve().parents[1] / "scripts/compare_g1_physics.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def record(times=(0, 0.1, 0.2), heights=(1, 0.8, 0.7), vertical_speeds=(0, -1, -2)):
    return {"trajectory": [{"time": t, "position": [t, 0, z], "velocity": [1, 0, vz],
                            "angular_velocity_body": [0, 0, 0]}
                           for t, z, vz in zip(times, heights, vertical_speeds)],
            "contacts": [], "outcome": "Timeout", "exit_event": None, "timestep": 0.1}


def protocol():
    return {"ball_mass_kg": 2, "ball_inertia_kg_m2": [0.1] * 3,
            "gravity_m_s2": [0, 0, -10], "ball_radius_m": 0.1}


def test_alignment_interpolates_different_grids_and_never_extrapolates():
    source = record((0, 1, 2), (1, 2, 3), (0, 1, 2))
    target = record((0, 0.5, 1.5), (1, 1.5, 2.5), (0, 0.5, 1.5))
    errors = comparison.aligned_errors(source, target)
    assert errors["overlap_end_s"] == 1.5
    assert errors["source_end_s"] == 2
    assert errors["aligned_samples"] == 4
    assert errors["position"]["max_m"] == 0
    assert errors["velocity"]["max_m_s"] == 0
    target["trajectory"][1]["position"][1] = 0.2
    assert comparison.aligned_errors(source, target)["position"]["max_m"] == pytest.approx(0.2)


def test_initial_validation_checks_delivered_first_sample_not_just_metadata():
    shot = {"name": "test", "position": [0, 0, 1], "velocity": [1, 0, 0],
            "angular_velocity": [0, 0, 0]}
    sample = record()
    sample["shot"] = shot
    sample["initial_state"] = copy.deepcopy(sample["trajectory"][0])
    assert comparison.validate_initial(sample, shot)["passed"]
    sample["trajectory"][0]["position"][2] += 0.001
    with pytest.raises(ValueError, match="initial position mismatch"):
        comparison.validate_initial(sample, shot)


def test_rebound_apex_energy_uses_original_samples_and_requires_actual_contact():
    sample = record((0, 0.1, 0.2, 0.3, 0.4), (1, 0.1, 0.3, 0.5, 0.4), (-2, -1, 2, 0, -1))
    assert comparison.observations(sample, protocol())["ground_rebounds"] == []
    sample["contacts"] = [{"time": 0.1, "geom": "ground", "kind": "ground",
                           "ball_velocity_before": [1, 0, -1]}]
    observed = comparison.observations(sample, protocol())
    rebound = observed["ground_rebounds"][0]
    assert rebound["sampled_apex_center_height_m"] == 0.5
    assert rebound["sampled_apex_clearance_m"] == 0.4
    assert rebound["apex_bracketed_by_descent"]
    # pre-onset sample E = 0.5*2*(1+1) + 2*10*0.1 = 4 J;
    # sampled apex E = 0.5*2*1 + 2*10*0.5 = 11 J. Report signed gain honestly.
    assert rebound["energy_change_to_apex_j"] == pytest.approx(7)
    assert observed["contact_duration_s"] is None
    sample["contacts"].append({"time": 0.15, "geom": "ground", "kind": "ground",
                               "ball_velocity_before": [1, 0, 1]})
    assert comparison.observations(sample, protocol())["ground_rebounds"] == []


def test_invalid_time_and_nonfinite_states_are_rejected():
    sample = record()
    sample["trajectory"][1]["time"] = 0
    with pytest.raises(ValueError, match="strictly increasing"):
        comparison.trajectory_arrays(sample)
    sample = record()
    sample["trajectory"][1]["velocity"][0] = np.nan
    with pytest.raises(ValueError, match="invalid trajectory velocity"):
        comparison.trajectory_arrays(sample)


def test_frozen_five_cases_two_repeats_self_comparison_and_energies():
    # Identity comparison verifies wiring against actual immutable evidence, not synthetic summaries.
    report = comparison.compare_directories(comparison.REFERENCE, comparison.REFERENCE)
    assert [case["case"] for case in report["cases"]] == list(comparison.CASES)
    assert report["gate_passed"] is False
    assert len(report["inputs"]) == 25
    for case in report["cases"]:
        assert len(case["repeats"]) == 2
        assert case["source_repeat_identical"] and case["target_repeat_identical"]
        for repeat in case["repeats"]:
            assert repeat["outcome_equal"]
            assert repeat["trajectory_errors"]["position"]["max_m"] == 0
            assert repeat["trajectory_errors"]["velocity"]["max_m_s"] == 0
    bounce = report["cases"][1]["repeats"][0]["source"]["ground_rebounds"]
    assert len(bounce) == 4
    assert bounce[0]["sampled_apex_center_height_m"] > 0.2
    assert bounce[0]["energy_change_to_apex_j"] < 0


def test_directory_comparison_rejects_incomplete_or_changed_protocol(tmp_path):
    frozen = json.loads((comparison.REFERENCE / "protocol.json").read_text())
    frozen["cases"][0]["position"][2] = 2
    (tmp_path / "protocol.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="identical cases"):
        comparison.compare_directories(comparison.REFERENCE, tmp_path)


def test_float32_spin_rounding_is_accepted_but_actual_offset_is_rejected():
    shot = {"name": "spin", "position": [0, 0, 1], "velocity": [1, 0, 0],
            "angular_velocity": [0, 72.72727272727273, 0]}
    sample = record()
    sample["shot"] = shot
    spin = [0, float(np.float32(shot["angular_velocity"][1])), 0]
    sample["trajectory"][0]["angular_velocity_body"] = spin
    sample["initial_state"] = copy.deepcopy(sample["trajectory"][0])
    assert comparison.validate_initial(sample, shot)["passed"]
    sample["trajectory"][0]["angular_velocity_body"][1] += 0.001
    with pytest.raises(ValueError, match="initial angular_velocity_body mismatch"):
        comparison.validate_initial(sample, shot)


def test_record_timestep_is_checked_independently_of_protocol():
    p = json.loads((comparison.REFERENCE / "protocol.json").read_text())
    source = json.loads((comparison.REFERENCE / "free_fall_repeat_0.json").read_text())
    target = copy.deepcopy(source)
    target["timestep"] = 0.002
    with pytest.raises(ValueError, match="record timestep"):
        comparison.compare_records(source, target, p, p, p["cases"][0])


def test_source_parameters_include_both_goal_frames():
    parameters = comparison.source_parameters(comparison.REFERENCE)
    names = {geom["name"] for geom in parameters["collision_geoms"]}
    assert names == {"ground", "ball_collision", "goal_left_left_post_collision",
                     "goal_left_right_post_collision", "goal_left_crossbar_collision",
                     "goal_right_left_post_collision", "goal_right_right_post_collision",
                     "goal_right_crossbar_collision"}


def test_free_fall_sanity_rejects_mislabeled_physics_clock():
    p = json.loads((comparison.REFERENCE / "protocol.json").read_text())
    sample = json.loads((comparison.REFERENCE / "free_fall_repeat_0.json").read_text())
    assert comparison.validate_free_fall(sample, p)["passed"]
    sample["trajectory"][-1]["position"][2] = 0.11
    with pytest.raises(ValueError, match="invalid free_fall clock/physics"):
        comparison.validate_free_fall(sample, p)


def test_free_fall_sanity_accepts_float32_delivery_and_rejects_false_contacts():
    p = json.loads((comparison.REFERENCE / "protocol.json").read_text())
    sample = json.loads((comparison.REFERENCE / "free_fall_repeat_0.json").read_text())
    for state in sample["trajectory"]:
        for key in ("position", "velocity"):
            state[key] = np.asarray(state[key], dtype=np.float32).astype(float).tolist()
    assert comparison.validate_free_fall(sample, p)["passed"]
    sample["contacts"] = [{"time": 0.1}]
    with pytest.raises(ValueError, match="invalid free_fall clock/physics"):
        comparison.validate_free_fall(sample, p)


def test_free_fall_sanity_accepts_accumulated_float32_callback_time():
    p = json.loads((comparison.REFERENCE / "protocol.json").read_text())
    sample = json.loads((comparison.REFERENCE / "free_fall_repeat_0.json").read_text())
    for step, state in enumerate(sample["trajectory"]):
        state["time"] = step * float(np.float32(p["timestep_s"]))
    assert comparison.validate_free_fall(sample, p)["passed"]
    sample["trajectory"][-1]["time"] -= 0.001
    with pytest.raises(ValueError, match="invalid free_fall clock/physics"):
        comparison.validate_free_fall(sample, p)


def test_initial_persistent_contact_is_labeled_without_claiming_new_impact():
    sample = record()
    sample["contacts"] = [{"time": 0, "geom": "ground", "kind": "ground",
                           "onset_observation": "CONTACT_PERSISTS", "ball_velocity_before": [1, 0, 0]}]
    observed = comparison.observations(sample, protocol())
    assert observed["first_contact_by_geom"]["ground"]["observation"] == "CONTACT_PERSISTS"
    assert observed["ground_rebounds"] == []
