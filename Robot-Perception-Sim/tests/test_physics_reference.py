"""Rebound labels require actual contact and a subsequent upward transition."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "record_physics_reference", Path(__file__).resolve().parents[1] / "scripts/record_physics_reference.py")
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def test_soft_contact_reversal_is_one_rebound():
    record = {"timestep": 0.001,
              "contacts": [{"geom": "ground", "time": 0.1, "ball_velocity_before": [0, 0, -2]}],
              "trajectory": [{"time": time, "velocity": [0, 0, velocity]}
                             for time, velocity in [(0.101, -1), (0.102, -0.1),
                                                    (0.103, 0.1), (0.104, 0.5)]]}
    events = recorder.ground_rebounds(record)
    assert len(events) == 1
    assert events[0]["upward_sample_time_s"] == 0.103


def test_no_ground_contact_or_no_upward_speed_is_not_rebound():
    record = {"timestep": 0.001, "contacts": [],
              "trajectory": [{"time": 0.2, "velocity": [0, 0, 1]}]}
    assert recorder.ground_rebounds(record) == []
    record["contacts"] = [{"geom": "ground", "time": 0.1, "ball_velocity_before": [0, 0, -1]}]
    record["trajectory"][0]["velocity"][2] = 0.01
    assert recorder.ground_rebounds(record) == []


def test_next_contact_bounds_previous_rebound_search():
    record = {"timestep": 0.001,
              "contacts": [{"geom": "ground", "time": time, "ball_velocity_before": [0, 0, -1]}
                           for time in (0.1, 0.2)],
              "trajectory": [{"time": 0.15, "velocity": [0, 0, -0.1]},
                             {"time": 0.21, "velocity": [0, 0, 0.2]}]}
    events = recorder.ground_rebounds(record)
    assert len(events) == 1
    assert events[0]["ground_contact_onset_s"] == 0.2
