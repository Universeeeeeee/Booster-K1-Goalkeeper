"""H4 current-observation logic and zero-wrapper boundaries; no H4 shots."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from k1_goalkeeper import defense_goalkeeper as module
from k1_goalkeeper.defense_controller import DefensePulse
from k1_goalkeeper.scene_simulation import SceneSimulation


@pytest.fixture
def logic(monkeypatch):
    adapter = object.__new__(module.DefenseGoalkeeper)
    adapter.wrapper = SimpleNamespace(pulse=DefensePulse(()))
    adapter.pulse = None
    adapter.stationary_since = None
    adapter.arm_request = False
    adapter.contact_seen = False
    adapter.records = []
    adapter.last_command = [0., 0., 0.]
    adapter.position = SimpleNamespace(direction=0, last_switch=0., filtered_velocity=0., turning=False)
    current = {"valid": True, "deadline": 4., "inside_aperture": True, "moving": False,
               "ball_x": 7., "ball_vx": 1., "emit": True}
    sim = SimpleNamespace(data=SimpleNamespace(time=0., qpos=np.zeros(22), qvel=np.zeros(22)),
                          controller=SimpleNamespace(qpos_addresses=np.arange(22), dof_addresses=np.arange(22)))
    def state(controller):
        return {"body_linear_velocity": np.array([0., .2 if current["moving"] else 0., 0.]),
                "body_angular_velocity": np.zeros(3), "rpy": np.array([0., 0., np.pi]),
                "base_position": np.array([10.4, 0., .55369])}
    monkeypatch.setattr(module, "body_state", state)
    def parent(self, scene):
        if not current["emit"]:
            return self.last_command
        prediction = {"valid": current["valid"], "time_to_plane_s": current["deadline"],
                      "inside_aperture": current["inside_aperture"],
                      "future_floor_contact_unmodeled": False}
        self.records.append({"time": float(scene.data.time), "prediction": prediction.copy(),
                             "interception_prediction": prediction.copy(),
                             "ball_velocity": [current["ball_vx"], 0., 0.],
                             "ball_position": [current["ball_x"], 0., .5],
                             "world_position": [10.4, 0., .55369],
                             "forecast_observed_time": float(scene.data.time),
                             "command": [0., .2, 0.]})
        self.last_command = [0., .2, 0.]
        return self.last_command
    monkeypatch.setattr(module.PredictiveGoalkeeper, "command", parent)
    def tick(time):
        sim.data.time = time
        return adapter.command(sim)
    return adapter, current, sim, tick


def test_current_usable_forecast_waits_for_one_second_actually_slow(logic):
    adapter, current, _, tick = logic
    current["moving"] = True
    assert tick(2.02) == [0., 0., 0.]
    assert adapter.pulse is None
    current["moving"] = False
    tick(2.50)
    tick(3.48)
    assert adapter.pulse is None
    assert tick(3.50) == [0., 0., 0.]
    assert adapter.pulse.start_s == 3.50
    assert dict(adapter.pulse.joint_offsets) == {"left_shoulder_roll_joint": .15,
                                               "right_shoulder_roll_joint": -.15}
    assert adapter.pulse.rise_s == adapter.pulse.hold_s == adapter.pulse.return_s == 2.


@pytest.mark.parametrize("field,value", [
    ("valid", False), ("deadline", -.01), ("inside_aperture", False),
    ("ball_x", 10.5), ("ball_vx", -.1),
])
def test_invalid_or_late_current_prediction_never_starts_arm(logic, field, value):
    adapter, current, _, tick = logic
    current[field] = value
    tick(2.02)
    tick(3.02)
    assert adapter.pulse is None
    assert not adapter.arm_request


def test_deadline_rechecked_while_waiting_to_stop(logic):
    adapter, current, _, tick = logic
    current["moving"] = True
    tick(2.02)
    assert adapter.arm_request
    current["moving"] = False
    tick(2.10)
    current["deadline"] = -.01
    assert tick(3.10) == [0., .2, 0.]
    assert adapter.pulse is None and not adapter.arm_request


def test_translation_stays_suspended_for_single_shot_episode_after_pulse(logic):
    adapter, current, _, tick = logic
    tick(2.02)
    tick(3.02)
    start = adapter.pulse.start_s
    for offset in (1., 2., 4., 6., 10.98):
        assert tick(start + offset) == [0., 0., 0.]
        assert adapter.records[-1]["command"] == [0., 0., 0.]
    current["emit"] = False
    assert tick(start + 10.99) == [0., 0., 0.]
    current["emit"] = True
    assert tick(start + 11.) == [0., 0., 0.]
    assert adapter.records[-1]["g_nominal_command"] == [0., .2, 0.]
    assert tick(start + 16.) == [0., 0., 0.]


def test_positive_short_deadline_is_explicit_partial_experiment(logic):
    adapter, current, _, tick = logic
    current["deadline"] = .25
    tick(2.02)
    tick(3.02)
    assert adapter.pulse is not None
    assert not adapter.records[-1]["h_full_rise_fits_current_deadline"]
    assert adapter.records[-1]["h_late_partial_experiment"]
    assert adapter.records[-1]["command"] == [0., 0., 0.]


def test_future_truth_cannot_enter_command_decisions(logic):
    adapter, _, _, tick = logic
    class ForbiddenTruth(dict):
        def get(self, *args):
            pytest.fail("future actual crossing used for arm decision")
        def __getitem__(self, key):
            pytest.fail("future actual crossing used for arm decision")
    adapter.crossings = ForbiddenTruth({"11.0": {"time": 1000., "position": [11., 0., 0.]}})
    tick(2.02)
    tick(3.02)
    assert adapter.pulse.start_s == 3.02


def test_h4_install_does_not_reinitialize_policy_and_zero_hook_matches_native(free_model, monkeypatch):
    torch.set_num_threads(1)
    reference = SceneSimulation(free_model, control_mode="k1_walk")
    intervention = SceneSimulation(free_model, control_mode="k1_walk")
    for _ in range(20):
        reference.step()
        intervention.step()
    def forbidden():
        pytest.fail("H4 installation reinitialized original controller/policy")
    monkeypatch.setattr(intervention.controller, "initialize", forbidden)
    monkeypatch.setattr(intervention.controller.policy, "reset", forbidden)
    adapter = object.__new__(module.DefenseGoalkeeper)
    adapter.install(intervention)
    for _ in range(100):
        reference.step()
        intervention.step()
        for key in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(getattr(reference.data, key), getattr(intervention.data, key))
        for key in ("obs_history", "last_action", "filtered_dof_target"):
            assert torch.equal(getattr(reference.controller.policy, key),
                               getattr(intervention.controller.policy, key))
    assert reference.data.time == intervention.data.time == pytest.approx(.12)
