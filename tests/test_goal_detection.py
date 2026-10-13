import unittest

import numpy as np
import pytest

from k1_goalkeeper.detection import GoalDetector
from k1_goalkeeper.simulation import Shot, preset_shots, random_shots, run_shot


class GoalDetectionTests(unittest.TestCase):
    def crossing(self, start, end):
        detector = GoalDetector()
        detector.update(0, start)
        return detector.update(1, end)

    def test_center_crossing_is_not_whole_ball_crossing(self):
        detector = GoalDetector()
        detector.update(0, (10.9, 0, 0.5))
        self.assertIsNone(detector.update(1, (11.05, 0, 0.5)))
        self.assertIsNone(detector.update(2, (11.11, 0, 0.5)))
        self.assertEqual(detector.update(3, (11.12, 0, 0.5)).outcome, "Goal")

    def test_both_goal_directions(self):
        for sign, goal in ((1, "right"), (-1, "left")):
            event = self.crossing((sign * 10, 0, 0.11), (sign * 12, 0, 0.11))
            self.assertEqual(event.outcome, "Goal")
            self.assertEqual(event.goal, goal)
            self.assertAlmostEqual(event.time, 0.555)
            self.assertAlmostEqual(event.position[0], sign * 11.11)

    def test_aperture_at_interpolated_crossing(self):
        # Endpoint is far above the crossbar, but the crossing was inside.
        event = self.crossing((11, 0, 1.7), (12, 0, 3))
        self.assertEqual(event.outcome, "Goal")
        self.assertAlmostEqual(event.position[2], 1.843)
        # Endpoint is inside; the ball actually crossed outside the post.
        event = self.crossing((11, 2, 0.5), (12, 0, 0.5))
        self.assertEqual(event.outcome, "Miss")

    def test_outside_aperture(self):
        for y, z in ((-1.6, 0.5), (1.6, 0.5), (0, 2.1), (0, -0.2)):
            self.assertEqual(self.crossing((10, y, z), (12, y, z)).outcome, "Miss")

    def test_sphere_extent_checks(self):
        for y, z in ((1.4, 0.5), (0, 1.9)):
            self.assertEqual(self.crossing((10, y, z), (12, y, z)).outcome, "Miss")
        self.assertEqual(self.crossing((10, 1.38, 1.88), (12, 1.38, 1.88)).outcome, "Goal")

    def test_goal_then_persistent_event(self):
        detector = GoalDetector()
        detector.update(0, (10, 0, 0.5))
        event = detector.update(1, (12, 0, 0.5))
        self.assertIs(detector.update(2, (9, 0, 0.5)), event)

    def test_outside_then_behind_goal_cannot_score(self):
        detector = GoalDetector()
        detector.update(0, (10, 2, 0.5))
        miss = detector.update(1, (12, 2, 0.5))
        self.assertEqual(miss.outcome, "Miss")
        self.assertIs(detector.update(2, (12, 0, 0.5)), miss)

    def test_initial_ball_behind_goal(self):
        detector = GoalDetector()
        self.assertEqual(detector.update(0, (12, 0, 0.5)).outcome, "Miss")

    def test_first_exit_wins(self):
        event = self.crossing((10, 6.9, 0.5), (12, 8, 0.5))
        self.assertEqual(event.reason, "touchline_exit")
        self.assertAlmostEqual(event.position[1], 7.11)

    def test_inward_motion_does_not_score(self):
        self.assertIsNone(self.crossing((11.1, 0, 0.5), (10, 0, 0.5)))

    def test_time_must_advance(self):
        detector = GoalDetector()
        detector.update(1, (10, 0, 0.5))
        with self.assertRaises(ValueError):
            detector.update(1, (11, 0, 0.5))


@pytest.mark.parametrize("name,outcome", [("center_low", "Goal"), ("left", "Goal"),
                                          ("right", "Goal"), ("high", "Goal"),
                                          ("miss_left", "Miss"), ("miss_right", "Miss"),
                                          ("over_crossbar", "Miss")])
def test_actual_shot_outcomes(absent_model, name, outcome):
    result = run_shot(absent_model, preset_shots()[name], "absent")
    assert result["outcome"] == outcome
    assert result["goal"] == (outcome == "Goal")
    assert not result["robot_contact"]
    assert result["control_status"] == "absent"


def test_actual_robot_contact(fixed_model):
    result = run_shot(fixed_model, preset_shots(duration=1)["center_low"], "fixed")
    assert result["robot_contact"]
    assert result["control_status"] == "fixed_isolation"
    contacts = [contact for contact in result["contacts"] if contact["kind"] == "robot"]
    assert contacts
    assert any(not np.allclose(c["ball_velocity_before"], c["ball_velocity_after"]) for c in contacts)


def test_robot_contact_then_goal_is_still_goal(fixed_model):
    shot = Shot(position=(9, -0.5, 0.8), velocity=(12, 0, 1), duration=1)
    result = run_shot(fixed_model, shot, "fixed")
    assert result["goal"] and result["robot_contact"]
    assert result["outcome"] == "Goal"
    assert any(c["kind"] == "robot" and c["time"] < result["exit_event"]["time"]
               for c in result["contacts"])


def test_reset_and_seed_reproduce_entire_result(absent_model):
    shot = random_shots(1, seed=42, duration=0.2)[0]
    first = run_shot(absent_model, shot, "absent")
    run_shot(absent_model, preset_shots()["high"], "absent")
    assert run_shot(absent_model, shot, "absent") == first
    assert random_shots(1, seed=42, duration=0.2)[0] == shot


def test_actual_goal_collisions_are_recorded(absent_model):
    for name in ("post", "crossbar"):
        result = run_shot(absent_model, preset_shots(duration=0.5)[name], "absent")
        assert result["goal_contact"]
        assert not result["robot_contact"]
        assert any(c["kind"] == "goal" for c in result["contacts"])


@pytest.mark.parametrize("kwargs", [{"duration": 0}, {"duration": float("nan")},
                                    {"duration": float("inf")}, {"position": (0, 0, float("nan"))},
                                    {"velocity": (float("inf"), 0, 0)},
                                    {"angular_velocity": (0, float("nan"), 0)}])
def test_invalid_shot_input(kwargs):
    with pytest.raises(ValueError):
        Shot(**kwargs)


def test_mujoco_silent_unstable_reset_is_error(absent_model):
    with pytest.raises(RuntimeError, match="mjWARN_BADQVEL"):
        run_shot(absent_model, Shot(position=(0, 0, 1), velocity=(1e100, 0, 0), duration=0.01), "absent")


if __name__ == "__main__":
    unittest.main()
