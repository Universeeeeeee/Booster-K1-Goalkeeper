"""Whole-ball goal and pitch-exit detection in the supplied world coordinates."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExitEvent:
    outcome: str
    reason: str
    time: float
    position: tuple[float, float, float]
    goal: str | None = None

    def to_dict(self):
        return asdict(self)


class GoalDetector:
    """Latch the first full-ball exit, interpolating between consecutive samples.

    A ball starting fully outside the pitch is a miss. Once a miss is latched,
    moving into the aperture behind the goal cannot become a goal. The small
    floor tolerance permits normal soft-contact penetration at the ground.
    """

    def __init__(self, radius=0.11, tolerance=1e-6, floor_tolerance=0.002):
        if radius <= 0 or tolerance < 0 or floor_tolerance < 0:
            raise ValueError("Radius must be positive and tolerances nonnegative")
        self.radius = radius
        self.tolerance = tolerance
        self.floor_tolerance = floor_tolerance
        self.previous = None
        self.event = None

    def update(self, time, position):
        position = tuple(float(v) for v in position)
        if self.event is not None:
            return self.event
        if self.previous is None:
            self.previous = (time, position)
            if (abs(position[0]) > 11 + self.radius + self.tolerance
                    or abs(position[1]) > 7 + self.radius + self.tolerance):
                self.event = ExitEvent("Miss", "initial_ball_outside_pitch", time, position)
            return self.event

        old_time, old = self.previous
        if time <= old_time:
            raise ValueError("Sample times must increase")
        candidates = []
        for axis, limit in ((0, 11 + self.radius), (1, 7 + self.radius)):
            for sign in (-1, 1):
                start, end = sign * old[axis], sign * position[axis]
                if start <= limit + self.tolerance and end > limit + self.tolerance:
                    fraction = max(0.0, (limit - start) / (end - start))
                    crossing = tuple(a + fraction * (b - a) for a, b in zip(old, position))
                    candidates.append((fraction, axis, sign, crossing))
        if candidates:
            fraction, axis, sign, crossing = min(candidates, key=lambda item: item[0])
            event_time = old_time + fraction * (time - old_time)
            if axis == 1:
                self.event = ExitEvent("Miss", "touchline_exit", event_time, crossing)
            else:
                goal = "right" if sign == 1 else "left"
                inside = (abs(crossing[1]) + self.radius <= 1.5 + self.tolerance
                          and crossing[2] + self.radius <= 2 + self.tolerance
                          and crossing[2] - self.radius >= -self.floor_tolerance)
                self.event = ExitEvent("Goal" if inside else "Miss",
                                       "whole_ball_goal" if inside else "outside_goal_aperture",
                                       event_time, crossing, goal)
        self.previous = (time, position)
        return self.event
