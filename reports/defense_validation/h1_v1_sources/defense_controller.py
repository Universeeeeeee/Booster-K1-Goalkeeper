"""H1 only: smooth arm residual after the unchanged official target filter."""
from dataclasses import dataclass
import math

import torch


ARM_JOINTS = (
    "aaleft_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "aaright_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
)


@dataclass(frozen=True)
class DefensePulse:
    joint_offsets: tuple
    start_s: float = 3.
    rise_s: float = 2.
    hold_s: float = 2.
    return_s: float = 2.

    def __post_init__(self):
        values = [self.start_s, self.rise_s, self.hold_s, self.return_s]
        if not all(math.isfinite(v) for v in values) or self.start_s < 0 or min(values[1:]) <= 0:
            raise ValueError("pulse timing must be finite, with nonnegative start and positive durations")
        names = [name for name, _ in self.joint_offsets]
        if len(set(names)) != len(names) or any(name not in ARM_JOINTS for name in names):
            raise ValueError("pulse needs unique named arm joints")
        if any(not math.isfinite(value) or abs(value) > .30 for _, value in self.joint_offsets):
            raise ValueError("arm residual must be finite and at most 0.30 rad")

    @staticmethod
    def _quintic(u):
        return u**3 * (10 + u * (-15 + 6*u))

    def factor(self, time):
        if not math.isfinite(time):
            raise ValueError("pulse time must be finite")
        elapsed = time - self.start_s
        if elapsed <= 0 or elapsed >= self.rise_s + self.hold_s + self.return_s:
            return 0.
        if elapsed < self.rise_s:
            return self._quintic(elapsed / self.rise_s)
        if elapsed < self.rise_s + self.hold_s:
            return 1.
        return 1 - self._quintic((elapsed - self.rise_s - self.hold_s) / self.return_s)

    def offset(self, time):
        factor = self.factor(time)
        return {name: value * factor for name, value in self.joint_offsets}


class PostFilteredArmController:
    """Instance wrapper; call once at each original policy tick, then original PD."""
    def __init__(self, controller, pulse):
        self.controller, self.pulse = controller, pulse
        self.original_compute_targets = controller.compute_targets
        self.joint_indices = {name: controller.cfg.robot.joint_names.index(name)
                              for name, _ in pulse.joint_offsets}
        self.reset()

    def reset(self):
        # Reset our telemetry only. The original policy owns its reset/history.
        self.nominal_targets = self.delivered_targets = self.offset = None

    def compute_targets(self):
        nominal = self.original_compute_targets()
        if nominal.shape != (22,) or not torch.isfinite(nominal).all():
            raise RuntimeError("H1 requires 22 finite official joint targets")
        delivered = nominal.clone()
        offset = torch.zeros_like(nominal)
        for name, value in self.pulse.offset(float(self.controller.data.time)).items():
            index = self.joint_indices[name]
            offset[index] = value
            if value != 0:
                delivered[index] += offset[index]
        self.nominal_targets = nominal.detach().clone()
        self.delivered_targets = delivered.detach().clone()
        self.offset = offset.detach().clone()
        return delivered
