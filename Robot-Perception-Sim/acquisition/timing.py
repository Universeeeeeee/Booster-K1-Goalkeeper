"""Exact sampling times; convert to simulator floats only at the state boundary.

RGB and IMU call these functions independently, using their own rate and index.
The ceil policy samples the first physics state at/after the target, with a lag
strictly less than one physics step. It is not state interpolation. Exposure
times describe one global shutter window shared by both eyes.
"""

from fractions import Fraction


def _fraction(value, name):
    # Decimal text avoids importing binary float rounding into the schedule.
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        return value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc


def sample_time(index, rate_hz, origin_s=0):
    """Return origin + index/rate exactly, without accumulating rounded periods."""
    if type(index) is not int or index < 0:
        raise ValueError("index must be a nonnegative integer")
    rate = _fraction(rate_hz, "rate_hz")
    origin = _fraction(origin_s, "origin_s")
    if rate <= 0 or origin < 0:
        raise ValueError("rate_hz must be positive and origin_s nonnegative")
    return origin + index / rate


def scheduled_step(index, rate_hz, physics_dt_s="0.001", origin_s=0):
    """Return a target and its ceil state on the physics grid starting at t=0.

    The returned lag is explicit so callers cannot label a sampled state as the
    exact target. A non-grid sampling origin remains independent of that grid.
    """
    target = sample_time(index, rate_hz, origin_s)
    dt = _fraction(physics_dt_s, "physics_dt_s")
    if dt <= 0:
        raise ValueError("physics_dt_s must be positive")
    ticks = target / dt
    step = -(-ticks.numerator // ticks.denominator)
    state = step * dt
    return {"step_index": step, "scheduled_s": target,
            "state_s": state, "lag_s": state - target}


def exposure_window(start_s, exposure_ms, fps):
    """One shared left/right window; zero is the instantaneous baseline.

    Positive exposures must be strictly shorter than the frame period, as G3
    requires. The returned midpoint is an exposure reference, not a new clock.
    """
    start = sample_time(0, fps, start_s)
    exposure = _fraction(exposure_ms, "exposure_ms") / 1000
    period = sample_time(1, fps)
    if exposure < 0 or exposure >= period:
        raise ValueError("exposure must be nonnegative and strictly shorter than the frame period")
    return {"start_s": start, "end_s": start + exposure,
            "mid_s": start + exposure / 2}


def rational_seconds(value):
    """JSON-safe exact seconds; retain this alongside any display float."""
    value = _fraction(value, "seconds")
    return {"numerator": value.numerator, "denominator": value.denominator}


def exposure_sample_times(start_s, exposure_ms, fps, samples):
    """Exact midpoint quadrature times shared by both eyes, including zero baseline."""
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    window = exposure_window(start_s, exposure_ms, fps)
    duration = window["end_s"] - window["start_s"]
    if duration == 0:
        return (window["start_s"],)
    return tuple(window["start_s"] + Fraction(2*i + 1, 2*samples) * duration for i in range(samples))
