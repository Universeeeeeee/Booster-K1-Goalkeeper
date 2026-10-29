"""Frozen shot loading and explicitly non-physical diagnostic replay sources."""
import csv
import hashlib
import json
import math
from pathlib import Path

SOURCES = {"physx_physical_shot", "procedural_kinematic", "measured_csv_replay"}


def validate_shot(shot, radius=.11):
    for key in ("position", "velocity", "angular_velocity"):
        if len(shot[key]) != 3 or not all(math.isfinite(v) for v in shot[key]):
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(shot["duration"]) or shot["duration"] <= 0:
        raise ValueError("Invalid duration")
    x, y, z = shot["position"]
    if z < radius:
        raise ValueError("Initial ground penetration")
    # All golden launches stay inside a frame-free interior slab. This is a
    # conservative guard, not a general collision detector for arbitrary USD.
    if abs(x) > 10 or abs(y) > 6:
        raise ValueError("Initial position outside audited frame-free launch slab")


def load_catalog(path):
    value = json.loads(Path(path).read_text())
    names = [shot["name"] for shot in value["cases"]]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate case names")
    for shot in value["cases"]:
        validate_shot(shot, value["ball_radius_m"])
    if value["source"] not in SOURCES:
        raise ValueError("Unknown source")
    return value


def procedural_samples(shot, timestep_s):
    """Ballistic commanded poses; never claim collision response or contacts."""
    validate_shot(shot)
    if not math.isfinite(timestep_s) or timestep_s <= 0:
        raise ValueError("Timestep must be positive")
    for index in range(math.ceil(shot["duration"] / timestep_s) + 1):
        t = min(index * timestep_s, shot["duration"])
        position = [p + v*t for p, v in zip(shot["position"], shot["velocity"])]
        position[2] -= .5 * 9.81 * t*t
        yield {"time": t, "position": position,
               "velocity": [*shot["velocity"][:2], shot["velocity"][2] - 9.81*t]}


def measured_csv_samples(path):
    """Read measured world-center samples, without filling gaps or contacts.

    Caller owns calibration and source authenticity. Merely choosing this source
    does not establish that a CSV was recorded from a real ball.
    """
    previous = -math.inf
    samples = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            values = [float(row[key]) for key in ("time_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")]
            if not all(map(math.isfinite, values)) or values[0] <= previous:
                raise ValueError("CSV requires finite values and strictly increasing times")
            previous = values[0]
            samples.append({"time": values[0], "position": values[1:4], "velocity": values[4:7]})
    if not samples:
        raise ValueError("Empty CSV")
    return samples, {"source": "measured_csv_replay", "path": str(path),
                     "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     "measurement_authenticity": "caller_supplied_not_independently_verified",
                     "physical_collision_response": False}
