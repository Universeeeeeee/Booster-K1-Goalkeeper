"""Deterministic current-state ball forecasts using the scene's contact physics."""
from copy import deepcopy
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def build_ball_floor_model(scene_path):
    """Copy the scene's solver, defaults, ground and ball; omit the robot and goals."""
    source = ET.parse(scene_path).getroot()
    root = ET.Element("mujoco", model="ball_floor_forecast")
    for tag in ("option", "default"):
        node = source.find(tag)
        if node is not None:
            root.append(deepcopy(node))
    world = ET.SubElement(root, "worldbody")
    ground = deepcopy(source.find("worldbody/geom[@name='ground']"))
    ball = deepcopy(source.find("worldbody/body[@name='ball']"))
    if ground is None or ball is None:
        raise ValueError("scene must contain the ground and ball")
    for geom in [ground, *ball.iter("geom")]:
        geom.attrib.pop("material", None)
    world.extend((ground, ball))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nq != 7 or model.nv != 6:
        raise ValueError("forecast model must contain one free ball")
    return model


class BallPredictor:
    """Clone a ball state into independent data and simulate at the original 1 ms step."""
    def __init__(self, model, horizon=18.):
        if model.nq != 7 or model.nv != 6 or not np.isclose(model.opt.timestep, .001, rtol=0, atol=1e-12):
            raise ValueError("ball predictor requires one free ball and a 1 ms timestep")
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("forecast horizon must be finite and positive")
        self.model, self.horizon = model, float(horizon)
        self.radius = float(model.geom_size[model.geom("ball_collision").id, 0])

    @staticmethod
    def _invalid(planes, reason, path=None, events=None):
        return {"valid": False, "reason": reason,
                "crossings": {str(plane): {"valid": False, "reason": reason, "plane_x": plane}
                              for plane in planes},
                "path": np.empty((0, 4)) if path is None else np.asarray(path),
                "floor_contact_events": [] if events is None else events}

    def forecast(self, qpos7, qvel6, planes=(11., 11.11)):
        """Receive current state only, including the quaternion and body-frame spin.

        The path is relative time and world x/y/z, sampled every 5 ms. This forecast
        describes free ball/floor dynamics and does not include robot or goal contact.
        """
        try:
            planes = tuple(dict.fromkeys(float(plane) for plane in planes))
        except (TypeError, ValueError):
            return self._invalid((), "invalid prediction planes")
        if not planes or not np.isfinite(planes).all():
            return self._invalid((), "invalid prediction planes")
        try:
            qpos, qvel = np.asarray(qpos7, dtype=float), np.asarray(qvel6, dtype=float)
        except (TypeError, ValueError):
            return self._invalid(planes, "invalid current state")
        if qpos.shape != (7,) or qvel.shape != (6,) or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            return self._invalid(planes, "current state must be finite qpos7 and qvel6")
        if not np.isclose(np.linalg.norm(qpos[3:]), 1., rtol=0, atol=1e-6):
            return self._invalid(planes, "invalid ball quaternion")
        if qvel[0] <= 0 or all(qpos[0] >= plane for plane in planes):
            return self._invalid(planes, "not approaching plane")

        model, dt = self.model, float(self.model.opt.timestep)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        path, events, episode = [[0., *data.qpos[:3]]], [], None
        crossings = {str(plane): {"valid": False, "reason": "not approaching plane", "plane_x": plane}
                     for plane in planes if qpos[0] >= plane}
        warning_counts = np.array([warning.number for warning in data.warning])
        if warning_counts.any() or not np.isfinite(data.qacc).all():
            return self._invalid(planes, "numerical warning at initial state", path)

        for step in range(1, math.ceil(self.horizon/dt)+1):
            before, before_velocity, before_time = data.qpos[:3].copy(), data.qvel[:3].copy(), float(data.time)
            mujoco.mj_step(model, data)
            if not math.isfinite(data.time) or abs(data.time-before_time-dt) > 1e-9:
                return self._invalid(planes, "numerical warning, nonfinite state or unexpected time reset", path, events)
            position = data.qpos[:3]
            if step % 5 == 0:
                if (not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc))
                        or not np.array_equal(warning_counts, [warning.number for warning in data.warning])):
                    return self._invalid(planes, "numerical warning or nonfinite state", path, events)
                path.append([float(data.time), *position])

            contact = next((contact for contact in data.contact[:data.ncon]
                            if contact.dist < 0 and contact.efc_address >= 0), None)
            if contact is not None:
                if episode is None:
                    episode = {"start_time": before_time, "end_time": float(data.time),
                               "minimum_dist_m": float(contact.dist),
                               "velocity_before": before_velocity.tolist(),
                               "velocity_after": data.qvel[:3].tolist()}
                    events.append(episode)
                episode["end_time"] = float(data.time)
                episode["minimum_dist_m"] = min(episode["minimum_dist_m"], float(contact.dist))
                episode["velocity_after"] = data.qvel[:3].tolist()
            else:
                episode = None

            for plane in planes:
                if str(plane) in crossings or not before[0] <= plane < position[0]:
                    continue
                fraction = float((plane-before[0])/(position[0]-before[0]))
                crossing = before+fraction*(position-before)
                y, z = float(crossing[1]), float(crossing[2])
                crossings[str(plane)] = {"valid": True, "plane_x": plane,
                    "time_to_plane_s": before_time+fraction*dt, "lateral_m": y, "height_m": z,
                    "inside_aperture": bool(abs(y)+self.radius <= 1.5 and self.radius-.002 <= z <= 2.-self.radius),
                    "model": "ball_floor_mujoco", "future_floor_contact_unmodeled": False}
            if len(crossings) == len(planes):
                break

        if (not all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc))
                or not np.array_equal(warning_counts, [warning.number for warning in data.warning])):
            return self._invalid(planes, "numerical warning or nonfinite state", path, events)
        if path[-1][0] != float(data.time):
            path.append([float(data.time), *data.qpos[:3]])
        for plane in planes:
            if str(plane) not in crossings:
                crossings[str(plane)] = {"valid": False, "reason": "no crossing within horizon", "plane_x": plane}
        valid = all(prediction["valid"] for prediction in crossings.values())
        return {"valid": valid, "reason": None if valid else "one or more requested planes unavailable",
                "crossings": crossings, "path": np.asarray(path), "floor_contact_events": events}
