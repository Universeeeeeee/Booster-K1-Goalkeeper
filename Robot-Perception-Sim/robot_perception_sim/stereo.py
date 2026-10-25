"""Small MuJoCo stereo acquisition interface; not an emulation of the ZED SDK."""

from copy import copy
from time import monotonic_ns

import mujoco
import numpy as np

from acquisition.timing import exposure_window
from .camera import CAMERA_NAMES, calibration_record, optical_pose


class StereoCamera:
    def __init__(self, model, config):
        for name in CAMERA_NAMES:
            model.camera(name)
        self.model = model
        exposure_window(0, config["exposure_time_ms"], config["reference_fps"])
        self.calibration = calibration_record(config)
        self.exposure_s = config["exposure_time_ms"] / 1000
        self.samples = config["exposure_samples"] if self.exposure_s else 1
        self.renderer = mujoco.Renderer(model, height=config["image_height_px"],
                                        width=config["image_width_px"])

    def capture(self, data):
        """Integrate a global exposure using constant qvel, without stepping caller state."""
        host_start = monotonic_ns()
        start = float(data.time)
        frame = {"timestamp_s": start, "t_sim_state_s": start, "exposure_start_s": start,
                 "exposure_end_s": start + self.exposure_s, "exposure_mid_s": start + self.exposure_s / 2,
                 "left_timestamp_s": start, "right_timestamp_s": start,
                 "exposure_samples": self.samples, "rolling_skew_s": 0,
                 "exposure_model": "constant_qvel_encoded_rgb_average" if self.exposure_s else "instantaneous_render",
                 "sample_timestamps_s": [], "sample_T_world_from_optical": [],
                 "sdk_timestamp": None, "capture_start_monotonic_ns": host_start,
                 "clock_domains": {"sim_state": "mujoco_simulation_seconds",
                                   "exposure": "mujoco_simulation_seconds", "sensor_sdk": None,
                                   "receive_processing": "host_monotonic_ns"},
                 "clock_mapping": {"exposure_from_sim_state_offset_s": 0,
                                   "host_monotonic_from_sim_state": None},
                 "calibration_sha256": self.calibration["sha256"],
                 "calibration_source": self.calibration["source"], "pose_source": "sim_oracle"}
        sums = {}
        sample_data = copy(data)
        for sample in range(self.samples):
            offset = (sample + 0.5) * self.exposure_s / self.samples
            sample_data.qpos[:] = data.qpos
            sample_data.time = start + offset
            mujoco.mj_integratePos(self.model, sample_data.qpos, data.qvel, offset)
            mujoco.mj_forward(self.model, sample_data)
            frame["sample_timestamps_s"].append(float(sample_data.time))
            frame["sample_T_world_from_optical"].append([optical_pose(sample_data, name) for name in CAMERA_NAMES])
            for side, name in zip(("left", "right"), CAMERA_NAMES):
                self.renderer.update_scene(sample_data, camera=name)
                rgb = self.renderer.render().astype(np.float32)
                if side not in sums:
                    sums[side] = rgb
                else:
                    sums[side] += rgb
        for side, total in sums.items():
            frame[side] = np.rint(total / self.samples).clip(0, 255).astype(np.uint8)
        frame["sample_T_world_from_optical"] = np.asarray(frame["sample_T_world_from_optical"])
        frame["capture_end_monotonic_ns"] = monotonic_ns()
        return frame

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
