"""Small MuJoCo stereo acquisition interface; not an emulation of the ZED SDK."""

from copy import copy

import mujoco
import numpy as np

from .camera import CAMERA_NAMES


class StereoCamera:
    def __init__(self, model, config):
        for name in CAMERA_NAMES:
            model.camera(name)
        self.model = model
        self.exposure_s = config["exposure_time_ms"] / 1000
        self.samples = config["exposure_samples"] if self.exposure_s else 1
        self.renderer = mujoco.Renderer(model, height=config["image_height_px"],
                                        width=config["image_width_px"])

    def capture(self, data):
        """Integrate a global exposure using constant qvel, without stepping caller state."""
        start = float(data.time)
        frame = {"timestamp_s": start, "exposure_start_s": start,
                 "exposure_end_s": start + self.exposure_s, "exposure_samples": self.samples}
        sums = {}
        sample_data = copy(data)
        for sample in range(self.samples):
            offset = (sample + 0.5) * self.exposure_s / self.samples
            sample_data.qpos[:] = data.qpos
            sample_data.time = start + offset
            mujoco.mj_integratePos(self.model, sample_data.qpos, data.qvel, offset)
            mujoco.mj_forward(self.model, sample_data)
            for side, name in zip(("left", "right"), CAMERA_NAMES):
                self.renderer.update_scene(sample_data, camera=name)
                rgb = self.renderer.render().astype(np.float32)
                if side not in sums:
                    sums[side] = rgb
                else:
                    sums[side] += rgb
        for side, total in sums.items():
            frame[side] = np.rint(total / self.samples).clip(0, 255).astype(np.uint8)
        return frame

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
