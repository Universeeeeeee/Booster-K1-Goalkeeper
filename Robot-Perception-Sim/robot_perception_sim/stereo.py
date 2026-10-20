"""Small MuJoCo stereo acquisition interface; not an emulation of the ZED SDK."""

import mujoco

from .camera import CAMERA_NAMES


class StereoCamera:
    def __init__(self, model, config):
        for name in CAMERA_NAMES:
            model.camera(name)
        self.renderer = mujoco.Renderer(model, height=config["image_height_px"],
                                        width=config["image_width_px"])

    def capture(self, data):
        """Render both eyes at the current state, without advancing simulation time."""
        frame = {"timestamp_s": float(data.time)}
        for side, name in zip(("left", "right"), CAMERA_NAMES):
            self.renderer.update_scene(data, camera=name)
            frame[side] = self.renderer.render().copy()
        return frame

    def close(self):
        self.renderer.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
