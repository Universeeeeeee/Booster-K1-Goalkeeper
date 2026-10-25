"""Feed completed exposure buffers through the installed official ZED bridge.

v3.2's worker unconditionally copies CUDA-to-host, despite its node docs also
mentioning host pointers. Its asynchronous queue retains raw pointers only.
Keep every published device allocation immutable and alive until stream teardown.
"""

import hashlib
import math

import numpy as np


def average_rgba(subframes):
    """Equal-weight encoded uint8 RGBA average, rounded to nearest even integer.

    This is an encoded-image approximation, not linear-light radiometry or ISP
    emulation. The caller must supply paired subframes from the same times.
    """
    frames = np.asarray(subframes)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 4 or any(n == 0 for n in frames.shape):
        raise ValueError("subframes must be a nonempty uint8 array of shape [sample,height,width,4]")
    return np.rint(frames.mean(axis=0, dtype=np.float64)).astype(np.uint8)


class ExposureIntegrator:
    def __init__(self, annotator, timestamp_node, device="cuda:0"):
        import warp as wp

        if not isinstance(device, str) or not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("the official bridge requires an explicit CUDA device, e.g. cuda:0")
        self.wp = wp
        self.device = device
        self.node = annotator.zed_
        self.retained_buffers = []
        self.hold()
        for side, source in (("Left", annotator.left_rgb_annot), ("Right", annotator.right_rgb_annot)):
            for field in ("dataPtr", "bufferSize"):
                source.get_node().get_attribute(f"outputs:{field}").disconnect(
                    self.node.get_attribute(f"inputs:{field}{side}"), True)
        timestamp_node.get_attribute("outputs:result").disconnect(
            self.node.get_attribute("inputs:simulationTime"), True)

    def hold(self):
        """Suppress enqueue without closing the initialized official streamer."""
        self.node.get_attribute("inputs:stream").set(False)

    def publish(self, left_np, right_np, stamp_s):
        """Arm one official execution; caller renders once, then calls hold().

        Inputs are completed, synchronized encoded RGBA averages. No rendering
        or SDK emulation occurs here. The installed bridge controls transport
        and pixel format; verify delivered SDK colors separately.
        """
        if isinstance(stamp_s, bool) or not math.isfinite(stamp_s) or stamp_s < 1:
            raise ValueError("stamp_s must be finite and at least the bridge warmup time of 1 second")
        width = self.node.get_attribute("inputs:width").get()
        height = self.node.get_attribute("inputs:height").get()
        arrays = (left_np, right_np)
        if any(not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.shape != (height, width, 4)
               for a in arrays):
            raise ValueError("both images must be uint8 RGBA matching the bridge resolution")
        host = [np.ascontiguousarray(a) for a in arrays]
        buffers = [self.wp.array(a.reshape(-1), dtype=self.wp.uint8, device=self.device, copy=True) for a in host]
        self.wp.synchronize_device(self.device)
        # Retain before exposing pointers to the asynchronous C++ worker.
        self.retained_buffers.append(tuple(buffers))
        for side, array, buffer in zip(("Left", "Right"), host, buffers):
            self.node.get_attribute(f"inputs:dataPtr{side}").set(int(buffer.ptr))
            self.node.get_attribute(f"inputs:bufferSize{side}").set(int(array.nbytes))
        self.node.get_attribute("inputs:simulationTime").set(float(stamp_s))
        self.node.get_attribute("inputs:stream").set(True)
        return {"source_sha256_left_right": [hashlib.sha256(a.tobytes()).hexdigest() for a in host],
                "bytes_per_eye": host[0].nbytes, "timestamp_s": float(stamp_s),
                "buffer_device": self.device, "exposure_model": "encoded_rgba_average"}
