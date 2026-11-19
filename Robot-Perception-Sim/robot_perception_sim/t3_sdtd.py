"""Reference SD/TD operators for the T3 stereo experiments."""

import numpy as np


def gray_from_hdr(hdr):
    """Convert scene-linear RGB(A) HDR data to linear luminance."""
    image = np.asarray(hdr)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("HDR image must have shape HxWx3 or HxWx4")
    rgb = image[..., :3].astype(np.float32)
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def sd(gray):
    """Return signed vertical-up and horizontal-right differences."""
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("gray image must be two-dimensional")
    up = np.zeros_like(image)
    right = np.zeros_like(image)
    up[1:] = image[1:] - image[:-1]
    right[:, :-1] = image[:, :-1] - image[:, 1:]
    return up, right


def td(current, previous):
    current = np.asarray(current, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError("TD inputs must be same-shaped grayscale images")
    return current - previous
