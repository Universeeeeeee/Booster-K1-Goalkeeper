"""Reference SD/TD operators for the T3 stereo experiments."""

import numpy as np


TD_ENCODING = "int8_half_trunc_current_minus_previous_first_zero"


def gray_from_hdr(hdr):
    """Convert scene-linear RGB(A) HDR data to linear luminance."""
    image = np.asarray(hdr)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("HDR image must have shape HxWx3 or HxWx4")
    rgb = image[..., :3].astype(np.float32)
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def sd(gray):
    """Return half-scale SD truncated toward zero and biased by 128."""
    image = np.asarray(gray, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("gray image must be two-dimensional")
    up = np.zeros_like(image)
    right = np.zeros_like(image)
    up[1:] = image[1:] - image[:-1]
    right[:, :-1] = image[:, :-1] - image[:, 1:]
    return (np.trunc(up * 0.5) + 128).astype(np.int16), (np.trunc(right * 0.5) + 128).astype(np.int16)


def td(current, previous):
    """Signed half-difference of two 8-bit grayscale samples from the same eye."""
    current = np.asarray(current, dtype=np.float32)
    if current.ndim != 2:
        raise ValueError("gray image must be two-dimensional")
    if previous is None:
        return np.zeros(current.shape, dtype=np.int8)
    previous = np.asarray(previous, dtype=np.float32)
    if current.shape != previous.shape:
        raise ValueError("TD inputs must be same-shaped grayscale images")
    return np.trunc((current - previous) * 0.5).astype(np.int8)
