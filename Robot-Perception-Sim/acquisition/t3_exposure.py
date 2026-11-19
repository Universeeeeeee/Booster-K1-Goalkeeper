"""Candidate T3 linear-light exposure reference; no hardware ISP claim."""
import numpy as np


def exposure_schedule(frame_start_s, exposure_s, fps=100, samples=4):
    if not 0 < exposure_s <= 1 / fps or samples < 1:
        raise ValueError('Exposure must fit the frame period; samples must be positive')
    return frame_start_s + exposure_s * (np.arange(samples) + .5) / samples


def integrate_linear(samples, exposure_s, reference_s=.002):
    """Midpoint quadrature on fresh scene-linear RGB samples, without clipping."""
    values = np.asarray(samples)
    if (values.ndim != 4 or values.shape[-1] != 3 or values.shape[0] == 0
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all() or np.any(values < 0)
            or exposure_s <= 0 or reference_s <= 0):
        raise ValueError('Expected finite nonnegative floating [sample,H,W,RGB]')
    return values.mean(axis=0) * (exposure_s / reference_s)


def next_shared_exposure(left, right, current_s, target=.45,
                         minimum_s=.0001, maximum_s=.008, max_step_ratio=2.):
    """Meter exposed linear RGB, jointly; apply the returned duration next frame.

    Candidate exposure-only AE: fixed gain, no gamma, no per-eye normalization.
    Diagnostic pixels must be excluded from the supplied metering crops.
    """
    means = []
    for image in (left, right):
        image = np.asarray(image)
        if (image.ndim != 3 or image.shape[-1] != 3 or image.size == 0
                or not np.isfinite(image).all() or np.any(image < 0)):
            raise ValueError('Expected finite nonnegative RGB metering crop')
        means.append(float((image @ np.array([.2126, .7152, .0722])).mean()))
    brightness = sum(means) / 2
    ratio = max_step_ratio if brightness == 0 else target / brightness
    ratio = np.clip(ratio, 1 / max_step_ratio, max_step_ratio)
    return float(np.clip(current_s * ratio, minimum_s, maximum_s))


def next_center_exposure(image, current_s, **parameters):
    """Single center RGB metering; identical-eye arithmetic shares the AE core."""
    return next_shared_exposure(image, image, current_s, **parameters)


def next_center_fast_exposure(image, current_s, target=.45, minimum_s=.0001,
                              maximum_s=.008):
    """Causal center-only AE: up to 8x shorter, at most 2x longer next frame."""
    image=np.asarray(image)
    if image.ndim!=3 or image.shape[-1]!=3 or image.size==0 or not np.isfinite(image).all() or np.any(image<0):
        raise ValueError('Expected finite nonnegative RGB metering crop')
    mean=float((image @ np.array([.2126,.7152,.0722])).mean())
    ratio=2. if mean==0 else target/mean
    return float(np.clip(current_s*np.clip(ratio,.125,2.),minimum_s,maximum_s))
