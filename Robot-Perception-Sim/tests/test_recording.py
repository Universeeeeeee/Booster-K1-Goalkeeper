import numpy as np
import pytest

from robot_perception_sim.recording import capture_schedule, find_ffmpeg


def test_60hz_sampling_has_bounded_jitter_without_drift():
    times = np.arange(0, 3.02, 0.001)
    targets, indices = capture_schedule(times, 2.0, 1.0, 60)
    assert len(indices) == 60
    np.testing.assert_array_equal(indices[:4], [2000, 2017, 2034, 2050])
    error = times[indices] - targets
    assert error.min() >= -1e-9
    assert error.max() < 0.001
    assert len(set(np.diff(indices))) == 2  # Alternate 16/17 ms, rather than drift at 17 ms.


@pytest.mark.parametrize("duration,fps", [(0, 60), (-1, 60), (1, 0), (1, float("nan"))])
def test_reject_invalid_recording_schedule(duration, fps):
    with pytest.raises(ValueError):
        capture_schedule(np.arange(0, 2, 0.001), 0, duration, fps)


def test_no_extrapolation_beyond_physics_trace():
    with pytest.raises(ValueError, match="cover"):
        capture_schedule(np.arange(0, 0.1, 0.001), 0, 1, 60)


def test_no_duplicate_physics_frames():
    with pytest.raises(ValueError, match="sampling rate"):
        capture_schedule(np.arange(0, 1, 0.01), 0, 0.1, 120)


def test_encoder_missing_is_an_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="FFmpeg not found"):
        find_ffmpeg(tmp_path / "does-not-exist")
