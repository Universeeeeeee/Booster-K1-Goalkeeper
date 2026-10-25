from fractions import Fraction
import json

import pytest

from acquisition.timing import exposure_window, rational_seconds, sample_time, scheduled_step


@pytest.mark.parametrize("rate", [60, 120, 200, Fraction(30000, 1001)])
def test_long_horizon_schedule_has_no_cumulative_drift(rate):
    # Check a billionth sample directly: independent of loop count or float dt.
    index = 10**9
    origin = Fraction(7, 3)
    assert sample_time(index, rate, origin) == origin + Fraction(index, 1) / rate
    assert sample_time(index + 1, rate, origin) - sample_time(index, rate, origin) == 1 / Fraction(rate)
    for index in [0, 1, 2, 3, 999999999, 10**9]:
        scheduled = scheduled_step(index, rate, origin_s=origin)
        assert scheduled["scheduled_s"] == origin + Fraction(index, 1) / rate
        assert scheduled["state_s"] == scheduled["step_index"] * Fraction(1, 1000)
        assert 0 <= scheduled["lag_s"] < Fraction(1, 1000)


def test_camera_and_imu_have_independent_schedules():
    assert [scheduled_step(i, 120)["step_index"] for i in range(7)] == [0, 9, 17, 25, 34, 42, 50]
    assert [scheduled_step(i, 60)["step_index"] for i in range(4)] == [0, 17, 34, 50]
    assert [scheduled_step(i, 200)["step_index"] for i in range(11)] == list(range(0, 51, 5))
    assert sample_time(6, 120) == sample_time(3, 60) == sample_time(10, 200)


@pytest.mark.parametrize("fps,exposure_ms", [(120, 1), (120, 3), (120, 5), (60, 1), (60, 3), (60, 5), (60, 10)])
def test_global_exposure_start_end_mid(fps, exposure_ms):
    start = sample_time(17, fps)
    shared = exposure_window(start, exposure_ms, fps)
    assert shared["start_s"] == start
    assert shared["end_s"] == start + Fraction(exposure_ms, 1000)
    assert shared["mid_s"] == start + Fraction(exposure_ms, 2000)
    assert shared["end_s"] < sample_time(18, fps)


def test_instantaneous_baseline_and_exact_serialization():
    window = exposure_window(Fraction(1, 120), 0, 120)
    assert set(window.values()) == {Fraction(1, 120)}
    encoded = json.loads(json.dumps(rational_seconds(window["mid_s"])))
    assert Fraction(encoded["numerator"], encoded["denominator"]) == Fraction(1, 120)
    assert sample_time(1, 120, 0.1) == Fraction(13, 120)


@pytest.mark.parametrize("index", [-1, 1.0, True, "1"])
def test_invalid_sample_indices(index):
    with pytest.raises(ValueError, match="index"):
        sample_time(index, 120)


@pytest.mark.parametrize("rate", [0, -60, float("nan"), float("inf"), True, "invalid"])
def test_invalid_rates(rate):
    with pytest.raises(ValueError):
        sample_time(0, rate)


@pytest.mark.parametrize("dt", [0, -0.001, float("nan"), float("inf"), True])
def test_invalid_physics_steps(dt):
    with pytest.raises(ValueError):
        scheduled_step(0, 120, dt)


@pytest.mark.parametrize("origin", [-1, float("nan"), float("inf"), True])
def test_invalid_origin(origin):
    with pytest.raises(ValueError):
        sample_time(0, 120, origin)


@pytest.mark.parametrize("exposure,fps", [(-1, 120), (10, 120), (Fraction(25, 3), 120),
                                        (Fraction(50, 3), 60), (float("nan"), 60),
                                        (float("inf"), 60), (True, 60)])
def test_invalid_exposures_including_equal_period(exposure, fps):
    with pytest.raises(ValueError):
        exposure_window(0, exposure, fps)


def test_camera_configuration_rejects_equal_period(tmp_path, config):
    from robot_perception_sim.camera import load_camera_config
    invalid = {**config, "reference_fps": 100, "exposure_time_ms": 10}
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="strictly shorter"):
        load_camera_config(path)


def test_calibration_hash_is_verifiable_and_independent_of_exposure(config):
    import hashlib
    from robot_perception_sim.camera import calibration_record
    record = calibration_record(config)
    payload = json.dumps(record["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(payload.encode()).hexdigest() == record["sha256"]
    assert record == calibration_record({**config, "exposure_time_ms": 0})
    assert record["source"] == "nominal_model"


def test_recording_checks_actual_frame_period_before_simulation(tmp_path, config):
    from robot_perception_sim.recording import record_shot
    with pytest.raises(ValueError, match="strictly shorter"):
        record_shot(None, tmp_path, {**config, "reference_fps": 60, "exposure_time_ms": 10}, fps=100)


def test_capture_records_actual_subframe_poses_and_clocks_without_stepping(monkeypatch, config):
    import mujoco
    import numpy as np
    from robot_perception_sim.stereo import StereoCamera

    # The parent render matrix verifies pixels. This test independently checks
    # timestamps/kinematic state passed to the renderer without a GL context.
    rendered = []

    class Renderer:
        def __init__(self, *args, **kwargs):
            pass

        def update_scene(self, data, camera):
            rendered.append((float(data.time), camera, data.qpos.copy()))

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, "Renderer", Renderer)
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size="0.01"/>
      <camera name="zedx_left_camera" pos="-0.06 0 0"/>
      <camera name="zedx_right_camera" pos="0.06 0 0"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.time = 2
    data.qvel[0] = 3
    mujoco.mj_forward(model, data)
    before = {key: getattr(data, key).copy() for key in ("qpos", "qvel", "qacc", "ctrl")}
    with StereoCamera(model, {**config, "exposure_time_ms": 5, "exposure_samples": 3}) as camera:
        frame = camera.capture(data)
    assert frame["exposure_mid_s"] == pytest.approx(2.0025)
    assert frame["left_timestamp_s"] == frame["right_timestamp_s"] == frame["t_sim_state_s"] == 2
    assert frame["rolling_skew_s"] == 0
    assert frame["sdk_timestamp"] is None
    assert frame["clock_domains"]["sensor_sdk"] is None
    assert frame["clock_domains"]["receive_processing"] == "host_monotonic_ns"
    assert frame["capture_end_monotonic_ns"] >= frame["capture_start_monotonic_ns"] > 0
    assert frame["clock_mapping"]["host_monotonic_from_sim_state"] is None
    assert frame["exposure_model"] == "constant_qvel_encoded_rgb_average"
    poses = frame["sample_T_world_from_optical"]
    assert poses.shape == (3, 2, 4, 4)
    for i, timestamp in enumerate(frame["sample_timestamps_s"]):
        assert rendered[2*i][0] == rendered[2*i+1][0] == timestamp
        np.testing.assert_array_equal(rendered[2*i][2], rendered[2*i+1][2])
        np.testing.assert_allclose(poses[i, :, 0, 3], [-0.06 + 3*(timestamp-2), 0.06 + 3*(timestamp-2)])
    for key, expected in before.items():
        np.testing.assert_array_equal(getattr(data, key), expected)
    assert data.time == 2
