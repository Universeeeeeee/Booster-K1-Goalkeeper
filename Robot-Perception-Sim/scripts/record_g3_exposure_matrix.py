#!/usr/bin/env python3
"""Exercise the existing stereo capture on static/ball/camera motion fixtures.

This is an offline constant-velocity render experiment, not a new physics
integrator or a substitute for SDK capture. Replay states remain separate.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from acquisition.timing import scheduled_step
from robot_perception_sim.camera import load_camera_config, intrinsics, vertical_fov_deg
from robot_perception_sim.stereo import StereoCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT / 'results'):
        parser.error('output must be a new directory inside Robot-Perception-Sim/results')
    output.mkdir(parents=True, exist_ok=False)
    config = {**load_camera_config(), 'image_width_px': 320, 'image_height_px': 200}
    xml = f'''<mujoco><option timestep="0.001"/>
      <visual><global offwidth="320" offheight="200"/></visual>
      <asset><material name="red" rgba="1 0 0 1" emission="1"/></asset>
      <worldbody>
        <body name="camera_rig"><freejoint name="camera_free"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <camera name="zedx_left_camera" pos="-0.06 0 0" quat="0 1 0 0" fovy="{vertical_fov_deg(config)}"/>
          <camera name="zedx_right_camera" pos="0.06 0 0" quat="0 1 0 0" fovy="{vertical_fov_deg(config)}"/>
        </body>
        <body name="ball" pos="0 0 2"><freejoint name="ball_free"/>
          <geom type="sphere" size="0.04" material="red"/>
        </body>
      </worldbody></mujoco>'''
    (output / 'scene.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    calibration = {'K': intrinsics(config).tolist(), 'baseline_m': .12, 'resolution': [320, 200]}
    calibration_sha = hashlib.sha256(json.dumps(calibration, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    metadata = {'source': 'mujoco_existing_stereo_capture', 'calibration': calibration,
                'calibration_sha256': calibration_sha, 'physics_dt_s': .001,
                'exposure_model': 'constant_qvel_encoded_rgb_average', 'mujoco_version': mujoco.__version__,
                'motion_source': 'diagnostic kinematic state replay, no controller and no mj_step',
                'capture_source_sha256': hashlib.sha256((PROJECT / 'robot_perception_sim/stereo.py').read_bytes()).hexdigest(),
                'scene_sha256': hashlib.sha256(xml.encode()).hexdigest(),
                'sdk_timestamp': None, 'sdk_used': False,
                'limitations': 'Uniform encoded RGB average at midpoint samples; not linear radiance, sensor ISP, contact or acceleration integration.'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    states, velocities, times, sample_poses, sample_times, rows = [], [], [], [], [], []
    cases = [(fps, ms) for fps, values in ((60, (1, 3, 5, 10)), (120, (1, 3, 5))) for ms in values]
    for fps, exposure_ms in cases:
        cfg = {**config, 'reference_fps': fps, 'exposure_time_ms': exposure_ms, 'exposure_samples': 20}
        with ExitStack() as stack:
            instant = stack.enter_context(StereoCamera(model, {**cfg, 'exposure_time_ms': 0}))
            exposed = stack.enter_context(StereoCamera(model, cfg))
            for motion in ('static', 'ball', 'camera'):
                for index in range(3):
                    schedule = scheduled_step(index, fps)
                    data = mujoco.MjData(model)
                    data.time = float(schedule['state_s'])
                    if motion != 'static':
                        joint = model.joint('ball_free' if motion == 'ball' else 'camera_free')
                        data.qvel[joint.dofadr[0]] = 40
                        mujoco.mj_integratePos(model, data.qpos, data.qvel, data.time)
                    mujoco.mj_forward(model, data)
                    qpos, qvel = data.qpos.copy(), data.qvel.copy()
                    frames = {'instant_start': instant.capture(data), 'exposed': exposed.capture(data)}
                    midpoint = mujoco.MjData(model)
                    midpoint.qpos[:] = qpos
                    midpoint.qvel[:] = qvel
                    midpoint.time = data.time + exposure_ms / 2000
                    mujoco.mj_integratePos(model, midpoint.qpos, qvel, exposure_ms / 2000)
                    mujoco.mj_forward(model, midpoint)
                    frames['instant_mid'] = instant.capture(midpoint)
                    np.testing.assert_array_equal(data.qpos, qpos)
                    np.testing.assert_array_equal(data.qvel, qvel)
                    assert data.time == float(schedule['state_s'])
                    name = f'{fps}hz_{exposure_ms}ms_{motion}_{index}'
                    images = {}
                    for variant, frame in frames.items():
                        images[variant] = {}
                        for side in ('left', 'right'):
                            path = Path('images') / f'{name}_{variant}_{side}.png'
                            (output / path).parent.mkdir(exist_ok=True)
                            Image.fromarray(frame[side]).save(output / path)
                            images[variant][side] = str(path)
                    frame = frames['exposed']
                    sample_poses.append(frame['sample_T_world_from_optical'])
                    sample_times.append(frame['sample_timestamps_s'])
                    rows.append({'case_id': name, 'motion': motion, 'fps': fps, 'exposure_ms': exposure_ms,
                                 'frame_index': index, 'scheduled_time_s': float(schedule['scheduled_s']),
                                 'sim_state_time_s': data.time, 'sampling_error_s': float(schedule['lag_s']),
                                 'exposure_start_s': frame['exposure_start_s'], 'exposure_mid_s': frame['exposure_mid_s'],
                                 'exposure_end_s': frame['exposure_end_s'], 'sample_timestamps_s': frame['sample_timestamps_s'],
                                 'calibration_sha256': calibration_sha, 'images': images,
                                 'relative_velocity_m_s': {'static': 0, 'ball': 40, 'camera': -40}[motion],
                                 'optical_depth_m': 2, 'state_file': 'states.npz', 'state_index': len(states),
                                 'capture_start_monotonic_ns': frame['capture_start_monotonic_ns'],
                                 'capture_end_monotonic_ns': frame['capture_end_monotonic_ns'],
                                 'exposure_model': 'constant_qvel_encoded_rgb_average',
                                 'sdk_timestamp': None})
                    states.append(qpos); velocities.append(qvel); times.append(data.time)
        print(f'G3_EXPOSURE {fps}Hz {exposure_ms}ms complete', flush=True)
    np.savez_compressed(output / 'states.npz', qpos=states, qvel=velocities, time=times,
                        sample_T_world_from_optical=sample_poses, sample_time=sample_times)
    (output / 'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    print(f'{len(rows)} synchronized stereo exposures saved to {output}')


if __name__ == '__main__':
    main()
