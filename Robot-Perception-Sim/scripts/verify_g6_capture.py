"""Independently verify delivered G6 pixels and archived exposure publications.

This checks acquisition integrity, not detection accuracy or physical equivalence.
No valid-depth fraction is required: invalid SDK depth remains valid evidence.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from verify_g3_sdk_composite import verify_nominal_calibration
from verify_zed_sdk_stream import decode_frame_id


def require(value, message):
    if not value:
        raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rgba(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGBA'))


def rational(value):
    return Fraction(value['numerator'], value['denominator'])


def verify_exposure(source, state, visual_id, metadata):
    """Recompute quadrature, pixels and uploaded bytes without producer helpers."""
    config = metadata['acquisition']
    window = {key: rational(value) for key, value in state['window_rational'].items()}
    duration = window['end_s']-window['start_s']
    require(duration >= 0 and duration == Fraction(str(config['realized_exposure_s'])), 'Exposure duration differs')
    require(window['mid_s'] == (window['start_s']+window['end_s'])/2, 'False exposure midpoint')
    require(duration < Fraction(1, metadata['configured_fps']), 'Exposure exceeds frame period')
    for key, value in window.items():
        require(abs(state['t_exposure_'+key]-float(value)) < 1e-10, 'Exposure rational/float mismatch')
    require(abs(state['publication']['timestamp_s']-float(window['mid_s'])) < 1e-10, 'Published timestamp differs from midpoint')
    require(state['sdk_composite_timestamp_ns'] == int(float(window['mid_s'])*1e9), 'False SDK timestamp')
    samples = state['subframes']
    require(len(samples) == config['samples_per_exposure'] and len(samples) > 0, 'Wrong subframe count')
    ids, images = [], {'left': [], 'right': []}
    for index, sample in enumerate(samples):
        expected = window['start_s']+Fraction(2*index+1, 2*len(samples))*duration
        require(rational(sample['target_s']) == expected, 'Subframe quadrature mismatch')
        require(rational(sample['state']['motion_target_s']) == expected, 'State/subframe timestamp mismatch')
        require(sample['host_authored_monotonic_ns'] <= sample['host_observed_monotonic_ns'] <= state['publication']['host_armed_monotonic_ns'], 'Noncausal source publication')
        sample_state = sample['state']
        require(sample_state['native_world_time_s'] == sample_state['native_world_time_after_render_s'], 'Physics advanced during held replay')
        ids.append(sample['subframe_id'])
        for side in images:
            pixels = rgba(source/sample[side+'_path'])
            require(decode_frame_id(pixels, metadata['frame_id_marker']['pixel_centres'][side]) == visual_id, 'Subframe primary pixel identity mismatch')
            require(decode_frame_id(pixels, config['subframe_marker_pixel_centres'][side]) == sample['subframe_id'], 'Subframe secondary pixel identity mismatch')
            images[side].append(pixels)
    for index, side in enumerate(images):
        mean = np.rint(np.mean(images[side], axis=0, dtype=np.float64)).astype(np.uint8)
        composite = rgba(source/state['composite_paths'][side])
        require(np.array_equal(mean, composite), 'Composite is not archived subframe average')
        require(hashlib.sha256(composite.tobytes()).hexdigest() == state['publication']['source_sha256_left_right'][index], 'Published CUDA bytes differ from composite')
        require(composite.nbytes == state['publication']['bytes_per_eye'], 'Published buffer size differs')
    return ids


def verify(source, capture, trajectory=None):
    source, capture = Path(source), Path(capture)
    metadata, sdk = (json.loads((p/'metadata.json').read_text()) for p in (source, capture))
    require(sdk['sdk_version'].startswith('5.1') and sdk['depth_mode'] == 'NEURAL', 'Requires actual SDK5.1 NEURAL')
    require(sdk['depth_source'] == 'zed_stereo_sdk' and sdk['depth_units'] == 'm', 'Wrong depth provenance/units')
    calibration = verify_nominal_calibration(capture, sdk, metadata)
    for item in metadata['acquisition']['source_files'].values():
        path = (source/item['path']).resolve()
        require(path.is_relative_to(source.resolve()), 'Snapshot escapes source')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Source snapshot hash mismatch')
    authored = rows(source/'source_frames.jsonl')
    by_id = {row['visual_frame_id']: row for row in authored}
    require(len(by_id) == len(authored), 'Duplicate source visual IDs')
    intended = metadata['acquisition']['target_visual_ids']
    require(bool(intended) and len(set(intended)) == len(intended), 'Invalid intended target identities')
    target_ids = {key for key, row in by_id.items() if row['acquisition']['phase'] == 'target'}
    require(target_ids == set(intended), 'Published target sequence differs from intended sequence')
    raw_trajectory = json.loads(Path(trajectory).read_text()) if trajectory else None
    trajectory_digest = hashlib.sha256(Path(trajectory).read_bytes()).hexdigest() if trajectory else None
    observed_packets = rows(source/'render_observations.jsonl')
    packet_keys = {(row.get('observed_left_subframe_id'), tuple(row['reference_time']),
                    row.get('observed_left_visual_id')) for row in observed_packets
                   if row.get('observation_valid') and 'reference_time' in row
                   and row.get('observed_left_subframe_id') == row.get('observed_right_subframe_id')}
    subframe_ids = set()
    maximum_quantization_error = 0.0
    for identity in intended:
        state = by_id[identity]['acquisition']
        ids = verify_exposure(source, state, identity, metadata)
        require(not subframe_ids.intersection(ids) and len(ids) == len(set(ids)), 'Reused target subframe IDs')
        subframe_ids.update(ids)
        for sample in state['subframes']:
            require((sample['subframe_id'], tuple(sample['reference_time']), identity) in packet_keys,
                    'Subframe lacks matching independent Writer packet')
            if raw_trajectory is not None:
                sample_state = sample['state']
                require(sample_state['source_trajectory_sha256'] == trajectory_digest, 'Frozen trajectory digest differs')
                requested = sample_state['trajectory_requested_time_s']
                frozen = raw_trajectory['trajectory']
                require(frozen[0]['time'] <= requested <= frozen[-1]['time'], 'Replay extrapolates beyond frozen trajectory')
                nearest = min(frozen, key=lambda row: abs(row['time']-requested))
                require(abs(sample_state['trajectory_realized_time_s']-nearest['time']) < 1e-10, 'Replay did not select nearest frozen sample')
                require(np.allclose(sample_state['ball_world_position_m'], nearest['position'], rtol=0, atol=1e-8),
                        'Replay position differs from frozen physical sample')
                maximum_quantization_error = max(maximum_quantization_error, abs(nearest['time']-requested))
    frames = rows(capture/'frames.jsonl')
    require(bool(frames), 'Empty SDK capture')
    received, joined, previous = set(), [], -1
    codec_errors = []
    for index, frame in enumerate(frames):
        require(frame['frame_index'] == index and frame['sdk_timestamp_ns'] > previous, 'Noncontiguous receiver indices or timestamp regression')
        previous = frame['sdk_timestamp_ns']
        require(frame['camera_timestamp_after_retrieval_ns'] == previous, 'SDK frame changed during retrieval')
        decoded, delivered = [], {}
        for side in ('left', 'right'):
            pixels = rgba(capture/frame[side+'_path'])
            require(pixels.shape[:2] == (frame['height'], frame['width']), 'Image size mismatch')
            decoded.append(decode_frame_id(pixels, metadata['frame_id_marker']['pixel_centres'][side]))
            delivered[side] = pixels
        require(decoded[0] == decoded[1] and decoded[0] in by_id, 'Delivered stereo pixels not uniquely linked to source')
        identity = decoded[0]
        require(identity not in received, 'Repeated delivered source ID')
        received.add(identity)
        state = by_id[identity]['acquisition']
        require(abs(previous-state['sdk_composite_timestamp_ns']) <= 2, 'Delivered timestamp/pixel state mismatch')
        require(state['publication']['host_armed_monotonic_ns'] <= frame['host_monotonic_ns'] <= frame['host_saved_monotonic_ns'], 'Noncausal receiver observation')
        if identity in target_ids:
            for side in delivered:
                composite = rgba(source/state['composite_paths'][side])
                error = np.abs(delivered[side][...,:3].astype(float)-composite[...,:3])
                mae = float(error.mean())
                require(mae <= 5, 'Delivered RGB differs substantially from published composite')
                codec_errors.append({'visual_id': identity, 'eye': side, 'mae_8bit': mae,
                                     'p95_8bit': float(np.percentile(error,95))})
        for kind in ('depth', 'confidence'):
            values = np.fromfile(capture/frame[kind+'_path'], dtype='<f4')
            require(values.size == frame['width']*frame['height'], 'Invalid SDK float payload size')
            if kind == 'depth':
                require(int((np.isfinite(values) & (values > 0)).sum()) == frame['valid_depth_count'], 'SDK depth validity count differs')
        joined.append({'frame_index': index, 'visual_id': identity})
    require(set(intended).issubset(received), 'Continuous target sequence has missing delivered frames')
    return {'frames_verified': len(frames), 'calibration': calibration, 'joined': joined,
            'target_frames_verified': len(intended), 'target_subframes_verified': len(subframe_ids),
            'maximum_replay_quantization_error_s': maximum_quantization_error if trajectory else None,
            'frozen_trajectory_sha256': trajectory_digest,
            'rgb_transport_mae_limit_8bit': 5, 'rgb_transport_errors': codec_errors,
            'verified_visual_ids': sorted(received), 'acquisition_integrity_only': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.source, args.capture, args.trajectory)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'frames_verified': report['frames_verified']}))


if __name__ == '__main__':
    main()
