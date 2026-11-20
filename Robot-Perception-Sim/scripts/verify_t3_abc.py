"""Independently inspect delivered camera pixels, spatial and temporal differences."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def decode_frame_id(image, strips):
    ids = set()
    for centres in strips:
        cells = np.array([image[y-2:y+3, x-2:x+3, :3].mean() for x, y in centres])
        bits = cells > (cells[0] + cells[1]) / 2
        if list(bits[[0, 1, 2, 3, 21, 22]]) != [True, False, True, False, False, True]:
            continue
        if bits[20] == bool(bits[4:20].sum() % 2):
            ids.add(int(np.dot(bits[4:20], 2 ** np.arange(15, -1, -1))))
    return next(iter(ids)) if len(ids) == 1 else None


def verify(root, reference):
    spec=importlib.util.spec_from_file_location('supplied_reference',reference)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    profile_path = root / 'profile.json'
    meta=json.loads(profile_path.read_text()) if profile_path.exists() else json.loads((root/'metadata.json').read_text())
    sd_encoding = meta.get('sd_encoding', 'full_difference')
    if sd_encoding not in ('full_difference', 'half_trunc', 'half_trunc_plus_128'):
        raise ValueError(f'unknown SD encoding: {sd_encoding}')
    biased_sd = sd_encoding == 'half_trunc_plus_128'
    td_encoding = meta.get('td_encoding')
    if td_encoding not in (None, 'int8_half_trunc_current_minus_previous_first_zero'):
        raise ValueError(f'unknown TD encoding: {td_encoding}')
    if td_encoding and meta['mode'] not in ('B', 'C'):
        raise ValueError('TD encoding requires stereo difference capture mode B or C')
    td_eyes = {'Left', 'Right'} if td_encoding else set()
    if {path.stem for path in root.glob('*.td')} != td_eyes:
        raise ValueError('TD files do not match the declared encoding')
    td_keys = {eye + '_td' for eye in td_eyes}
    rows=meta.get('rows')
    if rows is None:
        rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
    if len(rows) != meta['frames']: raise ValueError('profile frame count mismatch')
    timing_path = root / 'timing.jsonl'
    timing_rows = [json.loads(x) for x in timing_path.read_text().splitlines()] if timing_path.exists() else []
    c1_capture = biased_sd and meta.get('options', {}).get('clock_mode') == 'C1'
    if c1_capture and not timing_path.exists():
        raise ValueError('missing C1 timing log')
    if timing_path.exists():
        if len(timing_rows) != len(rows): raise ValueError('timing rows do not match frame rows')
        for i, timing in enumerate(timing_rows):
            if timing['index'] != i: raise ValueError(f'timing index mismatch at {i}')
            if abs(timing['source_time_s'] - i * 0.01) > 1e-9: raise ValueError(f'wrong source time at {i}')
            if abs((timing['simulation_after_s'] - timing['simulation_before_s']) - 0.01) > 1e-7:
                raise ValueError(f'physical step is not 10 ms at {i}')
            if abs((timing['timeline_after_s'] - timing['timeline_before_s']) - 0.01) > 5e-8:
                raise ValueError(f'timeline step is not 10 ms at {i}')
            # PhysX accumulates a float32 step. Account for its known
            # representation error separately from the per-sample check.
            hz = meta.get('options', {}).get('physics_hz', 100)
            substeps = meta.get('options', {}).get('physical_steps_per_sample', 1)
            rounding_per_sample = abs(float(np.float32(1 / hz)) * substeps - 0.01)
            if abs(timing['simulation_before_s'] - i * 0.01) > 5e-7 + i * rounding_per_sample:
                raise ValueError(f'wrong physical time before sample {i}')
            if abs(timing['simulation_after_s'] - (i + 1) * 0.01) > 5e-7 + (i + 1) * rounding_per_sample:
                raise ValueError(f'wrong physical time after sample {i}')
            if abs(timing['timeline_after_s'] - (i + 1) * 0.01) > 1e-7:
                raise ValueError(f'wrong timeline after sample {i}')
            if 'refresh_sim_deltas' in timing and any(abs(x) > 1e-9 for x in timing['refresh_sim_deltas']):
                raise ValueError(f'physics advanced during refresh at {i}')
            if 'refresh_timeline_deltas' in timing and any(abs(x) > 1e-9 for x in timing['refresh_timeline_deltas']):
                raise ValueError(f'timeline advanced during refresh at {i}')
            if c1_capture:
                row = rows[i]
                if any(row[key] != value for key, value in timing.items()):
                    raise ValueError(f'profile/timing mismatch at {i}')
                for key in ('refresh_sim_deltas', 'refresh_timeline_deltas'):
                    values = row[key]
                    if not 1 <= len(values) == row['refresh_count'] <= 16 or any(not abs(x) <= 1e-9 for x in values):
                        raise ValueError(f'invalid {key} at {i}')
    assert len(rows)==meta['frames']
    streams={}; sizes={}; previous={}; repeats={n:[] for n in meta['optics']}; changes={n:[] for n in meta['optics']}
    previous_gray = {}
    for n,c in meta['optics'].items():
        w,h=c['resolution']; sizes[n]=w*h*3;streams[n]=(root/(n+'.rgb')).open('rb')
        assert (root/(n+'.rgb')).stat().st_size==sizes[n]*len(rows)
        if meta['mode']!='A' and n!='CenterRGB':
            formats = [('gray', w*h), ('sd', w*h*4)]
            if td_encoding:
                formats.append(('td', w*h))
            for suffix,size in formats:
                key=n+'_'+suffix;sizes[key]=size;streams[key]=(root/(n+'.'+suffix)).open('rb')
                assert (root/(n+'.'+suffix)).stat().st_size==size*len(rows)
    for i,row in enumerate(rows):
        recorded_td = {key for key in row.get('hashes', {}) if key.endswith('_td')}
        if recorded_td != td_keys:
            raise ValueError(f'TD hashes do not match the declared encoding at frame {i}')
        assert row['index']==i and row['simulation_ns']==i*10000000
        if 'pixel_ids' in row:
            assert row['pixel_ids'] == {n:i for n in meta['optics']}
        if 'eye_timestamp_ns' in row:
            assert all(t==i*10000000 for t in row['eye_timestamp_ns'].values())
        if 'position' in row:
            np.testing.assert_allclose(row['position'],[-48+.015*i if meta['mode']=='C' else -48,-8.375,0],atol=1e-9)
        data={}
        for key,stream in streams.items():
            data[key]=stream.read(sizes[key])
            expected_hash = row.get('hashes', {}).get(key)
            if (sd_encoding != 'full_difference' or key.endswith('_td')) and not expected_hash:
                raise ValueError(f'missing SHA-256 for {key} frame {i}')
            if expected_hash and hashlib.sha256(data[key]).hexdigest() != expected_hash:
                raise ValueError(f'SHA-256 mismatch for {key} frame {i}')
        for n,c in meta['optics'].items():
            w,h=c['resolution'];img=np.frombuffer(data[n],np.uint8).reshape(h,w,3)
            strips = meta.get('marker_strips', {}).get(n, [meta['centres'][n]])
            assert decode_frame_id(img, strips) == i, (n, i)
            roi=img[150:].astype(np.int16)
            if n in previous:
                delta=np.abs(roi-previous[n]);changes[n].append(float(delta.mean()))
                if not np.any(delta):repeats[n].append(i)
            previous[n]=roi
            if n+'_gray' in data:
                gray=np.frombuffer(data[n+'_gray'],np.uint8).reshape(h,w)
                actual=np.frombuffer(data[n+'_sd'],np.int16).reshape(2,h,w)
                up,right,expected_td=module.compute_tdsd_numpy(
                    gray, previous_gray.get(n), legacy_sd=sd_encoding == 'full_difference', sd_offset=128 if biased_sd else 0)
                np.testing.assert_array_equal(actual,np.stack([up,right]))
                if td_encoding:
                    gray_id = decode_frame_id(gray[..., None], strips)
                    if gray_id != i:
                        raise ValueError(f'{n} TD source gray ID {gray_id}, expected {i}')
                    actual_td = np.frombuffer(data[n + '_td'], np.int8).reshape(h, w)
                    if i == 0:
                        expected_td = np.zeros_like(gray, dtype=np.int8)
                    np.testing.assert_array_equal(actual_td, expected_td, err_msg=f'{n} TD frame {i}')
                    previous_gray[n] = gray
    for stream in streams.values():stream.close()
    matrices={n:np.array(c['local_matrix']) for n,c in meta['optics'].items()}
    delta=np.linalg.inv(matrices['Left'])@matrices['Right'];np.testing.assert_allclose(delta[:3,:3],np.eye(3),atol=1e-9)
    np.testing.assert_allclose(delta[:3,3],[.07,0,0],atol=1e-9)
    def metric(key, fallback=0.0):
        values=[r.get(key, fallback) for r in rows]
        return {'mean':float(np.mean(values)),'max':float(max(values))}
    timings={k:metric(k) for k in ['render_read_ms','gray_ms','sd_ms','process_ms','total_ms']}
    if 'render_ms' in rows[0]: timings['render_ms']=metric('render_ms')
    if 'gpu_process_ms' in rows[0]: timings['gpu_process_ms']=metric('gpu_process_ms')
    result={'mode':meta['mode'],'verified_frames':len(rows),'sampling_hz':100,'interval_ms':10,'capture_duration_s':len(rows)/100,
        'first_last_timestamp_span_s':(len(rows)-1)/100,'eye_timestamp_difference_ns':0,'missing_pixel_ids':0,
        'sd_matches_supplied_numpy':meta['mode']!='A','sd_encoding':sd_encoding,'scene_exact_repeat_indices':repeats,
        'scene_frame_mae_percentiles':{n:np.percentile(v,[0,50,95,99,100]).tolist() if v else [] for n,v in changes.items()},
        'timings_ms':timings,'throughput_pairs_s':len(rows)/meta['elapsed_s'],'rtf':len(rows)/100/meta['elapsed_s'],
        'timing_rows_verified': bool(timing_path.exists()),
        'profile_verified': profile_path.exists(),
        'td_encoding':td_encoding, 'td_frames_verified_per_eye':len(rows) if td_encoding else 0,
        'scope':'pixel identity, saved data, SD arithmetic, signed TD when present, authored motion and timing when timing.jsonl exists'}
    if meta['mode']=='C':assert not any(repeats.values()),repeats
    (root/'verification.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--reference',required=True,type=Path);a,_=p.parse_known_args()
    print(json.dumps(verify(a.capture,a.reference),indent=2))
