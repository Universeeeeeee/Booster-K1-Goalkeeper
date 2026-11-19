"""Independently inspect every delivered three-camera pixel buffer and SD."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def verify(root, reference):
    spec=importlib.util.spec_from_file_location('supplied_reference',reference)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    profile_path = root / 'profile.json'
    meta=json.loads(profile_path.read_text()) if profile_path.exists() else json.loads((root/'metadata.json').read_text())
    rows=meta.get('rows')
    if rows is None:
        rows=[json.loads(x) for x in (root/'frames.jsonl').read_text().splitlines()]
    if len(rows) != meta['frames']: raise ValueError('profile frame count mismatch')
    timing_path = root / 'timing.jsonl'
    timing_rows = [json.loads(x) for x in timing_path.read_text().splitlines()] if timing_path.exists() else []
    if timing_path.exists():
        if len(timing_rows) != len(rows): raise ValueError('timing rows do not match frame rows')
        for i, timing in enumerate(timing_rows):
            if timing['index'] != i: raise ValueError(f'timing index mismatch at {i}')
            if timing['source_time_s'] != i * 0.01: raise ValueError(f'wrong source time at {i}')
            if abs((timing['simulation_after_s'] - timing['simulation_before_s']) - 0.01) > 1e-9:
                raise ValueError(f'physical step is not 10 ms at {i}')
            if abs((timing['timeline_after_s'] - timing['timeline_before_s']) - 0.01) > 1e-9:
                raise ValueError(f'timeline step is not 10 ms at {i}')
            if abs(timing['simulation_before_s'] - i * 0.01) > 1e-9:
                raise ValueError(f'wrong physical time before sample {i}')
            if abs(timing['simulation_after_s'] - (i + 1) * 0.01) > 1e-9:
                raise ValueError(f'wrong physical time after sample {i}')
            if abs(timing['timeline_after_s'] - (i + 1) * 0.01) > 1e-9:
                raise ValueError(f'wrong timeline after sample {i}')
    assert len(rows)==meta['frames']
    streams={}; sizes={}; previous={}; repeats={n:[] for n in meta['optics']}; changes={n:[] for n in meta['optics']}
    for n,c in meta['optics'].items():
        w,h=c['resolution']; sizes[n]=w*h*3;streams[n]=(root/(n+'.rgb')).open('rb')
        assert (root/(n+'.rgb')).stat().st_size==sizes[n]*len(rows)
        if meta['mode']!='A' and n!='CenterRGB':
            for suffix,size in [('gray',w*h),('sd',w*h*4)]:
                key=n+'_'+suffix;sizes[key]=size;streams[key]=(root/(n+'.'+suffix)).open('rb')
                assert (root/(n+'.'+suffix)).stat().st_size==size*len(rows)
    for i,row in enumerate(rows):
        assert row['index']==i and row['simulation_ns']==i*10000000
        if 'pixel_ids' in row:
            assert row['pixel_ids'] == {n:i for n in meta['optics']}
        assert all(t==i*10000000 for t in row['eye_timestamp_ns'].values())
        np.testing.assert_allclose(row['position'],[-48+.015*i if meta['mode']=='C' else -48,-8.375,0],atol=1e-9)
        data={}
        for key,stream in streams.items():
            data[key]=stream.read(sizes[key])
            expected_hash = row.get('hashes', {}).get(key)
            if expected_hash is None: raise ValueError(f'missing saved hash for {key} frame {i}')
            assert hashlib.sha256(data[key]).hexdigest()==expected_hash
        for n,c in meta['optics'].items():
            w,h=c['resolution'];img=np.frombuffer(data[n],np.uint8).reshape(h,w,3)
            cells=np.array([img[y-2:y+3,x-2:x+3].mean() for x,y in meta['centres'][n]])
            bits=cells>(cells[0]+cells[1])/2
            assert list(bits[[0,1,2,3,21,22]])==[True,False,True,False,False,True]
            assert bits[20]==bool(bits[4:20].sum()%2)
            assert int(np.dot(bits[4:20],2**np.arange(15,-1,-1)))==i
            roi=img[150:].astype(np.int16)
            if n in previous:
                delta=np.abs(roi-previous[n]);changes[n].append(float(delta.mean()))
                if not np.any(delta):repeats[n].append(i)
            previous[n]=roi
            if n+'_gray' in data:
                gray=np.frombuffer(data[n+'_gray'],np.uint8).reshape(h,w)
                actual=np.frombuffer(data[n+'_sd'],np.int16).reshape(2,h,w)
                up,right,_=module.compute_tdsd_numpy(gray)
                np.testing.assert_array_equal(actual,np.stack([up,right]))
    for stream in streams.values():stream.close()
    matrices={n:np.array(c['local_matrix']) for n,c in meta['optics'].items()}
    delta=np.linalg.inv(matrices['Left'])@matrices['Right'];np.testing.assert_allclose(delta[:3,:3],np.eye(3),atol=1e-9)
    np.testing.assert_allclose(delta[:3,3],[.07,0,0],atol=1e-9)
    timings={k:{'mean':float(np.mean([r[k] for r in rows])),'max':float(max(r[k] for r in rows))} for k in ['render_read_ms','gray_ms','sd_ms','process_ms','total_ms']}
    result={'mode':meta['mode'],'verified_frames':len(rows),'sampling_hz':100,'interval_ms':10,'capture_duration_s':len(rows)/100,
        'first_last_timestamp_span_s':(len(rows)-1)/100,'eye_timestamp_difference_ns':0,'missing_pixel_ids':0,
        'sd_matches_supplied_numpy':meta['mode']!='A','scene_exact_repeat_indices':repeats,
        'scene_frame_mae_percentiles':{n:np.percentile(v,[0,50,95,99,100]).tolist() for n,v in changes.items()},
        'timings_ms':timings,'throughput_pairs_s':len(rows)/meta['elapsed_s'],'rtf':len(rows)/100/meta['elapsed_s'],
        'timing_rows_verified': bool(timing_path.exists()),
        'profile_verified': profile_path.exists(),
        'scope':'pixel identity, saved data, SD arithmetic, authored motion and single-clock timing when timing.jsonl exists; ghosting/flicker require separate image review'}
    if meta['mode']=='C':assert not any(repeats.values()),repeats
    (root/'verification.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--reference',required=True,type=Path);a,_=p.parse_known_args()
    print(json.dumps(verify(a.capture,a.reference),indent=2))
