"""Run the audited reference detector in place, sending only RGB images over SSH.

No model/code is redistributed. CPU inference preserves the reference GPU jobs.
RGB is converted to grayscale: this is an explicit domain-shift experiment for
the reference IR model, not an IR camera simulation.
"""
import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r'''
import base64, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import ultralytics
torch.set_num_threads(2)
request=json.load(sys.stdin)
root=Path(request['root'])
weights=root/'runs/full_v2/weights/best.pt'
digest=hashlib.sha256(weights.read_bytes()).hexdigest()
if digest != '9723db63ec8e2a0588030f925755419a55719e8a06c6360d95d80925a713c107':
    raise ValueError('Audited reference weights changed')
sys.path.insert(0,str(root))
from ball.detector import YoloDetector
detector=YoloDetector(weights, confidence=.25, device='cpu')
rows=[]
for row in request['frames']:
    images=[cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(row[eye]),np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2GRAY) for eye in ('left','right')]
    detections=detector.detect(images)
    rows.append({'frame_index':row['frame_index'],'eyes':[[{'bbox_xyxy':d.xyxy,'confidence':d.score,'centroid_px':list(d.uv)} for d in eye] for eye in detections]})
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'ball/detector.py',root/'ball/boxes.py',root/'ball/geometry.py',root/'ball/deployment.py']}
print('G5_RESULT='+json.dumps({'weights_sha256':digest,'source_sha256':sources,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'device':'cpu','input':'RGB converted to grayscale; reference trained on IR','confidence_threshold':.25,'frames':rows}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='pc@100.75.243.43')
    parser.add_argument('--root', default='/home/pc/zean.xi/stereo_ir_ball_v1')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    frames = []
    for row in map(json.loads, (args.capture/'frames.jsonl').read_text().splitlines()):
        frames.append({'frame_index': row['frame_index'], **{
            eye: base64.b64encode((args.capture/row[f'{eye}_path']).read_bytes()).decode()
            for eye in ('left', 'right')}})
    command = ['env', 'PYTHONDONTWRITEBYTECODE=1', 'CUDA_VISIBLE_DEVICES=',
               args.root+'/.venv/bin/python', '-c', REMOTE]
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
                             shlex.join(command)], input=json.dumps({'root':args.root,'frames':frames}),
                            text=True, capture_output=True, check=True)
    responses = [line.removeprefix('G5_RESULT=') for line in result.stdout.splitlines()
                 if line.startswith('G5_RESULT=')]
    if len(responses) != 1:
        raise RuntimeError('Expected one reference detector result')
    payload = json.loads(responses[0])
    if [r['frame_index'] for r in payload['frames']] != [r['frame_index'] for r in frames]:
        raise RuntimeError('Reference detector returned mismatched frames')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(f"Saved {len(frames)} reference YOLO stereo predictions")


if __name__ == '__main__':
    main()
