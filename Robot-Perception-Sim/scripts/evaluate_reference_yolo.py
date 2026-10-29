"""Offline image-space evaluation of reference YOLO; GT never enters inference."""
import argparse
import json
from pathlib import Path

import numpy as np


def sphere_bbox(center, radius, intrinsics):
    x, y, z = center
    if z <= radius:
        raise ValueError('Sphere must lie in front of camera')
    bounds = []
    for offset, focal, principal in ((x, intrinsics['fx'], intrinsics['cx']),
                                     (y, intrinsics['fy'], intrinsics['cy'])):
        spread = radius*np.sqrt(offset*offset+z*z-radius*radius)
        bounds.append([principal+focal*(offset*z+s*spread)/(z*z-radius*radius)
                       for s in (-1, 1)])
    return [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.0


def evaluate(predictions, labels, calibration):
    lookup = {r['frame_index']: r for r in predictions}
    if len(lookup) != len(predictions) or len({r['frame_index'] for r in labels}) != len(labels):
        raise ValueError('Duplicate frame IDs')
    if set(lookup) != {r['frame_index'] for r in labels}:
        raise ValueError('Prediction/label frame coverage differs')
    groups = {}
    for label in labels:
        key = (label['depth_label_m'], label['angle_label'])
        group = groups.setdefault(key, {'depth_m':key[0], 'angle':key[1], 'frames':0,
                                       'true_positive':0, 'false_positive':0, 'false_negative':0,
                                       'centroid_errors_px':[]})
        group['frames'] += 1
        detections = lookup[label['frame_index']]['eyes'][0]
        if not label['expected_visible']:
            group['false_positive'] += len(detections)
            continue
        box = sphere_bbox(label['ball_center_camera_m'], label.get('ball_radius_m', .11), calibration['left'])
        overlaps = [iou(d['bbox_xyxy'], box) for d in detections]
        matched = bool(overlaps) and max(overlaps) >= .5
        group['true_positive'] += int(matched)
        group['false_negative'] += int(not matched)
        group['false_positive'] += len(detections)-int(matched)
        if matched:
            selected = detections[int(np.argmax(overlaps))]
            truth = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
            group['centroid_errors_px'].append(float(np.linalg.norm(np.asarray(selected['centroid_px'])-truth)))
    for group in groups.values():
        tp, fp, fn = (group[k] for k in ('true_positive','false_positive','false_negative'))
        group['precision'] = tp/(tp+fp) if tp+fp else None
        group['recall'] = tp/(tp+fn) if tp+fn else None
        errors = group.pop('centroid_errors_px')
        group['matched_centroid_mae_px'] = float(np.mean(errors)) if errors else None
    return {'iou_threshold':.5, 'eye':'left', 'bbox_ground_truth':'analytic sphere silhouette',
            'gt_used_only_after_predictions':True, 'groups':list(groups.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions','evaluation','metadata','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text())
    labels = [json.loads(line) for line in args.evaluation.read_text().splitlines()]
    result = evaluate(predictions['frames'], labels, json.loads(args.metadata.read_text())['calibration'])
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
