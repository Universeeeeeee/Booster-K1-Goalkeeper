"""Offline evaluator/transport tests without loading or redistributing weights."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load_script("evaluate_reference_yolo")
transport = load_script("compare_reference_yolo")
CALIBRATION = {"left": {"fx": 336., "fy": 336., "cx": 319.5, "cy": 179.5}}


def label(index=0, visible=True):
    return {"frame_index": index, "depth_label_m": 2, "angle_label": "center",
            "expected_visible": visible, "ball_center_camera_m": [.6, .1, 2], "ball_radius_m": .11}


def detection(box, offset=(0, 0), confidence=.9):
    return {"bbox_xyxy": box, "confidence": confidence,
            "centroid_px": [(box[0]+box[2])/2+offset[0], (box[1]+box[3])/2+offset[1]]}


def prediction(index, detections):
    return {"frame_index": index, "eyes": [detections, []]}


def test_sphere_bbox_bounds_are_independent_tangent_rays():
    center, radius = [.6, .1, 2], .11
    k = CALIBRATION["left"]
    box = reference.sphere_bbox(center, radius, k)
    for offset, lo, hi, focal, principal in ((center[0], box[0], box[2], k["fx"], k["cx"]),
                                             (center[1], box[1], box[3], k["fy"], k["cy"])):
        for bound in (lo, hi):
            slope = (bound-principal)/focal
            # In the x/z (or y/z) plane, the tangent line's distance to the
            # projected circle center equals its radius.
            distance = abs(offset-slope*center[2])/np.sqrt(1+slope*slope)
            assert distance == pytest.approx(radius, abs=1e-12)


def test_iou_threshold_is_inclusive_and_unmatched_boxes_count_as_false_positive():
    truth = label()
    box = reference.sphere_bbox(truth["ball_center_camera_m"], .11, CALIBRATION["left"])
    half = [box[0], box[1], (box[0]+box[2])/2, box[3]]
    assert reference.iou(box, half) == pytest.approx(.5)
    matched = reference.evaluate([prediction(0, [detection(half)])], [truth], CALIBRATION)["groups"][0]
    assert matched["true_positive"] == 1
    assert matched["false_positive"] == matched["false_negative"] == 0
    missed_box = [box[0], box[1], box[0]+.49*(box[2]-box[0]), box[3]]
    missed = reference.evaluate([prediction(0, [detection(missed_box)])], [truth], CALIBRATION)["groups"][0]
    assert missed["true_positive"] == 0
    assert missed["false_positive"] == missed["false_negative"] == 1
    assert missed["matched_centroid_mae_px"] is None


def test_one_gt_matches_once_and_uses_best_iou_instead_of_highest_confidence():
    truth = label()
    box = reference.sphere_bbox(truth["ball_center_camera_m"], .11, CALIBRATION["left"])
    bad = [box[0]-10, box[1], box[2]+10, box[3]]
    detections = [detection(bad, (100, 100), .99), detection(box, (3, 4), .3), detection(box, (8, 0), .8)]
    group = reference.evaluate([prediction(0, detections)], [truth], CALIBRATION)["groups"][0]
    assert group["true_positive"] == 1
    assert group["false_positive"] == 2
    assert group["false_negative"] == 0
    assert group["matched_centroid_mae_px"] == 5
    assert group["precision"] == pytest.approx(1/3)
    assert group["recall"] == 1


def test_invisible_duplicate_detections_all_count_as_false_positives():
    detections = [detection([0, 0, 10, 10])]*2
    labels = [label(0, False), label(1, False)]
    group = reference.evaluate([prediction(0, detections), prediction(1, [])], labels, CALIBRATION)["groups"][0]
    assert group["frames"] == 2
    assert group["false_positive"] == 2
    assert group["true_positive"] == group["false_negative"] == 0
    assert group["precision"] == 0
    assert group["recall"] is None
    assert group["matched_centroid_mae_px"] is None


def test_missing_visible_detection_counts_false_negative_without_invented_error():
    group = reference.evaluate([prediction(0, [])], [label()], CALIBRATION)["groups"][0]
    assert group["false_negative"] == 1
    assert group["recall"] == 0
    assert group["precision"] is None
    assert group["matched_centroid_mae_px"] is None


@pytest.mark.parametrize("predictions,labels", [
    ([prediction(0, []), prediction(0, [])], [label()]),
    ([prediction(0, [])], [label(), label()]),
    ([prediction(0, [])], [label(), label(1)]),
    ([prediction(0, []), prediction(1, [])], [label()]),
    ([prediction(1, [])], [label()]),
])
def test_frame_duplicates_and_coverage_mismatch_rejected(predictions, labels):
    with pytest.raises(ValueError, match="Duplicate frame IDs|coverage differs"):
        reference.evaluate(predictions, labels, CALIBRATION)


def test_frame_join_is_by_id_and_input_order_does_not_change_counts():
    labels = [label(1), label(0, False)]
    box = reference.sphere_bbox(labels[0]["ball_center_camera_m"], .11, CALIBRATION["left"])
    group = reference.evaluate([prediction(0, []), prediction(1, [detection(box)])], labels, CALIBRATION)["groups"][0]
    assert group["true_positive"] == 1
    assert group["false_positive"] == group["false_negative"] == 0


def test_transport_sends_only_images_and_rejects_remote_frame_mismatch(tmp_path, monkeypatch):
    capture = tmp_path/"capture"
    capture.mkdir()
    for eye in ("left", "right"):
        (capture/f"{eye}.png").write_bytes(b"example encoded image")
    frame = {"frame_index": 7, "left_path": "left.png", "right_path": "right.png",
             "ball_center_camera_m": [0, 0, 99]}
    (capture/"frames.jsonl").write_text(json.dumps(frame)+"\n")
    monkeypatch.setattr("sys.argv", ["compare_reference_yolo", "--capture", str(capture),
                                    "--output", str(tmp_path/"result.json")])
    calls = []
    def remote(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        return SimpleNamespace(stdout='G5_RESULT='+json.dumps({"frames": [{"frame_index": 8}]})+'\n')
    monkeypatch.setattr(transport.subprocess, "run", remote)
    with pytest.raises(RuntimeError, match="mismatched frames"):
        transport.main()
    assert not (tmp_path/"result.json").exists()
    assert set(calls[0][1]["frames"][0]) == {"frame_index", "left", "right"}
    assert calls[0][0][0] == "ssh"
