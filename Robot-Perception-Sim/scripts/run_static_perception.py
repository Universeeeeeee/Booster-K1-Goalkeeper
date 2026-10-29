"""Run image-only static perception on a saved ZED SDK receiver capture."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_perception_sim.ball_perception import process_frame, world_from_camera


def surface_errors(depth, confidence, mask, center, intrinsics, radius_m):
    """Evaluator-only SDK Z residual against analytic visible sphere surface."""
    v, u = np.nonzero(mask)
    rays = np.column_stack(((u-intrinsics["cx"])/intrinsics["fx"],
                            (v-intrinsics["cy"])/intrinsics["fy"], np.ones(len(u))))
    center = np.asarray(center, dtype=float)
    a = np.sum(rays*rays, axis=1)
    dot = rays @ center
    discriminant = dot*dot-a*(center@center-radius_m**2)
    observed = depth[v, u]
    quality = confidence[v, u]
    usable = (discriminant >= 0) & (dot > 0) & np.isfinite(observed) & (observed > 0) & np.isfinite(quality) & (quality >= 0) & (quality <= 95)
    expected = (dot[usable]-np.sqrt(discriminant[usable]))/a[usable]
    return (observed[usable]-expected).tolist()


def evaluate(predictions, labels, calibration=None, surface_reader=None, radius_m=.11):
    """Only evaluator reads GT, after predictions are complete and persisted.

    Optional surface_reader(frame_index) returns depth, confidence and left mask.
    It lets evaluation re-read SDK payloads without exposing GT to the predictor.
    """
    by_frame = {row["frame_index"]: row for row in predictions}
    if len(by_frame) != len(predictions):
        raise ValueError("Prediction frame indices must be unique")
    label_ids = [row["frame_index"] for row in labels]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("Evaluation frame labels must be unique")
    groups, detailed = {}, []
    positive_cells = set()
    for label in labels:
        prediction = by_frame.get(label["frame_index"])
        if prediction is None:
            raise ValueError(f"Evaluation frame absent: {label['frame_index']}")
        truth = label.get("ball_center_camera_m")
        visible = label.get("expected_visible", truth is not None)
        if type(visible) is not bool:
            raise ValueError("expected_visible must be a boolean")
        if visible and (truth is None or np.asarray(truth).shape != (3,) or not np.isfinite(truth).all() or truth[2] <= radius_m):
            raise ValueError("Visible evaluation frame needs finite ball center in front of camera")
        key = (label.get("depth_label_m"), label["angle_label"], visible)
        if visible and key[0] is not None and key[1] in ("left", "center", "right"):
            positive_cells.add(key[:2])
        group = groups.setdefault(key, {"depth_label_m": key[0], "angle_label": key[1],
            "expected_visible": visible, "frames": 0, "detected": 0,
            "false_positives": 0, "missed_detections": 0, "detection_invalid_reasons": {},
            "centroid_errors_px": [], "sdk_surface_errors_m": [],
            "mask_fractions": [], "depth_valid_fractions": [], "methods": {}})
        group["frames"] += 1
        detected = bool(prediction["left"]["valid"])
        group["detected"] += int(detected)
        group["false_positives"] += int(detected and not visible)
        group["missed_detections"] += int(not detected and visible)
        if not detected:
            reason = prediction["left"]["reason"]
            group["detection_invalid_reasons"][reason] = group["detection_invalid_reasons"].get(reason, 0)+1
        group["mask_fractions"].append(prediction["left"].get("mask_fraction", 0.0))
        group["depth_valid_fractions"].append(prediction["sdk_sphere"]["valid_depth_fraction"])
        detail = {"frame_index": label["frame_index"], "expected_visible": visible, "methods": {}}
        if "T_world_camera" in label:
            detail["T_world_camera"] = label["T_world_camera"]
        if visible and detected and calibration:
            k = calibration["left"]
            x, y, z = truth
            projected = np.array([k["cx"]+k["fx"]*x*z/(z*z-radius_m**2),
                                  k["cy"]+k["fy"]*y*z/(z*z-radius_m**2)])
            error = float(np.linalg.norm(np.asarray(prediction["left"]["centroid_px"])-projected))
            detail["projected_sphere_centroid_px"] = projected.tolist()
            detail["centroid_error_px"] = error
            group["centroid_errors_px"].append(error)
            if surface_reader:
                errors = surface_errors(*surface_reader(label["frame_index"]), truth, k, radius_m)
                group["sdk_surface_errors_m"].extend(errors)
                detail["sdk_surface_samples"] = len(errors)
                detail["sdk_surface_mae_m"] = float(np.mean(np.abs(errors))) if errors else None
        for method in ("triangulation", "sdk_sphere"):
            result = prediction[method]
            metrics = group["methods"].setdefault(method, {"valid": 0, "false_positives": 0,
                "invalid_reasons": {}, "depth_errors_m": [], "center_errors_m": []})
            metrics["valid"] += int(result["valid"])
            metrics["false_positives"] += int(result["valid"] and not visible)
            if not result["valid"]:
                reason = result["reason"]
                metrics["invalid_reasons"][reason] = metrics["invalid_reasons"].get(reason, 0)+1
            if result["valid"] and visible:
                estimate = np.asarray(result["center_camera_m"])
                error = estimate - np.asarray(truth)
                metrics["depth_errors_m"].append(float(error[2]))
                metrics["center_errors_m"].append(float(np.linalg.norm(error)))
                detail["methods"][method] = {"depth_error_m": float(error[2]), "center_error_m": float(np.linalg.norm(error))}
                if "T_world_camera" in label:
                    detail["methods"][method]["oracle_world_center_m"] = world_from_camera(estimate, label["T_world_camera"])
                    world_truth = world_from_camera(truth, label["T_world_camera"])
                    detail["methods"][method]["oracle_world_center_error_m"] = float(np.linalg.norm(np.asarray(detail["methods"][method]["oracle_world_center_m"])-world_truth))
        detailed.append(detail)
    for group in groups.values():
        group["detection_rate"] = group["detected"]/group["frames"]
        group["false_positive_rate"] = group["false_positives"]/group["frames"] if not group["expected_visible"] else None
        for name in ("mask", "depth_valid"):
            values = group.pop(f"{name}_fractions")
            group[f"mean_{name}_fraction"] = float(np.mean(values))
            group[f"minimum_{name}_fraction"] = float(min(values))
        for name, unit in (("centroid", "px"), ("sdk_surface", "m")):
            errors = group.pop(f"{name}_errors_{unit}")
            group[f"{name}_samples"] = len(errors)
            group[f"{name}_mae_{unit}"] = float(np.mean(np.abs(errors))) if errors else None
            group[f"{name}_rmse_{unit}"] = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        for metrics in group["methods"].values():
            for kind in ("depth", "center"):
                errors = metrics.pop(f"{kind}_errors_m")
                metrics[f"{kind}_samples"] = len(errors)
                metrics[f"{kind}_mae_m"] = float(np.mean(np.abs(errors))) if errors else None
                metrics[f"{kind}_rmse_m"] = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    depths = sorted({key[0] for key in positive_cells})
    required = {(depth, angle) for depth in depths for angle in ("left", "center", "right")}
    complete = len(depths) == 5 and positive_cells == required
    return {"evaluation_only_gt": True, "five_depth_three_angle_coverage": complete,
            "positive_depths_m": depths, "positive_grid_cells": len(positive_cells),
            "oracle_pose_only": True, "groups": list(groups.values()), "per_frame": detailed}


def run(capture, output, evaluation=None, radius_m=.11):
    metadata = json.loads((capture/"metadata.json").read_text())
    if metadata.get("depth_source") != "zed_stereo_sdk" or metadata.get("depth_units") != "m":
        raise ValueError("Requires ZED SDK stereo depth in metres")
    calibration = {**metadata["calibration"], "baseline_m": metadata["baseline_m"]}
    for eye in ("left", "right"):
        values = calibration[eye]
        if not all(np.isfinite(values[key]) for key in ("fx", "fy", "cx", "cy")) or min(values["fx"], values["fy"]) <= 0:
            raise ValueError("Invalid rectified calibration")
    if not np.isfinite(calibration["baseline_m"]) or calibration["baseline_m"] <= 0:
        raise ValueError("Invalid baseline")
    output.mkdir(parents=True, exist_ok=False)
    (output/"masks").mkdir()
    predictions = []
    frame_records = {}
    for frame in map(json.loads, (capture/"frames.jsonl").read_text().splitlines()):
        if frame["frame_index"] in frame_records:
            raise ValueError("Capture frame indices must be unique")
        frame_records[frame["frame_index"]] = frame
        width, height = frame["width"], frame["height"]
        images = [np.asarray(Image.open(capture/frame[f"{side}_path"]).convert("RGB")) for side in ("left", "right")]
        if any(image.shape[:2] != (height, width) for image in images):
            raise ValueError("Image size differs from frame record")
        measures = []
        for kind in ("depth", "confidence"):
            payload = np.fromfile(capture/frame[f"{kind}_path"], dtype="<f4")
            if payload.size != width*height:
                raise ValueError(f"Invalid {kind} payload size")
            measures.append(payload.reshape(height, width))
        result = process_frame(*images, *measures, calibration, radius_m)
        result["frame_index"] = frame["frame_index"]
        result["sdk_timestamp_ns"] = frame["sdk_timestamp_ns"]
        for side in ("left", "right"):
            mask_path = f"masks/{frame['frame_index']:06d}_{side}.png"
            Image.fromarray(result[side].pop("mask")).save(output/mask_path)
            result[side]["mask_path"] = mask_path
        predictions.append(result)
    (output/"predictions.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in predictions))
    summary = {"detector": "orange_hsv_contour", "radius_prior_m": radius_m,
               "frames": len(predictions), "ground_truth_used_by_predictor": False,
               "provenance_verified_by_this_script": False,
               "source_capture": str(capture.resolve()), "calibration": calibration}
    if evaluation:
        def read_surface(frame_index):
            frame = frame_records[frame_index]
            shape = (frame["height"], frame["width"])
            arrays = [np.fromfile(capture/frame[f"{kind}_path"], dtype="<f4").reshape(shape)
                      for kind in ("depth", "confidence")]
            mask = np.asarray(Image.open(output/f"masks/{frame_index:06d}_left.png")) > 0
            return *arrays, mask
        summary["evaluation"] = evaluate(predictions, [json.loads(line) for line in evaluation.read_text().splitlines()],
                                         calibration, read_surface, radius_m)
        fields = ["depth_label_m", "angle_label", "expected_visible", "method", "frames", "detection_rate",
                  "false_positive_rate", "centroid_mae_px", "sdk_surface_mae_m", "sdk_surface_rmse_m",
                  "mean_mask_fraction", "mean_depth_valid_fraction", "valid", "false_positives",
                  "depth_mae_m", "depth_rmse_m", "center_mae_m", "center_rmse_m"]
        with (output/"metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for group in summary["evaluation"]["groups"]:
                for method, metrics in group["methods"].items():
                    row = {key: group[key] for key in fields if key in group}
                    row.update({key: metrics[key] for key in fields if key in metrics})
                    writer.writerow(row | {"method": method})
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--radius-m", type=float, default=.11)
    args = parser.parse_args()
    result = run(args.capture, args.output, args.evaluation, args.radius_m)
    print(json.dumps({"frames": result["frames"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
