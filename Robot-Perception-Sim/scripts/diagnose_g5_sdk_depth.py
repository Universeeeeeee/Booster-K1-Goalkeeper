"""Offline SDK depth diagnosis using delivered pixels and separately joined labels.

Fixed generic image matching is a diagnostic, not a replacement SDK measurement.
No labels, authored positions or depths enter the image matcher.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def records(path):
    return list(map(json.loads, path.read_text().splitlines()))


def quantiles(values):
    values = values[np.isfinite(values)]
    return np.percentile(values, [10, 50, 90]).tolist() if values.size else None


def read_measure(root, frame, name):
    return np.fromfile(root / frame[f"{name}_path"], dtype="<f4").reshape(frame["height"], frame["width"])


def orange_mask(bgr):
    return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                       np.array([5, 100, 60]), np.array([35, 255, 255])) > 0


def experiment(source, capture, evaluation):
    source_meta = json.loads((source / "metadata.json").read_text())
    sdk = json.loads((capture / "metadata.json").read_text())
    labels = {row["frame_index"]: row for row in records(evaluation)} if evaluation else {}
    cameras = source_meta["cameras"]
    poses = [np.asarray(cameras[side]["world_from_usd_camera_row_matrix"]) for side in ("left", "right")]
    baseline = float(np.linalg.norm(poses[0][3, :3] - poses[1][3, :3]))
    fb = sdk["calibration"]["left"]["fx"] * sdk["baseline_m"]
    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=5,
                                  P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                  disp12MaxDiff=1, speckleWindowSize=100, speckleRange=2)
    rows, representatives = [], set()
    for frame in records(capture / "frames.jsonl"):
        left = cv2.imread(str(capture / frame["left_path"]))
        depth = read_measure(capture, frame, "depth")
        confidence = read_measure(capture, frame, "confidence")
        mask = orange_mask(left)
        valid = np.isfinite(depth) & (depth > 0)
        background = depth[380:450, 200:760]
        background = background[np.isfinite(background) & (background > 0)]
        row = {"frame_index": frame["frame_index"], "orange_pixels": int(mask.sum()),
               "sdk_orange_depth_p10_p50_p90_m": quantiles(depth[mask & valid]),
               "sdk_orange_confidence_p10_p50_p90": quantiles(confidence[mask]),
               "background_depth_p10_p50_p90_m": quantiles(background)}
        label = labels.get(frame["frame_index"])
        if label:
            case = (label["depth_label_m"], label["angle_label"])
            row.update(depth_label_m=case[0], angle_label=case[1], expected_visible=label["expected_visible"])
            if label["expected_visible"] and case not in representatives:
                representatives.add(case)
                right = cv2.imread(str(capture / frame["right_path"]))
                disparity = matcher.compute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)) / 16.0
                matched = mask & valid & (disparity > 0)
                row.update(rgb_sgbm_compared_pixels=int(matched.sum()),
                           rgb_sgbm_disparity_p10_p50_p90_px=quantiles(disparity[matched]),
                           rgb_sgbm_surface_z_p10_p50_p90_m=quantiles(fb / disparity[matched]),
                           sdk_implied_disparity_p10_p50_p90_px=quantiles(fb / depth[matched]))
        rows.append(row)
    return {"capture": str(capture), "sdk_mode": sdk["depth_mode"],
            "texture": source_meta.get("acquisition", {}).get("sphere_texture", "fine_orange"),
            "source_fx_px": cameras["left"]["fx_px"], "sdk_fx_px": sdk["calibration"]["left"]["fx"],
            "source_baseline_m": baseline, "sdk_baseline_m": sdk["baseline_m"],
            "expected_background_z_m": source_meta["plane_optical_z_m"],
            "source_files": source_meta["acquisition"]["source_files"],
            "metadata_sha256": hashlib.sha256((capture / "metadata.json").read_bytes()).hexdigest(),
            "per_frame": rows}


def disparity_consistency(capture):
    sdk = json.loads((capture / "metadata.json").read_text())
    expected = sdk["calibration"]["left"]["fx"] * sdk["baseline_m"]
    rows = []
    for frame in records(capture / "frames.jsonl"):
        depth, disparity = (read_measure(capture, frame, name) for name in ("depth", "disparity"))
        mask = orange_mask(cv2.imread(str(capture / frame["left_path"])))
        valid = mask & np.isfinite(depth) & np.isfinite(disparity) & (depth > 0)
        product = depth[valid] * disparity[valid]
        rows.append({"frame_index": frame["frame_index"], "valid_orange_pixels": int(valid.sum()),
                     "sdk_depth_p10_p50_p90_m": quantiles(depth[valid]),
                     "sdk_disparity_p10_p50_p90_px": quantiles(disparity[valid]),
                     "signed_depth_times_disparity_p10_p50_p90": quantiles(product),
                     "maximum_absolute_product_residual_m_px": float(np.max(abs(abs(product) - expected)))
                     if product.size else None})
    return {"fx_times_baseline_m_px": expected, "sdk_mode": sdk["depth_mode"], "per_frame": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", nargs=3, metavar=("SENDER", "CAPTURE", "EVALUATION"), default=[])
    parser.add_argument("--plane", nargs=2, metavar=("SENDER", "CAPTURE"))
    parser.add_argument("--disparity-control", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {"scope": "Offline diagnosis; fixed generic SGBM uses only delivered grayscale stereo pixels. No predictor threshold changes.",
              "experiments": [experiment(*map(Path, paths)) for paths in args.experiment]}
    if args.plane:
        report["plane_control"] = experiment(*map(Path, args.plane), None)
    if args.disparity_control:
        report["sdk_depth_disparity_consistency"] = disparity_consistency(args.disparity_control)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
