"""Measure the diagnostic orange sphere in delivered SDK PNGs and depth."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .verify_zed_sdk_stream import decode_frame_id
except ImportError:
    from verify_zed_sdk_stream import decode_frame_id


def orange_component(rgb, predicted, radius):
    """Find the largest orange connected component near the projected sphere."""
    x, y = predicted
    span = int(np.ceil(1.5 * radius + 10))
    x0, x1 = max(0, int(x) - span), min(rgb.shape[1], int(x) + span + 1)
    y0, y1 = max(0, int(y) - span), min(rgb.shape[0], int(y) + span + 1)
    crop = rgb[y0:y1, x0:x1].astype(float)
    r, g, b = crop.transpose(2, 0, 1)
    mask = (r > 200) & (g > 115) & (g < 170) & (b < 110) & (r - g > 55) & (g - b > 40)
    remaining = mask.copy()
    largest = []
    for yy, xx in zip(*np.where(mask)):
        if not remaining[yy, xx]:
            continue
        todo, component = deque([(yy, xx)]), []
        remaining[yy, xx] = False
        while todo:
            cy, cx = todo.popleft()
            component.append((cy + y0, cx + x0))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    todo.append((ny, nx))
        if len(component) > len(largest):
            largest = component
    if len(largest) < np.pi * radius ** 2 * 0.4:
        raise ValueError("Delivered PNG has insufficient orange sphere support")
    pixels = np.asarray(largest)
    return np.array([pixels[:, 1].mean(), pixels[:, 0].mean()]), pixels


def verify_geometry(capture, source):
    capture, source = Path(capture), Path(source)
    metadata = json.loads((source / "metadata.json").read_text())
    sdk = json.loads((capture / "metadata.json").read_text())
    source_rows = [json.loads(line) for line in (source / "source_frames.jsonl").read_text().splitlines()]
    authored = {row["visual_frame_id"]: row for row in source_rows}
    frames = [json.loads(line) for line in (capture / "frames.jsonl").read_text().splitlines()]
    measured = []
    for frame in frames:
        depth = np.fromfile(capture / frame["depth_path"], dtype="<f4").reshape(frame["height"], frame["width"])
        for side in ("left", "right"):
            rgb = np.array(Image.open(capture / frame[f"{side}_path"]).convert("RGB"))
            visual_id = decode_frame_id(rgb, metadata["frame_id_marker"]["pixel_centres"][side])
            state = authored[visual_id]["acquisition"]
            pose = np.asarray(state["world_from_usd_camera_row_matrix"][side])
            local = np.r_[state["ball_world_position_m"], 1] @ np.linalg.inv(pose)
            calibration = sdk["calibration"][side]
            predicted = np.array([calibration["cx"] + calibration["fx"] * local[0] / -local[2],
                                  calibration["cy"] - calibration["fy"] * local[1] / -local[2]])
            radius = calibration["fx"] * state["ball_radius_m"] / -local[2]
            centre, pixels = orange_component(rgb, predicted, radius)
            error = float(np.linalg.norm(centre - predicted))
            if error > 3:
                raise ValueError(f"Delivered sphere centre disagrees with archived pose by {error:.3f}px")
            diagnostic = None
            if side == "left":
                values = depth[pixels[:, 0], pixels[:, 1]]
                foreground = np.isfinite(values) & (values > 1) & (values < 1.5)
                diagnostic = {"foreground_fraction": float(foreground.mean()),
                              "foreground_median_m": float(np.median(values[foreground])) if foreground.any() else None}
            measured.append({"frame_index": frame["frame_index"], "side": side, "visual_id": visual_id,
                             "predicted_xy_px": predicted.tolist(), "measured_xy_px": centre.tolist(),
                             "centre_error_px": error, "orange_pixels": len(pixels), "sdk_depth": diagnostic})
    ranges = {}
    for side in ("left", "right"):
        rows = [row for row in measured if row["side"] == side]
        predicted_range = float(np.ptp([row["predicted_xy_px"][0] for row in rows]))
        actual_range = float(np.ptp([row["measured_xy_px"][0] for row in rows]))
        if abs(actual_range - predicted_range) > 3:
            raise ValueError("Delivered sphere motion range disagrees with archived camera/ball motion")
        ranges[side] = {"predicted_horizontal_range_px": predicted_range, "measured_horizontal_range_px": actual_range}
    return {"delivered_scene_geometry_verified": True,
            "scope": "Sphere PNG geometry selected by delivered visual identity; SDK timestamp association is a separate gate",
            "frames": len(frames), "maximum_centre_error_px": max(row["centre_error_px"] for row in measured),
            "motion_ranges": ranges, "per_eye_frame": measured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = verify_geometry(args.capture, args.source)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in report.items() if key != "per_eye_frame"}))


if __name__ == "__main__":
    main()
