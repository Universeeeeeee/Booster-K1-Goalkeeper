"""Independently read a saved SDK stream; does not load Isaac or the ZED SDK.

This verifies delivered data, not SDK provenance by itself. Pair the report
with the official sender source, runtime logs and renderer-GT isolation audit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def associate_source_frames(frames, source_frames):
    """Match the vendor's truncated simulation-nanosecond stamps, without fitting.

    The official v3.2 RGB bridge sends simulationTime * 1e9. This checks that
    contract and counts skipped source frames; it does not identify pixels.
    """
    instrumented = all("zed_input_sim_time_s" in row for row in source_frames)
    if not instrumented and any("zed_input_sim_time_s" in row for row in source_frames):
        raise ValueError("Bridge input time must be recorded for every source frame")
    # World-state rows are unique. Post-step reads of an asynchronous graph's
    # input may repeat; they are observations, not a log of stream emissions.
    source_ns = np.asarray([int(row["sim_time_s"] * 1e9) for row in source_frames], dtype=np.int64)
    sdk_ns = np.asarray([row["sdk_timestamp_ns"] for row in frames], dtype=np.int64)
    if len(source_ns) < 2 or np.any(np.diff(source_ns) <= 0):
        raise ValueError("Source simulation timestamps must strictly increase")
    if len(sdk_ns) < 2 or np.any(np.diff(sdk_ns) <= 0):
        raise ValueError("SDK timestamps must strictly increase")
    if sdk_ns[0] < source_ns[0] - 2 or sdk_ns[-1] > source_ns[-1] + 2:
        raise ValueError("SDK timestamp outside recorded source range")
    indices = np.searchsorted(source_ns, sdk_ns).clip(1, len(source_ns) - 1)
    previous = indices - 1
    indices = np.where(abs(source_ns[previous] - sdk_ns) <= abs(source_ns[indices] - sdk_ns),
                       previous, indices)
    residuals = sdk_ns - source_ns[indices]
    if np.max(np.abs(residuals)) > 2:
        raise ValueError("SDK/source simulation timestamps do not match within 2 ns")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("SDK frames must map to distinct increasing source frames")
    source_ids = [source_frames[index]["source_frame"] for index in indices]
    all_source_ids = [row["source_frame"] for row in source_frames]
    if any(b != a + 1 for a, b in zip(all_source_ids, all_source_ids[1:])):
        raise ValueError("Recorded source frame IDs must be contiguous")
    for row in source_frames:
        if instrumented and "og_sim_time_s" in row:
            if abs(row["og_sim_time_s"] - row["zed_input_sim_time_s"]) > 2e-9:
                raise ValueError("OmniGraph time differs from the bridge input time")
    graph_candidates = [[] for _ in frames]
    repeated_graph_samples = None
    if instrumented:
        graph_ns = np.asarray([int(row["zed_input_sim_time_s"] * 1e9) for row in source_frames], dtype=np.int64)
        if np.any(np.diff(graph_ns) < 0):
            raise ValueError("Observed bridge clock regressed")
        repeated_graph_samples = int(np.count_nonzero(np.diff(graph_ns) == 0))
        graph_candidates = [[source_frames[i]["source_frame"] for i in np.flatnonzero(abs(graph_ns - stamp) <= 2)]
                            for stamp in sdk_ns]
    sdk_host = np.asarray([row["host_monotonic_ns"] for row in frames], dtype=np.int64)
    source_host = np.asarray([source_frames[index]["host_monotonic_ns"] for index in indices],
                             dtype=np.int64)
    if np.any(np.diff(sdk_host) <= 0):
        raise ValueError("Receiver monotonic clock regressed")
    span = int(indices[-1] - indices[0] + 1)
    return {"contract": "official v3.2 simulationTime seconds truncated to nanoseconds",
            "timestamp_mapping_verified": False, "pixel_identity_verified": False,
            "source_clock_field": "sim_time_s; numeric match requires independent visual identity",
            "repeated_graph_input_observations": repeated_graph_samples,
            "graph_observation_source_frame_candidates": graph_candidates,
            "post_step_world_minus_bridge_s": float(np.median(
                [row["sim_time_s"] - row["zed_input_sim_time_s"] for row in source_frames])) if instrumented else None,
            "max_timestamp_residual_ns": int(np.max(np.abs(residuals))),
            "matched_source_frame_ids": source_ids,
            "source_frames_in_capture_span": span,
            "unobserved_source_frames": span - len(frames),
            "delivered_fraction_in_capture_span": len(frames) / span,
            "sdk_host_fps": (len(frames) - 1) * 1e9 / int(sdk_host[-1] - sdk_host[0]),
            "grab_minus_post_step_source_record_ms": dict(zip(
                ("min", "median", "max"),
                np.percentile((sdk_host - source_host) / 1e6, (0, 50, 100)).tolist())),
            "host_latency_scope": "same-host monotonic comparison; may be negative because graph dispatch occurs inside step; not transport or exposure latency"}


def decode_frame_id(rgb, centres):
    """Read the diagnostic 1010/uint16/parity/01 strip from actual pixels."""
    if len(centres) != 23:
        raise ValueError("Frame marker requires 23 cell centres")
    values = []
    for x, y in centres:
        x, y = round(x), round(y)
        if not (2 <= x < rgb.shape[1] - 2 and 2 <= y < rgb.shape[0] - 2):
            raise ValueError("Frame marker outside image")
        values.append(float(np.median(rgb[y-2:y+3, x-2:x+3, :3].mean(axis=2))))
    dark = float(np.median([values[i] for i in (1, 3, 21)]))
    bright = float(np.median([values[i] for i in (0, 2, 22)]))
    if bright - dark < 40:
        raise ValueError("Frame marker has insufficient contrast")
    threshold = (bright + dark) / 2
    if min(abs(value - threshold) for value in values) < 0.15 * (bright - dark):
        raise ValueError("Ambiguous frame marker cell")
    bits = [int(value > threshold) for value in values]
    if bits[:4] != [1, 0, 1, 0] or bits[-2:] != [0, 1]:
        raise ValueError("Frame marker sentinel mismatch")
    if sum(bits[4:20]) % 2 != bits[20]:
        raise ValueError("Frame marker parity mismatch")
    value = 0
    for bit in bits[4:20]:
        value = (value << 1) | bit
    return value


def verify_visual_ids(root, frames, source_frames, marker, association):
    if len(frames) != len(association["matched_source_frame_ids"]):
        raise ValueError("Every received frame requires a source association")
    sources = {row["source_frame"]: row for row in source_frames}
    visual_sources = {row["visual_frame_id"]: row for row in source_frames}
    if len(visual_sources) != len(source_frames):
        raise ValueError("Visual frame IDs must not wrap within this diagnostic recording")
    decoded = []
    graph_observations = association.get("graph_observation_source_frame_candidates", [[] for _ in frames])
    for frame, source_id, observed_rows in zip(frames, association["matched_source_frame_ids"], graph_observations):
        ids = {}
        for side in ("left", "right"):
            with Image.open(root / frame[f"{side}_path"]) as image:
                ids[side] = decode_frame_id(np.asarray(image.convert("RGB")),
                                           marker["pixel_centres"][side])
        if ids["left"] != ids["right"]:
            raise ValueError(f"Frame {frame['frame_index']} eye IDs disagree: {ids}")
        if ids["left"] not in visual_sources:
            raise ValueError("Image frame ID is absent from source recording")
        image_source = visual_sources[ids["left"]]
        image_id = image_source["source_frame"]
        if image_id > source_id:
            raise ValueError("Image source frame is newer than its SDK dispatch timestamp")
        if decoded and image_id <= decoded[-1]["image_source_frame"]:
            raise ValueError("Image source IDs must be unique and increasing")
        image_time_residual = frame["sdk_timestamp_ns"] - int(image_source["sim_time_s"] * 1e9)
        if abs(image_time_residual) > 2:
            raise ValueError("SDK timestamp does not match the visually identified source world state")
        decoded.append({"frame_index": frame["frame_index"],
                        "timestamp_world_source_frame": source_id, "image_source_frame": image_id,
                        "left_visual_id": ids["left"], "right_visual_id": ids["right"],
                        "graph_observation_source_frame_candidates": observed_rows,
                        "sdk_minus_image_world_time_ns": image_time_residual,
                        "marker_authored_sim_time_s": image_source.get("marker_authored_sim_time_s")})
    lags = [index - row["image_source_frame"] for row in decoded
            for index in row["graph_observation_source_frame_candidates"]]
    return {"pixel_identity_verified": True, "frames_verified": len(decoded),
            "sampled_graph_minus_image_source_frames": {"min": min(lags), "max": max(lags),
                                                         "median": float(np.median(lags))} if lags else None,
            "max_sdk_image_time_residual_ns": max(abs(row["sdk_minus_image_world_time_ns"]) for row in decoded),
            "capture_time_scope": "authored marker-state identity; not a measured exposure timestamp",
            "per_frame": decoded}


def associate_render_observations(root, frames, source_frames, marker, observations=None):
    """Associate actual source pixels and SDK pixels through the paired-render ID.

    Render reference time is a separate clock. No offset to World or host time
    is fitted: each observed image identity selects its authored state record.
    """
    authored = {row["visual_frame_id"]: row for row in source_frames}
    if len(authored) != len(source_frames):
        raise ValueError("Visual IDs wrapped within the source recording")
    references = {}
    invalid = 0
    for row in source_frames if observations is None else observations:
        if row.get("reference_conflict"):
            raise ValueError("Source reports a conflicting render reference")
        if not row.get("observation_valid"):
            invalid += 1
            continue
        if "reference_time" in row:
            # A Replicator Writer packet supplies both RGBs and this reference.
            before = after = row["reference_time"]
        else:
            # Retained only to reproduce failure of the earlier polling probe.
            before, after = row["observed_ref_before"], row["observed_ref_after"]
        if before != after or len(before) != 2 or before[1] <= 0:
            raise ValueError("Source reference changed during an allegedly valid observation")
        left, right = row["observed_left_visual_id"], row["observed_right_visual_id"]
        if left != right or left not in authored:
            raise ValueError("Source observed eye IDs disagree or have no authored state")
        stamp = int(before[0] / before[1] * 1e9)
        if stamp in references and references[stamp]["visual_id"] != left:
            raise ValueError("One render reference identifies conflicting source images")
        if stamp not in references:
            references[stamp] = {"visual_id": left, "rational_reference": before, "observed_rows": []}
        references[stamp]["observed_rows"].append(row.get("observation_index", row.get("source_frame")))
    keys = np.asarray(sorted(references), dtype=np.int64)
    if len(keys) == 0:
        raise ValueError("No valid source render observations")
    matched = []
    for frame in frames:
        candidates = keys[abs(keys - frame["sdk_timestamp_ns"]) <= 2]
        if len(candidates) != 1:
            raise ValueError("SDK render reference has no unique observed source image")
        ref_ns = int(candidates[0])
        observed = references[ref_ns]
        ids = {}
        for side in ("left", "right"):
            with Image.open(root / frame[f"{side}_path"]) as image:
                ids[side] = decode_frame_id(np.asarray(image.convert("RGB")), marker["pixel_centres"][side])
        if ids["left"] != ids["right"] or ids["left"] != observed["visual_id"]:
            raise ValueError(f"SDK frame {frame['frame_index']} pixels disagree with its paired render reference")
        source = authored[ids["left"]]
        if matched and source["source_frame"] <= matched[-1]["image_source_frame"]:
            raise ValueError("SDK image source identities must uniquely increase")
        matched.append({"frame_index": frame["frame_index"], "sdk_render_reference_ns": frame["sdk_timestamp_ns"],
                        "render_reference_residual_ns": frame["sdk_timestamp_ns"] - ref_ns,
                        "rational_reference": observed["rational_reference"],
                        "source_observation_indices": observed["observed_rows"],
                        "image_source_frame": source["source_frame"],
                        "left_visual_id": ids["left"], "right_visual_id": ids["right"],
                        "source_world_state_time_s": source["sim_time_s"],
                        "marker_authored_sim_time_s": source.get("marker_authored_sim_time_s")})
    host = [frame["host_monotonic_ns"] for frame in frames]
    if len(host) < 2 or any(b <= a for a, b in zip(host, host[1:])):
        raise ValueError("Receiver monotonic clock must increase")
    span = matched[-1]["image_source_frame"] - matched[0]["image_source_frame"] + 1
    return {"timestamp_mapping_verified": True, "pixel_identity_verified": True,
            "sdk_timestamp_domain": "paired_render_reference; not World or Unix time",
            "world_mapping": "observed pixel ID to authored state; no clock offset fit",
            "matched_frames": len(matched), "valid_unique_source_references": len(references),
            "invalid_source_observations_preserved": invalid,
            "unobserved_source_state_updates": span - len(frames),
            "max_reference_residual_ns": max(abs(row["render_reference_residual_ns"]) for row in matched),
            "sdk_host_fps": (len(host) - 1) * 1e9 / (host[-1] - host[0]),
            "per_frame": matched}


def read_capture(root, expected_frames):
    metadata = json.loads((root / "metadata.json").read_text())
    frames = [json.loads(line) for line in (root / "frames.jsonl").read_text().splitlines()]
    if len(frames) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, found {len(frames)}")
    if metadata["depth_source"] != "zed_stereo_sdk" or metadata["depth_units"] != "m":
        raise ValueError("Capture must declare SDK stereo depth in metres")
    timestamps = [frame["sdk_timestamp_ns"] for frame in frames]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("SDK frame timestamps must strictly increase")
    rows = []
    disparity_checks = []
    for index, frame in enumerate(frames):
        if frame["frame_index"] != index:
            raise ValueError("Frame indices must be contiguous")
        if frame["camera_timestamp_after_retrieval_ns"] != frame["sdk_timestamp_ns"]:
            raise ValueError("Camera frame changed during retrieval")
        width, height = frame["width"], frame["height"]
        arrays = {}
        for side in ("left", "right"):
            path = root / frame[f"{side}_path"]
            with Image.open(path) as image:
                image.load()
                arrays[side] = np.asarray(image.convert("RGB"))
            if arrays[side].shape != (height, width, 3):
                raise ValueError(f"Image dimensions disagree for {path}")
        for kind in ("depth", "confidence"):
            path = root / frame[f"{kind}_path"]
            if path.stat().st_size != width * height * 4:
                raise ValueError(f"Wrong float32 payload size: {path}")
            arrays[kind] = np.fromfile(path, dtype="<f4").reshape(height, width)
        valid = np.isfinite(arrays["depth"]) & (arrays["depth"] > 0)
        if int(valid.sum()) != frame["valid_depth_count"]:
            raise ValueError("Recomputed valid depth count disagrees with SDK record")
        roi = arrays["depth"][height // 4:3 * height // 4, width // 4:3 * width // 4]
        roi_valid = np.isfinite(roi) & (roi > 0)
        if not roi_valid.any():
            raise ValueError(f"No valid central-plane depth in frame {index}")
        rows.append({"frame_index": index, "valid_fraction": float(valid.mean()),
                     "plane_valid_fraction": float(roi_valid.mean()),
                     "plane_median_m": float(np.median(roi[roi_valid])),
                     "left_std": float(arrays["left"].std()),
                     "right_std": float(arrays["right"].std())})
        if index in (0, len(frames) // 2, len(frames) - 1):
            # Independently estimate disparity from delivered images. This is a
            # static textured-plane check, not a replacement depth estimator.
            left = arrays["left"].astype(np.float32).mean(axis=2)
            right = arrays["right"].astype(np.float32).mean(axis=2)
            y0, y1, x0, x1 = height // 3, 2 * height // 3, width // 3, 2 * width // 3
            scores = []
            for disparity in range(1, 65):
                a = left[y0:y1, x0:x1]
                b = right[y0:y1, x0 - disparity:x1 - disparity]
                scores.append(float(np.mean((a - b) ** 2)))
            disparity_checks.append({"frame_index": index,
                                     "integer_disparity_px": int(np.argmin(scores)) + 1,
                                     "mean_squared_difference": min(scores)})
    intervals = np.diff(np.asarray(timestamps, dtype=np.int64)) / 1e9
    return {"frames": len(frames), "all_files_decoded": True,
            "timestamp_interval_s": {"min": float(intervals.min()),
                                      "median": float(np.median(intervals)),
                                      "max": float(intervals.max())},
            "depth_source_declared": metadata["depth_source"],
            "provenance_verified_by_this_script": False,
            "frame_association": "single-grab timestamp guard; no independent Mat timestamp API",
            "plane_median_m": float(np.median([row["plane_median_m"] for row in rows])),
            "minimum_plane_valid_fraction": min(row["plane_valid_fraction"] for row in rows),
            "disparity_checks": disparity_checks, "per_frame": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--expected-plane-m", type=float, default=2.0)
    parser.add_argument("--plane-tolerance-m", type=float, default=0.2)
    parser.add_argument("--source", type=Path,
                        help="Same-host sender directory with source_frames.jsonl")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 2 or args.plane_tolerance_m <= 0:
        parser.error("At least two frames and a positive tolerance are required")
    report = read_capture(args.capture, args.frames)
    if args.source:
        frames = [json.loads(line) for line in (args.capture / "frames.jsonl").read_text().splitlines()]
        source = [json.loads(line) for line in (args.source / "source_frames.jsonl").read_text().splitlines()]
        source_metadata = json.loads((args.source / "metadata.json").read_text())
        if source_metadata.get("sdk_timestamp_source") == "paired_render_sync_rational_seconds":
            observations = None
            if "render_observations_file" in source_metadata:
                observations = [json.loads(line) for line in
                                (args.source / source_metadata["render_observations_file"]).read_text().splitlines()]
            report["source_association"] = associate_render_observations(
                args.capture, frames, source, source_metadata["frame_id_marker"], observations)
        else:
            report["source_association"] = associate_source_frames(frames, source)
        if "frame_id_marker" in source_metadata and not report["source_association"]["pixel_identity_verified"]:
            report["visual_frame_identity"] = verify_visual_ids(
                args.capture, frames, source, source_metadata["frame_id_marker"], report["source_association"])
            report["source_association"]["pixel_identity_verified"] = True
            report["source_association"]["timestamp_mapping_verified"] = True
    error = abs(report["plane_median_m"] - args.expected_plane_m)
    report.update(expected_plane_m=args.expected_plane_m, absolute_plane_error_m=error,
                  plane_tolerance_m=args.plane_tolerance_m,
                  plane_accuracy_pass=error <= args.plane_tolerance_m)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "per_frame"}, indent=2))
    if not report["plane_accuracy_pass"]:
        raise SystemExit("Known-plane depth tolerance failed")


if __name__ == "__main__":
    main()
