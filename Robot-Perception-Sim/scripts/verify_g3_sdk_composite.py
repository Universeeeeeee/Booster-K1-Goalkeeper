"""Verify immutable official-bridge SDK publications from saved paired subframes.

The SDK clock uses the declared authored window start or midpoint. Its known
model offset to optical midpoint is never fitted from observations. Neither is
interpreted as the renderer NewFrame clock or native World physics clock.
"""

import argparse
import configparser
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .verify_zed_sdk_stream import read_capture, decode_frame_id
    from .verify_g3_sdk_geometry import orange_component
except ImportError:
    from verify_zed_sdk_stream import read_capture, decode_frame_id
    from verify_g3_sdk_geometry import orange_component


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    return Fraction(value["numerator"], value["denominator"])


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rgba(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGBA"))


def project(state, side, calibration):
    pose = np.asarray(state["world_from_usd_camera_row_matrix"][side])
    require(pose.shape == (4, 4) and np.isfinite(pose).all(), "Invalid archived camera transform")
    local = np.r_[state["ball_world_position_m"], 1] @ np.linalg.inv(pose)
    require(local[2] < 0, "Diagnostic sphere is behind camera")
    return np.array([calibration["cx"] + calibration["fx"] * local[0] / -local[2],
                     calibration["cy"] - calibration["fy"] * local[1] / -local[2]])



def verify_nominal_calibration(capture, sdk, source_metadata):
    """Read the receiver's copied virtual-camera seed and actual SDK calibration."""
    capture = Path(capture).resolve()
    relative = sdk.get("calibration_seed_relative_path", "")
    require(bool(relative), "SDK 5.1 requires the recorded nominal calibration seed")
    path = (capture / relative).resolve()
    require(path.is_relative_to(capture), "Calibration seed path escapes receiver recording")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    canonical = Path(__file__).resolve().parents[1] / "config/cameras/SN45626933.conf"
    require(digest == sdk["calibration_declared_sha256"] == hashlib.sha256(canonical.read_bytes()).hexdigest(),
            "Nominal calibration copied bytes disagree with declared or repository digest")
    require(sdk["calibration_hardware_verified"] is False and sdk["serial_number"] == 45626933,
            "Nominal seed must describe the virtual serial, not hardware calibration")
    profile = configparser.ConfigParser()
    profile.read_string(content.decode())
    require(source_metadata["resolution"] == [960, 600] and [sdk["width"], sdk["height"]] == [960, 600],
            "Nominal SVGA calibration requires matching 960x600 source and SDK images")
    for side in ("left", "right"):
        section = profile[f"{side.upper()}_CAM_SVGA"]
        camera = source_metadata["cameras"][side]
        require(abs(camera["fx_px"] - 960 * camera["focal_length_usd"] / camera["horizontal_aperture_usd"]) <= 1e-6
                and abs(camera["fy_px"] - 600 * camera["focal_length_usd"] / camera["vertical_aperture_usd"]) <= 1e-6,
                "Source intrinsics disagree with recorded USD focal/aperture parameters")
        for key in ("fx", "fy", "cx", "cy"):
            nominal = section.getfloat(key)
            require(abs(sdk["calibration"][side][key] - nominal) <= 1e-3,
                    f"SDK runtime {side} {key} disagrees with nominal SVGA seed")
            require(abs(camera[f"{key}_px"] - nominal) <= 1e-3,
                    f"Source USD {side} {key} disagrees with nominal SVGA seed")
        require(abs(camera["horizontal_aperture_offset_usd"]) <= 1e-9
                and abs(camera["vertical_aperture_offset_usd"]) <= 1e-9,
                "Nominal SVGA seed requires centred source film apertures")
    baseline = profile["STEREO"].getfloat("Baseline") / 1000
    require(abs(sdk["baseline_m"] - baseline) <= 1e-6, "SDK runtime baseline disagrees with nominal millimetre seed")
    left = np.asarray(source_metadata["cameras"]["left"]["world_from_usd_camera_row_matrix"])
    right = np.asarray(source_metadata["cameras"]["right"]["world_from_usd_camera_row_matrix"])
    displacement = (right[3, :3] - left[3, :3]) @ left[:3, :3].T
    require(np.allclose(displacement, [baseline, 0, 0], rtol=0, atol=1e-6)
            and np.allclose(left[:3, :3], right[:3, :3], rtol=0, atol=1e-8),
            "Source USD stereo separation disagrees with nominal parallel-camera seed")
    return {"profile": "nominal virtual SN45626933 SVGA; not hardware calibration",
            "sha256": digest, "copied_path": relative, "baseline_m": baseline,
            "source_and_sdk_K_verified": True}


def verify_composites(capture, source, expected_frames, codec_baseline=None):
    capture, source = Path(capture), Path(source)
    payload = read_capture(capture, expected_frames)
    frames, authored = rows(capture / "frames.jsonl"), rows(source / "source_frames.jsonl")
    meta, sdk = (json.loads((path / "metadata.json").read_text()) for path in (source, capture))
    config = meta["acquisition"]
    if codec_baseline is not None:
        require(codec_baseline["sdk_version"] == sdk["sdk_version"], "Static codec baseline SDK version differs from evaluated capture")
    nominal_calibration = None
    if sdk["sdk_version"].startswith("5.1") or sdk.get("calibration_seed_relative_path"):
        nominal_calibration = verify_nominal_calibration(capture, sdk, meta)
        if config["scenario"] == "static" and config["realized_exposure_s"] == 0:
            require(abs(payload["plane_median_m"] - 2) <= 0.2,
                    "Static SDK nominal-calibration plane differs from known 2m geometry")
    reference = config.get("sdk_timestamp_reference", "exposure_midpoint")
    require(reference in ("exposure_start", "exposure_midpoint"), "Unknown SDK presentation timestamp convention")
    expected_clock_source = ("immutable_composite_exposure_start" if reference == "exposure_start"
                             else "immutable_composite_exposure_midpoint")
    require(meta["sdk_timestamp_source"] == expected_clock_source,
            "SDK clock must declare its explicit composite presentation reference")
    source_digests = {}
    for name, description in config["source_files"].items():
        path = (source / description["path"]).resolve()
        require(path.is_relative_to(source.resolve()), "Snapshot path escapes source recording")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == description["sha256"], "Producer source snapshot digest mismatch")
        source_digests[name] = digest
    require(bool(source_digests), "Producer source snapshots are missing")
    if "sender" in source_digests:
        require(source_digests["sender"] == meta["sender_script_sha256"], "Sender snapshot differs from executed source hash")
    if "sdk_state" in source_digests:
        require(source_digests["sdk_state"] == config["state_module_sha256"], "State snapshot differs from executed module hash")
    duration = Fraction(str(config["realized_exposure_s"]))
    require(config["finite_exposure_supported"] is True, "Controlled buffer integration must be declared")
    require(config["exposure_model"] == ("encoded_rgba_box_average" if duration else "instantaneous_render_baseline"),
            "Exposure provenance mismatch")
    require(config["rolling_skew_s"] == 0, "Global exposure requires zero rolling skew")
    calibration_payload = {"cameras": meta["cameras"], "resolution": meta["resolution"], "right_shift_m": meta["right_shift_m"]}
    calibration_version = hashlib.sha256(json.dumps(calibration_payload, sort_keys=True).encode()).hexdigest()
    require(config["calibration_version"] == calibration_version, "Calibration version mismatch")
    for side in ("left", "right"):
        require(abs(sdk["calibration"][side]["fx"] - meta["cameras"][side]["fx_px"]) < 1e-3,
                "SDK runtime calibration differs from source render intrinsics")
    sdk_version = hashlib.sha256(json.dumps(
        {"calibration": sdk["calibration"], "resolution": meta["resolution"], "baseline_m": sdk["baseline_m"]}, sort_keys=True).encode()).hexdigest()
    observations = rows(source / meta["render_observations_file"])
    packet_map = {}
    for observation in observations:
        require(not observation.get("reference_conflict"), "Conflicting paired render reference")
        if observation.get("observation_valid") and "observed_left_subframe_id" in observation:
            sid = observation["observed_left_subframe_id"]
            if sid == observation.get("observed_right_subframe_id"):
                packet_map.setdefault((sid, tuple(observation["reference_time"])), []).append(observation)
    prepared_plans = {}
    if config.get("publication_mode") == "offline_render_then_replay":
        prepared_rows = rows(source / config["prepared_plan_file"])
        prepared_plans = {row["source_frame"]: row["acquisition"] for row in prepared_rows}
        require(len(prepared_plans) == len(prepared_rows) == config["prepared_window_count"],
                "Prepared exposure plan count disagrees with metadata")
    publications = {}
    seen_subframes = set()
    maximum_subframe_geometry_error = 0.0
    for author in authored:
        state = author["acquisition"]
        window = {key: rational(value) for key, value in state["window_rational"].items()}
        index = author["source_frame"]
        if prepared_plans:
            require(index in prepared_plans and {key: value for key, value in state.items() if key != "publication"} == prepared_plans[index],
                    "Published exposure differs from prepared source plan")
            require(max(sample["host_observed_monotonic_ns"] for sample in state["subframes"])
                    <= state["host_prepared_monotonic_ns"] <= state["publication"]["host_armed_monotonic_ns"],
                    "Offline preparation/publication causal bounds disagree")
        require(window["start_s"] == 1 + Fraction(index, meta["configured_fps"]), "Publication schedule drift")
        require(window["end_s"] - window["start_s"] == duration
                and window["mid_s"] == (window["start_s"] + window["end_s"]) / 2
                and 0 <= duration < Fraction(1, meta["configured_fps"]), "Invalid shared exposure window")
        for key, value in window.items():
            require(abs(state[f"t_exposure_{key}"] - float(value)) < 1e-10, "Float/rational exposure disagreement")
        reference_time = window["start_s"] if reference == "exposure_start" else window["mid_s"]
        declared_midpoint_offset = Fraction(str(config.get("sdk_to_exposure_midpoint_offset_s", 0)))
        require(declared_midpoint_offset == window["mid_s"] - reference_time,
                "Declared SDK-to-exposure-midpoint offset disagrees with exposure model")
        stamp = int(float(reference_time) * 1e9)
        require(state["sdk_composite_timestamp_ns"] == stamp and stamp not in publications,
                "Duplicate or false composite SDK timestamp")
        require(abs(state["publication"]["timestamp_s"] - float(reference_time)) < 1e-10,
                "Published bridge timestamp disagrees with declared exposure reference")
        subframes = state["subframes"]
        require(len(subframes) == config["samples_per_exposure"] and bool(subframes), "Incorrect integration sample count")
        images = {side: [] for side in ("left", "right")}
        for sample_index, sample in enumerate(subframes):
            require(sample["host_authored_monotonic_ns"] <= sample["host_observed_monotonic_ns"]
                    <= state["publication"]["host_armed_monotonic_ns"],
                    "Exposure subframe observation occurs outside publication causal bounds")
            sid = sample["subframe_id"]
            require(sid not in seen_subframes, "Reused subframe identity")
            seen_subframes.add(sid)
            target = window["start_s"] + Fraction(2 * sample_index + 1, 2 * len(subframes)) * duration
            require(rational(sample["target_s"]) == target, "Subframe is outside declared integration quadrature")
            require(rational(sample["state"]["motion_target_s"]) == target, "Archived pose time differs from subframe")
            sample_state = sample["state"]
            require(np.isfinite(sample_state["native_world_time_s"])
                    and sample_state["native_world_time_s"] == subframes[0]["state"]["native_world_time_s"],
                    "Native physics clock advanced inside held exposure rendering")
            phase = np.sin(2 * np.pi * config["motion_frequency_hz"] * float(target - 1))
            ball_x = config["ball_motion_amplitude_m"] * phase if config["scenario"] == "ball_motion" else 0
            camera_x = config["camera_motion_amplitude_m"] * phase if config["scenario"] == "camera_motion" else 0
            require(abs(sample_state["ball_optical_x_m"] - ball_x) <= 1e-9
                    and abs(sample_state["camera_optical_x_m"] - camera_x) <= 1e-9,
                    "Archived subframe motion disagrees with declared analytic trajectory")
            base = np.asarray(meta["cameras"]["left"]["world_from_usd_camera_row_matrix"])
            ball_world = np.array([ball_x, -0.15, -1.2, 1]) @ base
            require(np.allclose(sample_state["ball_world_position_m"], ball_world[:3], rtol=0, atol=1e-8),
                    "Archived ball world state cannot replay trajectory")
            displacement = np.array([camera_x, 0, 0]) @ base[:3, :3]
            for side in ("left", "right"):
                expected_pose = np.asarray(meta["cameras"][side]["world_from_usd_camera_row_matrix"]).copy()
                expected_pose[3, :3] += displacement
                require(np.allclose(sample_state["world_from_usd_camera_row_matrix"][side], expected_pose, rtol=0, atol=1e-8),
                        "Archived camera world pose cannot replay trajectory")
            packet_key = (sid, tuple(sample["reference_time"]))
            require(packet_key in packet_map, "Saved subframe has no same-packet paired RGB observation")
            packet = packet_map[packet_key][0]
            require(packet["observed_left_visual_id"] == packet["observed_right_visual_id"] == author["visual_frame_id"],
                    "Writer paired subframe has wrong held publication identity")
            for side in images:
                image = rgba(source / sample[f"{side}_path"])
                require(decode_frame_id(image[..., :3], meta["frame_id_marker"]["pixel_centres"][side]) == author["visual_frame_id"],
                        "Saved subframe primary pixels disagree with authored ID")
                require(decode_frame_id(image[..., :3], config["subframe_marker_pixel_centres"][side]) == sid,
                        "Saved subframe pixels disagree with secondary ID")
                predicted = project(sample_state, side, sdk["calibration"][side])
                radius = sdk["calibration"][side]["fx"] * sample_state["ball_radius_m"] / 1.2
                observed, _ = orange_component(image[..., :3], predicted, radius)
                geometry_error = float(np.linalg.norm(observed - predicted))
                require(geometry_error <= 3, "Actual subframe sphere pixels disagree with archived pose")
                maximum_subframe_geometry_error = max(maximum_subframe_geometry_error, geometry_error)
                images[side].append(image)
        for eye, side in enumerate(images):
            mean = np.rint(np.mean(images[side], axis=0)).astype(np.uint8)
            composite = rgba(source / state["composite_paths"][side])
            require(np.array_equal(mean, composite), "Saved composite is not the actual PNG subframe mean")
            digest = hashlib.sha256(composite.tobytes()).hexdigest()
            require(digest == state["publication"]["source_sha256_left_right"][eye], "Published CUDA source digest mismatch")
            require(composite.nbytes == state["publication"]["bytes_per_eye"], "Published buffer size mismatch")
        publications[stamp] = author
    joined, metrics, received = [], [], set()
    previous_receive = previous_saved = 0
    for frame in frames:
        stamp = frame["sdk_timestamp_ns"]
        candidates = [key for key in publications if abs(key - stamp) <= 2]
        require(len(candidates) == 1, "SDK timestamp has no unique explicit publication")
        publication_stamp = candidates[0]
        author = publications[publication_stamp]
        state, index = author["acquisition"], author["visual_frame_id"]
        require(author["source_frame"] not in received, "Repeated SDK publication")
        received.add(author["source_frame"])
        require(state["publication"]["host_armed_monotonic_ns"] <= frame["host_monotonic_ns"],
                "SDK arrival precedes composite publication")
        completion_key = ("host_after_execution_monotonic_ns"
                          if config.get("publication_execution") in ("synchronous_omnigraph_node_request", "synchronous_impulse_execution_graph")
                          else "host_after_render_monotonic_ns")
        require(state["publication"][completion_key] >= state["publication"]["host_armed_monotonic_ns"],
                "Publication execution completion precedes arming")
        require(frame["host_monotonic_ns"] > previous_receive
                and frame["host_saved_monotonic_ns"] >= frame["host_monotonic_ns"]
                and frame["host_saved_monotonic_ns"] > previous_saved, "Invalid arrival/processing monotonic bounds")
        previous_receive, previous_saved = frame["host_monotonic_ns"], frame["host_saved_monotonic_ns"]
        for side in ("left", "right"):
            delivered = rgba(capture / frame[f"{side}_path"])[..., :3].astype(float)
            require(decode_frame_id(delivered.astype(np.uint8), meta["frame_id_marker"]["pixel_centres"][side]) == index,
                    "SDK pixels disagree with explicit publication ID")
            composite = rgba(source / state["composite_paths"][side])[..., :3].astype(float)
            centres = np.array([project(sample["state"], side, sdk["calibration"][side]) for sample in state["subframes"]])
            centre = centres.mean(axis=0)
            radius = sdk["calibration"][side]["fx"] * state["subframes"][0]["state"]["ball_radius_m"] / 1.2
            margin = radius * 1.5 + np.ptp(centres[:, 0]) + 8
            x0, x1 = max(0, int(centre[0] - margin)), min(delivered.shape[1], int(centre[0] + margin + 1))
            y0, y1 = max(0, int(centre[1] - margin)), min(delivered.shape[0], int(centre[1] + margin + 1))
            roi = np.s_[y0:y1, x0:x1]
            observed_centre, sphere_pixels = orange_component(delivered.astype(np.uint8), centre, radius)
            centre_error = float(np.linalg.norm(observed_centre - centre))
            composite_centre, _ = orange_component(composite.astype(np.uint8), centre, radius)
            delivered_mean_geometry_error = float(np.linalg.norm(observed_centre - composite_centre))
            require(delivered_mean_geometry_error <= 2, "Delivered sphere geometry disagrees with actual integrated source pixels")
            if not duration:
                require(centre_error <= 3, "Delivered sphere geometry disagrees with archived pose")
            error = float(np.mean(np.abs(delivered[roi] - composite[roi])))
            instant_sample = state["subframes"][len(state["subframes"]) // 2]
            instant = rgba(source / instant_sample[f"{side}_path"])[..., :3].astype(float)
            instant_error = float(np.mean(np.abs(delivered[roi] - instant[roi])))
            source_instant_difference = float(np.mean(np.abs(composite[roi] - instant[roi])))
            marker_errors = []
            if duration and len(state["subframes"]) > 1:
                for x, y in config["subframe_marker_pixel_centres"][side]:
                    x, y = int(round(x)), int(round(y))
                    patch = np.s_[y - 2:y + 3, x - 2:x + 3]
                    if np.abs(composite[patch] - instant[patch]).mean() >= 30:
                        marker_errors.append((float(np.abs(delivered[patch] - composite[patch]).mean()),
                                              float(np.abs(delivered[patch] - instant[patch]).mean())))
                require(bool(marker_errors) and np.mean([value[0] for value in marker_errors])
                        < 0.5 * np.mean([value[1] for value in marker_errors]),
                        "SDK secondary-marker pixels do not distinguish composite from instantaneous buffer")
            metrics.append({"frame_index": frame["frame_index"], "side": side, "composite_roi_mae": error,
                            "instant_mid_roi_mae": instant_error, "source_mean_minus_instant_roi_mae": source_instant_difference,
                            "projected_smear_px": float(np.ptp(centres[:, 0])),
                            "sphere_centre_error_px": centre_error,
                            "sdk_minus_source_composite_centre_error_px": delivered_mean_geometry_error,
                            "sphere_orange_pixels": len(sphere_pixels),
                            "secondary_marker_mixed_cells": len(marker_errors),
                            "secondary_marker_composite_mae": float(np.mean([value[0] for value in marker_errors])) if marker_errors else None,
                            "secondary_marker_instant_mae": float(np.mean([value[1] for value in marker_errors])) if marker_errors else None})
        joined.append({"frame_index": frame["frame_index"], "sdk_timestamp_ns": stamp,
                       "source_frame": author["source_frame"], "visual_frame_id": index,
                       "sdk_minus_publication_reference_ns": stamp - publication_stamp,
                       "sdk_timestamp_reference": reference,
                       "sdk_to_exposure_midpoint_offset_s": float(config.get("sdk_to_exposure_midpoint_offset_s", 0)),
                       "earliest_available_authored_s": state["t_exposure_end_s"],
                       "availability_semantics": "derived causal lower bound at exposure end; not measured World/host availability",
                       "calibration_version": calibration_version, "sdk_calibration_version": sdk_version,
                       "sdk_nominal_calibration_sha256": nominal_calibration["sha256"] if nominal_calibration else None,
                       "acquisition": state,
                       "host_receive_monotonic_ns": frame["host_monotonic_ns"],
                       "host_saved_monotonic_ns": frame["host_saved_monotonic_ns"]})
    maximum_error = max(metric["composite_roi_mae"] for metric in metrics)
    if codec_baseline is not None:
        require(codec_baseline["verified"] and codec_baseline["realized_exposure_s"] == 0
                and codec_baseline["scenario"] == "static", "Codec baseline must be independently verified static zero exposure")
        require(codec_baseline["transport"] == meta["transport"], "Static codec baseline transport differs from evaluated capture")
        require(codec_baseline["source_calibration_version"] == calibration_version
                and codec_baseline["sdk_calibration_version"] == sdk_version,
                "Static codec baseline calibration differs from evaluated capture")
        if nominal_calibration is not None:
            require(codec_baseline["nominal_calibration_verification"] is not None
                    and codec_baseline["nominal_calibration_verification"]["sha256"] == nominal_calibration["sha256"],
                    "Static codec baseline nominal calibration seed differs from evaluated capture")
        threshold = max(3.0, codec_baseline["maximum_composite_roi_mae"] * 1.5 + 1)
        require(maximum_error <= threshold, "Delivered composite exceeds frozen static codec error bound")
    else:
        require(duration == 0 and config["scenario"] == "static", "Finite/moving acceptance requires a frozen static codec baseline")
        require(maximum_error <= 8, "Static delivered pixels exceed the fixed 8-level RGB codec quality bound")
        threshold = None
    if duration and config["scenario"] != "static":
        high_motion = [metric for metric in metrics if metric["projected_smear_px"] >= 3]
        require(len(high_motion) >= 4, "Insufficient delivered motion for exposure negative control")
        improvement = np.array([metric["instant_mid_roi_mae"] - metric["composite_roi_mae"] for metric in high_motion])
        require(np.mean([metric["source_mean_minus_instant_roi_mae"] for metric in high_motion]) >= 1,
                "Integrated source is not visibly distinct from instantaneous negative control")
        require(float(improvement.mean()) >= 0.25 and float((improvement > 0).mean()) >= 0.75,
                "SDK output does not consistently distinguish integrated pixels from instantaneous negative control")
    return {"verified": True, "scenario": config["scenario"], "transport": meta["transport"], "sdk_version": sdk["sdk_version"], "realized_exposure_s": float(duration),
            "scope": "Immutable encoded-RGBA subframe average transported by official SDK; no native sensor ISP claim",
            "sdk_timestamp_reference": reference,
            "sdk_to_exposure_midpoint_offset_s": float(config.get("sdk_to_exposure_midpoint_offset_s", 0)),
            "negative_control_contract": "For projected smear >=3px: source mean/instant ROI MAE >=1 on average; delivered mean error improves >=0.25 levels on average and >0 in >=75% of eye-frames",
            "maximum_composite_roi_mae": maximum_error, "static_codec_mae_threshold": threshold,
            "source_author_ids_not_received": sorted(set(row["source_frame"] for row in authored) - received),
            "prepared_window_count": len(prepared_plans),
            "prepared_source_ids_not_published": sorted(set(prepared_plans) - set(row["source_frame"] for row in authored)),
            "paired_invalid_observations": sum(not row.get("observation_valid", False) for row in observations),
            "secondary_pair_mismatches": sum("observed_left_subframe_id" in row and "observed_right_subframe_id" in row
                and row["observed_left_subframe_id"] != row["observed_right_subframe_id"] for row in observations),
            "secondary_observation_decode_failures": sum(bool(row.get("observation_error")) for row in observations),
            "source_subframes_verified": len(seen_subframes), "sdk_payload": payload,
            "source_calibration_version": calibration_version, "sdk_calibration_version": sdk_version,
            "nominal_calibration_verification": nominal_calibration,
            "producer_source_digests": source_digests,
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "maximum_subframe_sphere_centre_error_px": maximum_subframe_geometry_error,
            "image_metrics": metrics, "joined_frames": joined}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--codec-baseline", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--joined", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.codec_baseline.read_text()) if args.codec_baseline else None
    report = verify_composites(args.capture, args.source, args.frames, baseline)
    args.joined.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in report.pop("joined_frames")))
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({key: report[key] for key in ("verified", "scenario", "realized_exposure_s", "maximum_composite_roi_mae")}))


if __name__ == "__main__":
    main()
