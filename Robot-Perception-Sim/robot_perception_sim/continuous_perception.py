"""Offline continuous-observation evaluation. No GT enters image prediction.

Visibility and timing here are source/oracle diagnostics, never sensor decisions.
Missing deliveries remain absent predictions and are scored as observation loss.
"""
from collections import Counter, defaultdict

import numpy as np

from robot_perception_sim.ball_perception import world_from_camera

METHODS = ("triangulation", "sdk_sphere")


def reason_category(reason):
    if reason is None:
        return None
    if reason in ("missing_valid_stereo_detection", "epipolar_mismatch", "nonpositive_disparity"):
        return "stereo"
    if reason.startswith("sphere_") or reason == "insufficient_surface_curvature":
        return "sphere_fit"
    if "depth" in reason:
        return "depth"
    return "detection"


def project(point, intrinsics):
    x, y, z = point
    if z <= 0:
        return None
    return np.array([intrinsics["fx"]*x/z+intrinsics["cx"],
                     intrinsics["fy"]*y/z+intrinsics["cy"]])


def sphere_bbox(center, radius, intrinsics):
    x,y,z=center
    if z<=radius:return None
    bounds=[]
    for offset,focal,principal in ((x,intrinsics["fx"],intrinsics["cx"]),(y,intrinsics["fy"],intrinsics["cy"])):
        spread=radius*np.sqrt(offset*offset+z*z-radius*radius)
        bounds.append([principal+focal*(offset*z+sign*spread)/(z*z-radius*radius)for sign in(-1,1)])
    return [bounds[0][0],bounds[1][0],bounds[0][1],bounds[1][1]]


def association_boxes(label,intrinsics):
    """Offline-only finite-exposure annotation geometry, never detector input."""
    radius=label.get("ball_radius_m",.11)
    midpoint=sphere_bbox(label["ball_center_camera_m"],radius,intrinsics)
    samples=label.get("subframe_visibility",[])
    bounds=[]
    for sample in samples:
        center=sample.get("ball_center_camera_m",sample.get("visibility_evidence",{}).get("center_optical_m"))
        if center is not None and sample.get("expected_visible",False):
            box=sphere_bbox(center,radius,intrinsics)
            if box is not None:bounds.append(box)
    union=None
    if bounds:
        array=np.asarray(bounds)
        union=[float(array[:,0].min()),float(array[:,1].min()),float(array[:,2].max()),float(array[:,3].max())]
    if label.get("renderer_ball_bbox_xyxy")is not None:
        selected=label["renderer_ball_bbox_xyxy"];basis="renderer_ball_bbox"
    elif label["exposure_ms"]>0 and union is not None:
        selected=union;basis="analytic_visible_subframe_sphere_union"
    elif label["exposure_ms"]>0:
        selected=midpoint;basis="analytic_midpoint_fallback_missing_visible_subframe_gt"
    else:
        selected=midpoint;basis="analytic_midpoint_sphere_silhouette"
    return midpoint,union,selected,basis,len(bounds),len(samples)


def bbox_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def _key(row):
    return (row.get("sequence_id", row["trajectory_name"]), row["source_frame"])


def _visibility(label):
    if type(label.get("expected_visible")) is not bool:
        raise ValueError("Independent expected_visible boolean is required")
    if not label.get("visibility_basis"):
        raise ValueError("Independent visibility_basis is required")
    midpoint_visible=label["expected_visible"]
    any_visible=label.get("any_visible",midpoint_visible)
    all_support=label.get("all_visible",midpoint_visible)
    if type(any_visible)is not bool or type(all_support)is not bool or(all_support and not any_visible)or(midpoint_visible and not any_visible):
        raise ValueError("Invalid any_visible/all_visible exposure visibility")
    expected=any_visible if label["exposure_ms"]>0 else midpoint_visible
    top=label.get("visibility_evidence",{})
    evidence=[top]+[r.get("visibility_evidence",{})for r in label.get("subframe_visibility",[])]
    spatial_partial=bool(label.get("any_partial_occlusion",False))or any(
        r.get("fully_in_fov")is False or r.get("physical_mesh_blocked_ray_count",0)>0 or
        r.get("near_clipped_ray_count",0)>0 for r in evidence)
    source_reason=label.get("visibility_reason","visible"if midpoint_visible else"unknown")
    reason=source_reason
    # Preserve current explicit helper reasons. Only the legacy all-near-clipped
    # "occluded" label is normalized, with no actual physical mesh blockers.
    paths=top.get("blocked_mesh_paths_per_ray",[])
    all_near=bool(paths)and all(p and all(v=="target_behind_near_clip_plane"for v in p)for p in paths)
    no_physical=top.get("physical_mesh_blocked_ray_count",0)==0
    if source_reason=="occluded"and all_near and no_physical:
        reason="near_clip_or_camera_sphere_intersection"
    if expected and not midpoint_visible:
        if "near_clip"in reason:reason="partial_exposure_near_clip"
        elif "fov"in reason.lower():reason="partial_exposure_fov"
        elif "occlu"in reason:reason="partial_exposure_occlusion"
        else:reason="partial_exposure_visibility"
    return {"expected_visible":expected,"source_midpoint_expected_visible":midpoint_visible,
        "any_visible":any_visible,"all_visible":all_support and not spatial_partial,
        "all_subframes_have_visible_support":all_support,"spatial_partial_evidence":spatial_partial,
        "partial_visibility":any_visible and(not all_support or spatial_partial),
        "reason":reason,"source_reason":source_reason,"basis":label["visibility_basis"],
        "visibility_rule":"any_subframe_support"if label["exposure_ms"]>0 else"instantaneous_midpoint_support",
        "source":"offline_independent_visibility_diagnostic"}


def evaluate_frames(predictions, labels, calibration, period_s, run_id="run", source_sequence=None):
    """Join immutable image predictions with separately stored source labels.

    Complete source_sequence rows establish missing prefix, interior and suffix
    deliveries. Without it only interior source-ID/time gaps can be reported.
    """
    lookup = {r["frame_index"]: r for r in predictions}
    observed = {r["frame_index"]: r for r in labels}
    if len(lookup) != len(predictions) or len(observed) != len(labels):
        raise ValueError("Prediction and evaluation frame indices must be unique")
    if set(lookup) != set(observed):
        raise ValueError("Prediction/evaluation delivered frame coverage differs")
    if not np.isfinite(period_s) or period_s <= 0:
        raise ValueError("Frame period must be positive")
    observed_sources = {_key(r): r for r in labels if r.get("segment",r.get("phase","target")) == "target"}
    if len(observed_sources) != sum(r.get("segment",r.get("phase","target")) == "target" for r in labels):
        raise ValueError("Delivered target source identities must be unique")
    if source_sequence is not None:
        intended = {_key(r): r for r in source_sequence}
        if len(intended) != len(source_sequence):
            raise ValueError("Intended source identities must be unique")
        if not set(observed_sources).issubset(intended):
            raise ValueError("Delivered target source ID absent from intended sequence")
        schedule = []
        for key, original in intended.items():
            joined = observed_sources.get(key)
            if joined is not None:
                if abs(original["source_time_s"]-joined["source_time_s"]) > 1e-6:
                    raise ValueError("Delivered source time differs from intended source time")
                schedule.append(joined)
            else:
                schedule.append(original)
    else:
        schedule = list(observed_sources.values())
    records = []
    for label in schedule:
        source = observed_sources.get(_key(label))
        prediction = lookup[source["frame_index"]] if source is not None else None
        visibility = _visibility(label)
        time_s = float(label["source_time_s"])
        if not np.isfinite(time_s):
            raise ValueError("Source midpoint time must be finite")
        timing_valid = bool(label.get("timing_valid", True)) and prediction is not None
        row = {"run_id": run_id, "sequence_id": label.get("sequence_id", label["trajectory_name"]),
               "trajectory_name": label["trajectory_name"],"head_pose":label.get("head_pose","unspecified"), "exposure_ms": label["exposure_ms"],
               "source_frame": label["source_frame"], "source_time_s": time_s,
               "trajectory_time_s": label.get("trajectory_time_s", time_s), "period_s": period_s,
               "frame_index": source["frame_index"] if source else None,
               "observation_missing": prediction is None, "timing_valid": timing_valid,
               "visibility_diagnostic": visibility, "pose_source": "oracle",
               "source_sequence_complete": source_sequence is not None,
               "detection": {"valid": bool(prediction and prediction["left"]["valid"]),
                             "reason": prediction["left"]["reason"] if prediction else "observation_missing"},
               "methods": {}}
        for key in ("exposure_start_s", "exposure_end_s", "subframe_times_s", "subframe_visibility", "visibility_evidence", "observation_interval_end_s", "time_start_s", "time_end_s",
                    "desired_physical_midpoint_s", "trajectory_requested_time_s", "trajectory_realized_time_s",
                    "subframe_realized_times_s", "subframe_quantization_errors_s", "maximum_subframe_time_error_s",
                    "observation_interval_basis"):
            if key in label:
                row[key] = label[key]
        candidate_valid=row["detection"]["valid"]
        midpoint_bbox,union_bbox,target_bbox,bbox_basis,union_used,union_total=association_boxes(label,calibration["left"])
        overlap=midpoint_overlap=union_overlap=None
        if candidate_valid:
            x,y,w,h=prediction["left"]["bbox_xywh"]
            candidate_bbox=[x,y,x+w,y+h]
            if target_bbox is not None:overlap=bbox_iou(candidate_bbox,target_bbox)
            if midpoint_bbox is not None:midpoint_overlap=bbox_iou(candidate_bbox,midpoint_bbox)
            if union_bbox is not None:union_overlap=bbox_iou(candidate_bbox,union_bbox)
        matched=bool(candidate_valid and visibility["expected_visible"]and overlap is not None and overlap>=.5)
        row["detection"].update({"candidate_valid":candidate_valid,"ball_matched":matched,
            "match_iou":overlap,"midpoint_iou":midpoint_overlap,"union_iou":union_overlap,
            "match_iou_threshold":.5,"match_bbox_basis":bbox_basis,
            "midpoint_bbox_xyxy":midpoint_bbox,"union_bbox_xyxy":union_bbox,
            "union_subframes_used":union_used,"union_subframes_total":union_total,
            "match_bbox_xyxy":target_bbox,
            "evaluation_failure_reason":"wrong_object_detection"if candidate_valid and not matched else
                row["detection"]["reason"]if not matched else None,
            "evaluation_failure_category":"detection"if candidate_valid and not matched else None,
            "match_is_evaluation_only":True})
        if not prediction:
            row["detection"]["category"] = "timing"
        else:
            row["detection"]["category"] = reason_category(prediction["left"]["reason"])
        truth = np.asarray(label["ball_center_camera_m"], dtype=float)
        pose = label["T_world_camera"]
        world_truth = np.asarray(label["ball_world_position_m"], dtype=float)
        transformed_truth = world_from_camera(truth, pose)
        if not np.allclose(transformed_truth, world_truth, atol=1e-5, rtol=0):
            raise ValueError("Oracle camera/world center labels disagree with actual pose")
        row["T_world_camera"] = pose
        k = calibration["left"]
        truth_pixel = project(truth, k)
        if prediction and prediction["left"]["valid"] and truth_pixel is not None and timing_valid:
            radius = label.get("ball_radius_m", .11)
            x, y, z = truth
            if z > radius:
                silhouette = np.array([k["cx"]+k["fx"]*x*z/(z*z-radius*radius),
                                       k["cy"]+k["fy"]*y*z/(z*z-radius*radius)])
                row["detection"]["gt_silhouette_centroid_error_px"] = float(np.linalg.norm(np.asarray(prediction["left"]["centroid_px"])-silhouette))
        for method in METHODS:
            estimate = prediction[method] if prediction else {"valid": False, "reason": "observation_missing"}
            result = {"valid": estimate["valid"], "ball_matched_valid":bool(estimate["valid"]and matched), "reason": estimate["reason"],
                      "category": reason_category(estimate["reason"]) if prediction else "timing"}
            if estimate["valid"] and timing_valid:
                point = np.asarray(estimate["center_camera_m"])
                world = world_from_camera(point, pose)
                result["oracle_center_world_m"] = world
                if visibility["expected_visible"]:
                    result["center_camera_error_m"] = float(np.linalg.norm(point-truth))
                    result["center_world_error_m"] = float(np.linalg.norm(np.asarray(world)-world_truth))
                pixel = project(point, k)
                if pixel is not None:
                    if truth_pixel is not None and visibility["expected_visible"]:
                        result["gt_center_projection_error_px"] = float(np.linalg.norm(pixel-truth_pixel))
                    if prediction["left"]["valid"]:
                        result["center_projection_to_detection_residual_px"] = float(np.linalg.norm(pixel-np.asarray(prediction["left"]["centroid_px"])))
                        result["projection_to_detection_is_independent_accuracy_check"] = False
            row["methods"][method] = result
        records.append(row)
    return records


def error_statistics(values):
    return {"samples": len(values), "mae": float(np.mean(np.abs(values))) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": float(max(values)) if values else None}


def loss_intervals(records, valid_field):
    """Measured invalid streaks and unobserved temporal gaps are separate.

    Known intended-but-undelivered rows retain their real scheduled midpoints.
    With no full source sequence, an unexpectedly large time/ID gap splits a
    streak; its unknown interval is reported instead of invented observations.
    """
    sequences = defaultdict(list)
    for r in records:
        sequences[(r["run_id"], r["sequence_id"])].append(r)
    intervals, gaps = [], []
    for sequence, rows in sequences.items():
        rows = sorted(rows, key=lambda r:r["source_time_s"])
        if any(b["source_time_s"] <= a["source_time_s"] for a,b in zip(rows, rows[1:])):
            raise ValueError("Source midpoint times must strictly increase within sequence")
        streak = None
        for i, row in enumerate(rows):
            time_s, period = row["source_time_s"], row["period_s"]
            next_row = rows[i+1] if i+1 < len(rows) else None
            next_time = next_row["source_time_s"] if next_row else time_s+period
            gap = next_row is not None and (next_time-time_s > 1.5*period or
                (not row["source_sequence_complete"] and next_row["source_frame"] > row["source_frame"]+1))
            stop = min(next_time, time_s+period) if gap else next_time
            declared_end=row.get("observation_interval_end_s")
            if declared_end is not None:
                if not np.isfinite(declared_end)or declared_end<time_s:
                    raise ValueError("Observation interval end precedes its midpoint")
                stop=min(stop,declared_end)
            if gap:
                gaps.append({"run_id": sequence[0], "sequence_id": sequence[1],
                             "start_s": stop, "end_s": next_time,
                             "duration_s": max(0, next_time-stop),
                             "unobserved_source_ids": max(0, next_row["source_frame"]-row["source_frame"]-1)})
            visible = row["visibility_diagnostic"]["expected_visible"]
            valid = row["detection"]["ball_matched"] if valid_field == "detection" else row["methods"][valid_field]["ball_matched_valid"]
            loss = visible and (not valid or not row["timing_valid"])
            if loss:
                kind = "missing_delivery" if row["observation_missing"] else "timing_invalid" if not row["timing_valid"] else "wrong_object_detection" if row["detection"]["candidate_valid"]and not row["detection"]["ball_matched"]else"algorithm_invalid"
                if streak is None or streak["kind"] != kind:
                    if streak is not None:intervals.append(streak)
                    streak = {"run_id": sequence[0], "sequence_id": sequence[1], "kind": kind,
                              "start_s": time_s, "end_s": stop, "last_midpoint_s":time_s,"frames": 1,
                              "terminal_support_assumed":declared_end is None and next_row is None}
                else:
                    streak["end_s"] = stop
                    streak["last_midpoint_s"] = time_s
                    streak["terminal_support_assumed"] = declared_end is None and next_row is None
                    streak["frames"] += 1
            elif streak is not None:
                intervals.append(streak)
                streak = None
            if gap and streak is not None:
                intervals.append(streak)
                streak = None
        if streak is not None:intervals.append(streak)
    for r in intervals:
        r["duration_s"] = r["end_s"]-r["start_s"]
        r["observed_midpoint_span_s"] = r["last_midpoint_s"]-r["start_s"]
    return {"intervals": intervals, "maximum_duration_s": max((r["duration_s"]for r in intervals),default=0.),
            "total_duration_s": sum(r["duration_s"]for r in intervals), "unobserved_gaps": gaps,
            "unobserved_gap_duration_s": sum(r["duration_s"]for r in gaps)}


def summarize(records):
    grouped = defaultdict(list)
    for row in records:grouped[(row["trajectory_name"],row["head_pose"],row["exposure_ms"])].append(row)
    groups = []
    for (trajectory, head_pose, exposure), rows in grouped.items():
        visible = [r for r in rows if r["visibility_diagnostic"]["expected_visible"]]
        eligible = [r for r in visible if r["timing_valid"]]
        invisible = [r for r in rows if not r["visibility_diagnostic"]["expected_visible"]]
        detected = sum(r["detection"]["ball_matched"]for r in eligible)
        group = {"trajectory_name":trajectory,"head_pose":head_pose,"exposure_ms":exposure,"intended_frames":len(rows),
            "delivered_frames":sum(not r["observation_missing"]for r in rows),
            "missing_delivery_frames":sum(r["observation_missing"]for r in rows),
            "visible_frames":len(visible),"timing_eligible_visible_frames":len(eligible),
            "timing_invalid_frames":sum(not r["timing_valid"]for r in rows),
            "detection_valid_visible_frames":detected,
            "raw_candidate_valid_visible_frames":sum(r["detection"]["candidate_valid"]for r in eligible),
            "raw_candidate_valid_visible_rate":sum(r["detection"]["candidate_valid"]for r in eligible)/len(eligible)if eligible else None,
            "wrong_object_false_positives":sum(r["detection"]["candidate_valid"]and not r["detection"]["ball_matched"]for r in rows if r["timing_valid"]),
            "ball_missed_visible_frames":len(eligible)-detected,
            "detection_evaluation_failure_reasons":dict(Counter(r["detection"]["evaluation_failure_reason"]for r in rows if r["detection"]["evaluation_failure_reason"])),
            "visible_detection_rate":detected/len(eligible) if eligible else None,
            "all_intended_visible_detection_rate":sum(r["detection"]["ball_matched"] and r["timing_valid"]for r in visible)/len(visible) if visible else None,
            "fully_visible_frames":sum(r["visibility_diagnostic"]["all_visible"]for r in rows),
            "partly_visible_frames":sum(r["visibility_diagnostic"]["partial_visibility"]for r in rows),
            "invisible_false_detections":sum(r["detection"]["valid"]for r in invisible),
            "visibility_diagnostic_counts":dict(Counter(r["visibility_diagnostic"]["reason"]for r in rows)),
            "visibility_reason_categories":dict(Counter(
                "near_clip"if "near_clip"in r["visibility_diagnostic"]["reason"].lower()else
                "FOV" if "fov" in r["visibility_diagnostic"]["reason"].lower() else
                "occlusion" if "occlu" in r["visibility_diagnostic"]["reason"].lower() else
                "visible_or_other" for r in rows)),
            "detection_invalid_reasons":dict(Counter(r["detection"]["reason"]for r in rows if not r["detection"]["valid"])),
            "pose_source":"oracle","methods":{},"losses":{"detection":loss_intervals(rows,"detection")}}
        for method in METHODS:
            results = [r["methods"][method]for r in eligible]
            valid = sum(r["valid"]for r in results)
            metrics = {"valid_visible_frames":valid,"visible_3d_valid_rate":valid/len(eligible)if eligible else None,
                "ball_matched_valid_visible_frames":sum(r["ball_matched_valid"]for r in results),
                "ball_matched_visible_3d_valid_rate":sum(r["ball_matched_valid"]for r in results)/len(eligible)if eligible else None,
                "all_intended_ball_matched_visible_3d_valid_rate":sum(r["methods"][method]["ball_matched_valid"]and r["timing_valid"]for r in visible)/len(visible)if visible else None,
                "all_intended_visible_3d_valid_rate":sum(r["methods"][method]["valid"]and r["timing_valid"]for r in visible)/len(visible)if visible else None,
                "invalid_reasons":dict(Counter(r["methods"][method]["reason"]for r in rows if not r["methods"][method]["valid"])),
                "algorithm_reason_categories":dict(Counter(r["methods"][method]["category"]for r in rows if r["methods"][method]["category"]and not r["observation_missing"])),
                "invisible_valid_estimates":sum(r["methods"][method]["valid"]for r in invisible)}
            for field in ("center_camera_error_m","center_world_error_m","gt_center_projection_error_px","center_projection_to_detection_residual_px"):
                metrics[field] = error_statistics([r[field]for r in results if field in r])
                metrics["ball_matched_"+field] = error_statistics([r[field]for r in results if field in r and r["ball_matched_valid"]])
            group["methods"][method] = metrics
            group["losses"][method] = loss_intervals(rows,method)
        groups.append(group)
    comparisons = []
    for trajectory,head_pose in sorted({(r["trajectory_name"],r["head_pose"])for r in records}):
        zero = [r for r in records if r["trajectory_name"]==trajectory and r["head_pose"]==head_pose and r["exposure_ms"]==0]
        finite = [r for r in records if r["trajectory_name"]==trajectory and r["head_pose"]==head_pose and r["exposure_ms"]==10]
        if not zero or not finite:continue
        a, b = defaultdict(list), defaultdict(list)
        for row in zero:a[round(row["trajectory_time_s"],9)].append(row)
        for row in finite:b[round(row["trajectory_time_s"],9)].append(row)
        common = sorted(a.keys()&b.keys())
        pairs = [(x,y) for time in common for x,y in zip(a[time],b[time])]
        eligible = [(x,y)for x,y in pairs if x["timing_valid"]and y["timing_valid"]and
            x["visibility_diagnostic"]["expected_visible"]and y["visibility_diagnostic"]["expected_visible"]]
        comparison = {"trajectory_name":trajectory,"head_pose":head_pose,"baseline_exposure_ms":0,"finite_exposure_ms":10,
            "paired_midpoints":len(pairs),"baseline_only_midpoints":len(zero)-len(pairs),
            "finite_only_midpoints":len(finite)-len(pairs),
            "physical_midpoints_match":a.keys()==b.keys()and all(len(a[t])==len(b[t])for t in common),
            "jointly_visible_timing_eligible_pairs":len(eligible),
            "paired_visible_detection_changes_10ms_minus_0ms":sum(int(y["detection"]["ball_matched"])-int(x["detection"]["ball_matched"])for x,y in eligible),
            "methods":{},
            "note":"Midpoint pairing is source/oracle evidence, not a sensor pose or timing estimate"}
        for method in METHODS:
            comparison["methods"][method] = {}
            for name, position in (("baseline_0ms",0),("finite_10ms",1)):
                measurements = [pair[position]["methods"][method]for pair in eligible]
                comparison["methods"][method][name] = {
                    "valid":sum(r["valid"]for r in measurements),
                    "valid_rate":sum(r["valid"]for r in measurements)/len(eligible)if eligible else None,
                    "ball_matched_valid":sum(r["ball_matched_valid"]for r in measurements),
                    "ball_matched_valid_rate":sum(r["ball_matched_valid"]for r in measurements)/len(eligible)if eligible else None,
                    "center_world_error_m":error_statistics([r["center_world_error_m"]for r in measurements if "center_world_error_m"in r])}
        comparisons.append(comparison)
    return {"pose_source":"oracle","ground_truth_used_by_predictor":False,
            "visibility_source":"independent offline diagnostic; not sensor-observed FOV/occlusion",
            "ball_match_iou_threshold":.5,"groups":groups,"exposure_comparisons":comparisons}
