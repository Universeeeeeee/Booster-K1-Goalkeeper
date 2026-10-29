"""Image-only static orange-ball baseline; optical coordinates are right/down/forward.

Ground truth is deliberately absent from these APIs. SDK confidence is 0..100,
lower is better. Radius is a physical object prior, never an observed position.
"""

import cv2
import numpy as np


def invalid(reason, **details):
    return {"valid": False, "reason": reason, "center_camera_m": None, **details}


def detect_ball(rgb):
    """Return a filled foreground contour mask, bbox and image centroid."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    raw = cv2.inRange(hsv, (5, 100, 60), (35, 255, 255))
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area >= 6 and perimeter > 0:
            circularity = 4 * np.pi * area / perimeter**2
            if circularity >= .55:
                candidates.append((area, circularity, contour))
    mask = np.zeros(raw.shape, dtype=np.uint8)
    if not candidates:
        return {"valid": False, "reason": "no_ball_contour", "mask": mask,
                "bbox_xywh": None, "centroid_px": None, "confidence": 0.0, "mask_fraction": 0.0, "mask_pixels": 0}
    _, circularity, contour = max(candidates, key=lambda item: item[0])
    x, y, w, h = cv2.boundingRect(contour)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    moments = cv2.moments(contour)
    centroid = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
    clipped = x == 0 or y == 0 or x + w == rgb.shape[1] or y + h == rgb.shape[0]
    ambiguous = len(candidates) > 1 and sorted((c[0] for c in candidates), reverse=True)[1] > .5 * max(c[0] for c in candidates)
    return {"valid": not (clipped or ambiguous),
            "reason": "clipped_contour" if clipped else "ambiguous_contours" if ambiguous else None,
            "mask": mask, "bbox_xywh": [x, y, w, h], "centroid_px": centroid,
            "confidence": float(min(1, circularity)),
            "mask_pixels": int(np.count_nonzero(mask)),
            "mask_fraction": float(np.count_nonzero(mask)/mask.size)}


def triangulate(left, right, calibration, disparity_sigma_px=.25):
    if not left["valid"] or not right["valid"]:
        return invalid("missing_valid_stereo_detection")
    kl, kr = calibration["left"], calibration["right"]
    ul, vl = left["centroid_px"]
    ur, vr = right["centroid_px"]
    if abs((vl-kl["cy"])/kl["fy"] - (vr-kr["cy"])/kr["fy"]) * kl["fy"] > 2:
        return invalid("epipolar_mismatch")
    disparity = (ul-kl["cx"])/kl["fx"] - (ur-kr["cx"])/kr["fx"]
    if not np.isfinite(disparity) or disparity <= 0:
        return invalid("nonpositive_disparity")
    z = calibration["baseline_m"] / disparity
    center = [z*(ul-kl["cx"])/kl["fx"], z*(vl-kl["cy"])/kl["fy"], z]
    return {"valid": True, "reason": None, "center_camera_m": center,
            "disparity_px": float(disparity*kl["fx"]),
            "sigma_z_m": float(z*z/(kl["fx"]*calibration["baseline_m"])*disparity_sigma_px)}


def fit_sdk_sphere(depth, confidence, detection, intrinsics, radius_m=.11,
                   confidence_max=95):
    """Robust known-radius sphere fit on eroded, confidence-filtered foreground.

    Fit metric surface points, not bbox depth. Reject inadequate support, rank,
    residual and visible-surface geometry rather than filling from another source.
    """
    if not detection["valid"]:
        return invalid("missing_valid_left_detection", valid_depth_fraction=0.0)
    foreground = cv2.erode(detection["mask"], np.ones((3, 3), np.uint8)) > 0
    usable = foreground & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence >= 0) & (confidence <= confidence_max)
    fraction = float(usable.sum()/max(1, foreground.sum()))
    if usable.sum() < 12 or fraction < .15:
        return invalid("insufficient_foreground_depth", valid_depth_fraction=fraction)
    v, u = np.nonzero(usable)
    z = depth[usable].astype(float)
    # A robust cluster removes background leakage before geometric fitting.
    median = np.median(z)
    keep = np.abs(z-median) <= max(2*radius_m, 4*np.median(np.abs(z-median)))
    u, v, z = u[keep], v[keep], z[keep]
    points = np.column_stack(((u-intrinsics["cx"])/intrinsics["fx"]*z,
                              (v-intrinsics["cy"])/intrinsics["fy"]*z, z))
    if len(points) < 12:
        return invalid("insufficient_depth_cluster", valid_depth_fraction=fraction)
    points = points[::max(1, len(points)//4000)]
    # An exactly planar SDK patch does not constrain sphere curvature. The
    # relative threshold detects numerical rank loss, not far-ball curvature.
    singular = np.linalg.svd(points-np.mean(points, axis=0), compute_uv=False)
    if singular[0] == 0 or singular[-1]/singular[0] < 1e-4:
        return invalid("insufficient_surface_curvature", valid_depth_fraction=fraction)
    front = np.median(points, axis=0)
    center = front + radius_m*front/np.linalg.norm(front)
    for _ in range(40):
        delta = center-points
        distances = np.linalg.norm(delta, axis=1)
        residual = distances-radius_m
        scale = max(.001, 1.4826*np.median(np.abs(residual-np.median(residual))))
        weights = np.minimum(1, 1.5*scale/np.maximum(np.abs(residual), 1e-12))
        jacobian = delta/np.maximum(distances[:, None], 1e-12)
        step, _, rank, _ = np.linalg.lstsq(jacobian*weights[:, None]**.5,
                                          -residual*weights**.5, rcond=None)
        center += step
        if np.linalg.norm(step) < 1e-7:
            break
    residual = np.abs(np.linalg.norm(points-center, axis=1)-radius_m)
    inliers = residual <= max(.005, radius_m*.15)
    if rank < 3 or np.linalg.cond(jacobian) > 100 or inliers.mean() < .65:
        return invalid("sphere_fit_inconsistent", valid_depth_fraction=fraction)
    if center[2] <= 0 or np.median(np.sum((points-center)*points, axis=1)) >= 0:
        return invalid("sphere_fit_not_visible_surface", valid_depth_fraction=fraction)
    if center[2] <= radius_m:
        return invalid("sphere_fit_invalid_projection", valid_depth_fraction=fraction)
    # The known physical radius must agree with the detected silhouette size.
    # A small orange patch on a nearer background plane can otherwise fit a
    # local sphere despite having the wrong angular extent.
    predicted_extent = np.array([
        2*intrinsics[focal]*radius_m*np.sqrt(center[axis]**2+center[2]**2-radius_m**2)
        /(center[2]**2-radius_m**2) for axis, focal in ((0, "fx"), (1, "fy"))])
    observed_extent = np.asarray(detection["bbox_xywh"][2:], dtype=float)
    extent_ratio = predicted_extent/observed_extent
    if np.any(extent_ratio < .65) or np.any(extent_ratio > 1.5):
        return invalid("sphere_silhouette_size_mismatch", valid_depth_fraction=fraction)
    return {"valid": True, "reason": None, "center_camera_m": center.tolist(),
            "valid_depth_fraction": fraction, "surface_points": len(points),
            "sphere_inlier_fraction": float(inliers.mean()),
            "median_surface_residual_m": float(np.median(residual))}


def process_frame(left_rgb, right_rgb, depth, confidence, calibration, radius_m=.11):
    if radius_m <= 0 or not np.isfinite(radius_m):
        raise ValueError("radius_m must be finite and positive")
    if depth.shape != left_rgb.shape[:2] or confidence.shape != depth.shape:
        raise ValueError("SDK depth/confidence and left image sizes must agree")
    left, right = detect_ball(left_rgb), detect_ball(right_rgb)
    return {"left": left, "right": right,
            "triangulation": triangulate(left, right, calibration),
            "sdk_sphere": fit_sdk_sphere(depth, confidence, left, calibration["left"], radius_m)}


def world_from_camera(center_camera_m, transform):
    """Separate oracle-pose experiment; never used to detect or fill invalids."""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_world_camera must be finite 4x4")
    if not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(transform[:3, :3]), 1):
        raise ValueError("T_world_camera must be a rigid transform")
    return (transform @ np.r_[center_camera_m, 1])[:3].tolist()
