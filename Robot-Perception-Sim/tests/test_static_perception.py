"""Analytic fixtures test algorithms only; these are not SDK collection evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from robot_perception_sim.ball_perception import process_frame, world_from_camera

spec = importlib.util.spec_from_file_location("static_perception", Path(__file__).parents[1]/"scripts/run_static_perception.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

CALIBRATION = {"left": {"fx": 336., "fy": 336., "cx": 319.5, "cy": 179.5},
               "right": {"fx": 336., "fy": 336., "cx": 319.5, "cy": 179.5}, "baseline_m": .12}


def sphere_image(center, eye_x=0):
    v, u = np.indices((360, 640))
    rays = np.stack(((u-319.5)/336, (v-179.5)/336, np.ones_like(u)), axis=-1)
    relative = np.asarray(center)-[eye_x, 0, 0]
    dot = rays @ relative
    a = (rays*rays).sum(axis=-1)
    discriminant = dot*dot-a*(relative@relative-.11**2)
    visible = (discriminant >= 0) & (dot > 0)
    depth = np.full((360, 640), np.nan, dtype=np.float32)
    depth[visible] = ((dot-np.sqrt(np.maximum(0, discriminant)))/a)[visible]
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    rgb[visible] = [255, 130, 0]
    return rgb, depth


@pytest.mark.parametrize("z", [.8, 1.2, 2, 3, 4])
@pytest.mark.parametrize("angle", [-20, 0, 20])
def test_five_depth_three_angle_geometry(z, angle):
    center = np.array([z*np.tan(np.radians(angle)), 0, z])
    left, depth = sphere_image(center)
    right, _ = sphere_image(center, .12)
    result = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    assert result["left"]["valid"]
    assert result["sdk_sphere"]["valid"], result["sdk_sphere"]
    assert np.linalg.norm(np.asarray(result["sdk_sphere"]["center_camera_m"])-center) < .002
    assert result["triangulation"]["valid"]
    # Raster centroid disparity becomes sensitive at the far end.
    assert np.linalg.norm(np.asarray(result["triangulation"]["center_camera_m"])-center) < .2


def test_off_axis_sphere_fit_is_not_surface_z_plus_radius():
    center = [1, .2, 2]
    left, depth = sphere_image(center)
    right, _ = sphere_image(center, .12)
    result = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    assert result["sdk_sphere"]["center_camera_m"] == pytest.approx(center, abs=.001)
    assert abs(float(np.nanmedian(depth))+.11-center[2]) > .01


def test_invalids_and_sdk_confidence_direction():
    left, depth = sphere_image([0, 0, 2])
    right, _ = sphere_image([0, 0, 2], .12)
    poor = process_frame(left, right, depth, np.full_like(depth, 99), CALIBRATION)
    assert poor["sdk_sphere"]["reason"] == "insufficient_foreground_depth"
    assert poor["sdk_sphere"]["center_camera_m"] is None
    missing = process_frame(left*0, right*0, depth, depth*0, CALIBRATION)
    assert not missing["left"]["valid"]
    assert not missing["triangulation"]["valid"]
    assert not missing["sdk_sphere"]["valid"]
    no_depth = process_frame(left, right, np.full_like(depth, np.inf), depth*0, CALIBRATION)
    assert not no_depth["sdk_sphere"]["valid"]
    assert no_depth["triangulation"]["valid"]


def test_clipped_and_wrong_epipolar_return_invalid():
    left, depth = sphere_image([-.95, 0, 1])
    right, _ = sphere_image([-.95, 0, 1], .12)
    result = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    assert result["left"]["reason"] == "clipped_contour"
    left, depth = sphere_image([0, 0, 2])
    right, _ = sphere_image([0, .1, 2], .12)
    result = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    assert result["triangulation"]["reason"] == "epipolar_mismatch"


def test_surface_outliers_are_robustly_rejected():
    left, depth = sphere_image([.6, 0, 2])
    right, _ = sphere_image([.6, 0, 2], .12)
    y, x = np.nonzero(np.isfinite(depth))
    depth[y[::7], x[::7]] = 8
    result = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    assert result["sdk_sphere"]["valid"]
    assert result["sdk_sphere"]["center_camera_m"] == pytest.approx([.6, 0, 2], abs=.003)


def test_oracle_world_transform_separate_and_rigid():
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    assert world_from_camera([0, 0, 2], transform) == [1, 2, 5]
    transform[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        world_from_camera([0, 0, 2], transform)


def test_cli_roundtrip_does_not_read_gt_until_evaluation(tmp_path):
    capture = tmp_path/"capture"
    capture.mkdir()
    left, depth = sphere_image([0, 0, 2])
    right, _ = sphere_image([0, 0, 2], .12)
    for side, rgb in (("left", left), ("right", right)):
        Image.fromarray(rgb).save(capture/f"{side}.png")
    depth.astype("<f4").tofile(capture/"depth.f32")
    np.zeros_like(depth).astype("<f4").tofile(capture/"confidence.f32")
    (capture/"metadata.json").write_text(json.dumps({"depth_source": "zed_stereo_sdk", "depth_units": "m", "baseline_m": .12, "calibration": {k: CALIBRATION[k] for k in ("left", "right")}}))
    (capture/"frames.jsonl").write_text(json.dumps({"frame_index": 0, "sdk_timestamp_ns": 1, "width": 640, "height": 360, "left_path": "left.png", "right_path": "right.png", "depth_path": "depth.f32", "confidence_path": "confidence.f32"})+"\n")
    first = cli.run(capture, tmp_path/"first")
    labels = tmp_path/"evaluation.jsonl"
    labels.write_text(json.dumps({"frame_index": 0, "depth_label_m": 2, "angle_label": "center", "ball_center_camera_m": [0, 0, 99]})+"\n")
    second = cli.run(capture, tmp_path/"second", labels)
    assert (tmp_path/"first/predictions.jsonl").read_bytes() == (tmp_path/"second/predictions.jsonl").read_bytes()
    assert not second["evaluation"]["five_depth_three_angle_coverage"]
    assert first["ground_truth_used_by_predictor"] is False
    assert (tmp_path/"second/metrics.csv").is_file()


def test_evaluator_reports_surface_and_centroid_errors_separately():
    center = [.6, 0, 2]
    left, depth = sphere_image(center)
    right, _ = sphere_image(center, .12)
    prediction = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    prediction["frame_index"] = 1
    label = {"frame_index": 1, "depth_label_m": 2, "angle_label": "right",
             "expected_visible": True, "ball_center_camera_m": center}
    # Inject measured SDK depth bias only into evaluation surface reader. It
    # must not alter the already-computed sphere-center prediction.
    result = cli.evaluate([prediction], [label], CALIBRATION,
        lambda _: (depth+.025, np.zeros_like(depth), prediction["left"]["mask"] > 0))
    group = result["groups"][0]
    assert group["sdk_surface_mae_m"] == pytest.approx(.025, abs=1e-6)
    assert group["sdk_surface_samples"] > 100
    assert group["centroid_mae_px"] < .2
    assert group["methods"]["sdk_sphere"]["center_mae_m"] < .001
    assert group["mean_mask_fraction"] > 0
    assert group["mean_depth_valid_fraction"] == 1


def test_evaluator_visibility_false_positives_invalid_reasons_and_unique_labels():
    left, depth = sphere_image([0, 0, 2])
    right, _ = sphere_image([0, 0, 2], .12)
    positive = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    positive["frame_index"] = 1
    missing = process_frame(left*0, right*0, depth, np.zeros_like(depth), CALIBRATION)
    missing["frame_index"] = 2
    labels = [{"frame_index": index, "angle_label": "occluded", "expected_visible": False,
               "ball_center_camera_m": [0, 0, 2]} for index in (1, 2)]
    result = cli.evaluate([positive, missing], labels, CALIBRATION)
    group = result["groups"][0]
    assert group["false_positives"] == 1
    assert group["false_positive_rate"] == .5
    assert group["methods"]["sdk_sphere"]["false_positives"] == 1
    assert group["methods"]["sdk_sphere"]["center_samples"] == 0
    assert group["detection_invalid_reasons"] == {"no_ball_contour": 1}
    assert group["methods"]["sdk_sphere"]["invalid_reasons"] == {"missing_valid_left_detection": 1}
    with pytest.raises(ValueError, match="labels must be unique"):
        cli.evaluate([positive], [labels[0], labels[0]])
    with pytest.raises(ValueError, match="indices must be unique"):
        cli.evaluate([positive, positive], [labels[0]])


def test_evaluator_requires_exact_five_by_three_positive_grid():
    left, depth = sphere_image([0, 0, 2])
    right, _ = sphere_image([0, 0, 2], .12)
    prediction = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    predictions, labels = [], []
    for z in (.8, 1.2, 2, 3, 4):
        for angle in ("left", "center", "right"):
            index = len(labels)
            predictions.append(prediction | {"frame_index": index})
            labels.append({"frame_index": index, "depth_label_m": z, "angle_label": angle,
                           "expected_visible": True, "ball_center_camera_m": [0, 0, z]})
    assert cli.evaluate(predictions, labels)["five_depth_three_angle_coverage"]
    assert not cli.evaluate(predictions, labels[:-1])["five_depth_three_angle_coverage"]
    labels[-1]["expected_visible"] = False
    assert not cli.evaluate(predictions, labels)["five_depth_three_angle_coverage"]
    labels[-1]["expected_visible"] = True
    predictions.append(prediction | {"frame_index": 99})
    labels.append(labels[-1] | {"frame_index": 99, "depth_label_m": 6})
    assert not cli.evaluate(predictions, labels)["five_depth_three_angle_coverage"]


@pytest.mark.parametrize("background_z", [1, 2, 4, 8])
def test_uniform_background_depth_is_not_accepted_as_sphere_center(background_z):
    left, depth = sphere_image([.6, 0, 2])
    right, _ = sphere_image([.6, 0, 2], .12)
    result = process_frame(left, right, np.full_like(depth, background_z), np.zeros_like(depth), CALIBRATION)
    assert not result["sdk_sphere"]["valid"]
    assert result["sdk_sphere"]["reason"] == "insufficient_surface_curvature"
    assert result["sdk_sphere"]["center_camera_m"] is None
    assert result["triangulation"]["valid"]


def test_wrong_depth_curved_patch_rejected_by_physical_silhouette_extent():
    left, depth = sphere_image([.6, 0, 2])
    right, _ = sphere_image([.6, 0, 2], .12)
    # A curved depth patch can pass the point-rank check but still represent a
    # different angular object size from the observed image contour.
    v, u = np.indices(depth.shape)
    depth = np.full_like(depth, 1) + .000001*((u-420)**2+(v-180)**2)
    result = process_frame(left, right, depth, np.zeros_like(depth), CALIBRATION)
    assert not result["sdk_sphere"]["valid"]
    assert result["sdk_sphere"]["reason"] == "sphere_silhouette_size_mismatch"
