"""Continuous evaluation fixtures are algorithm tests, never SDK evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from robot_perception_sim.continuous_perception import evaluate_frames, summarize

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("continuous_cli",ROOT/"scripts/run_continuous_perception.py")
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
K={"fx":100.,"fy":100.,"cx":60.,"cy":40.}
CALIBRATION={"left":K,"right":K,"baseline_m":.12}


def label(index,visible=True,exposure=0):
    return {"frame_index":index,"source_frame":index,"source_time_s":index*.1,
            "trajectory_time_s":index*.1,"trajectory_name":"direct_goal","sequence_id":"shot",
            "exposure_ms":exposure,"ball_center_camera_m":[0,0,2],
            "ball_world_position_m":[1,2,5],"T_world_camera":[[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]],
            "expected_visible":visible,"visibility_basis":"independent analytic fixture with no occluder",
            "visibility_reason":"visible"if visible else "out_of_fov","any_visible":visible,"all_visible":visible,
            "timing_valid":True}


def prediction(index,detected=True,sdk=True):
    def result(valid,reason,point):
        return {"valid":valid,"reason":None if valid else reason,"center_camera_m":point if valid else None}
    return {"frame_index":index,"left":{"valid":detected,"reason":None if detected else"no_ball_contour","centroid_px":[60,40]if detected else None,"bbox_xywh":[54.5,34.5,11,11]if detected else None},
            "triangulation":result(detected,"missing_valid_stereo_detection",[0,0,2]),
            "sdk_sphere":result(sdk,"sphere_fit_inconsistent",[.02,0,2])}


def test_oracle_pose_errors_and_projection_residual_are_separate():
    rows=evaluate_frames([prediction(0)],[label(0)],CALIBRATION,.1)
    sdk=rows[0]["methods"]["sdk_sphere"]
    assert sdk["center_world_error_m"]==pytest.approx(.02)
    assert sdk["center_camera_error_m"]==pytest.approx(.02)
    assert sdk["oracle_center_world_m"]==pytest.approx([1.02,2,5])
    assert sdk["gt_center_projection_error_px"]==pytest.approx(1)
    assert sdk["center_projection_to_detection_residual_px"]==pytest.approx(1)
    assert sdk["projection_to_detection_is_independent_accuracy_check"]is False
    # The triangulation projection is self-consistent by construction and is
    # explicitly not presented as independent evidence of position accuracy.
    assert rows[0]["methods"]["triangulation"]["center_projection_to_detection_residual_px"]==0
    assert rows[0]["pose_source"]=="oracle"


def test_missing_prefix_and_suffix_are_losses_not_synthetic_predictions():
    sources=[label(i)for i in range(5)]
    rows=evaluate_frames([prediction(1),prediction(2)],[label(1),label(2)],CALIBRATION,.1,
                         source_sequence=sources)
    assert [r["source_frame"]for r in rows if r["observation_missing"]]==[0,3,4]
    assert all(r["frame_index"]is None for r in rows if r["observation_missing"])
    assert all("oracle_center_world_m"not in r["methods"]["sdk_sphere"]for r in rows if r["observation_missing"])
    group=summarize(rows)["groups"][0]
    assert group["missing_delivery_frames"]==3
    assert group["methods"]["sdk_sphere"]["visible_3d_valid_rate"]==1
    assert group["methods"]["sdk_sphere"]["all_intended_visible_3d_valid_rate"]==.4
    losses=group["losses"]["sdk_sphere"]
    assert losses["maximum_duration_s"]==pytest.approx(.2)
    assert losses["total_duration_s"]==pytest.approx(.3)
    assert {r["kind"]for r in losses["intervals"]}=={"missing_delivery"}


def test_missing_observation_gap_does_not_inflate_algorithm_streak():
    predictions=[prediction(0,sdk=False),prediction(1,sdk=False),prediction(5,sdk=False)]
    rows=evaluate_frames(predictions,[label(0),label(1),label(5)],CALIBRATION,.1)
    losses=summarize(rows)["groups"][0]["losses"]["sdk_sphere"]
    assert losses["maximum_duration_s"]==pytest.approx(.2)
    assert losses["total_duration_s"]==pytest.approx(.3)
    assert losses["unobserved_gap_duration_s"]==pytest.approx(.3)
    assert losses["unobserved_gaps"][0]["unobserved_source_ids"]==3


def test_timing_invalid_and_visibility_diagnostics_do_not_rewrite_algorithm_reason():
    labels=[label(0,False),label(1),label(2)]
    labels[0]["visibility_reason"]="occluded"
    labels[1]["timing_valid"]=False
    labels[2].update(any_visible=True,all_visible=False,visibility_reason="partial_occlusion")
    predictions=[prediction(0,False,False),prediction(1),prediction(2)]
    rows=evaluate_frames(predictions,labels,CALIBRATION,.1)
    assert rows[0]["detection"]["reason"]=="no_ball_contour"
    assert rows[0]["detection"]["category"]=="detection"
    assert rows[0]["visibility_diagnostic"]["reason"]=="occluded"
    assert "center_world_error_m"not in rows[1]["methods"]["sdk_sphere"]
    group=summarize(rows)["groups"][0]
    assert group["partly_visible_frames"]==1
    assert group["visibility_reason_categories"]["occlusion"]==2
    assert group["timing_invalid_frames"]==1
    assert group["losses"]["sdk_sphere"]["intervals"][0]["kind"]=="timing_invalid"


def test_exposure_comparison_requires_same_physical_midpoints():
    zero=evaluate_frames([prediction(i)for i in range(3)],[label(i)for i in range(3)],CALIBRATION,.1,"zero")
    finite_labels=[label(i,exposure=10)for i in range(3)]
    finite=evaluate_frames([prediction(i,sdk=i!=1)for i in range(3)],finite_labels,CALIBRATION,.1,"finite")
    comparison=summarize(zero+finite)["exposure_comparisons"][0]
    assert comparison["physical_midpoints_match"]
    assert comparison["paired_midpoints"]==3
    assert comparison["methods"]["sdk_sphere"]["finite_10ms"]["valid_rate"]==pytest.approx(2/3)
    shifted=copy.deepcopy(finite)
    for row in shifted:row["trajectory_time_s"]+=.005
    comparison=summarize(zero+shifted)["exposure_comparisons"][0]
    assert not comparison["physical_midpoints_match"]
    assert comparison["paired_midpoints"]==0


@pytest.mark.parametrize("change",["visibility","pose","duplicate","time"])
def test_invalid_evaluation_contract_rejected(change):
    labels=[label(0)]
    sources=None
    if change=="visibility":labels[0].pop("visibility_basis")
    elif change=="pose":labels[0]["ball_world_position_m"]=[99,0,0]
    elif change=="duplicate":labels.append(label(0))
    else:
        sources=[label(0)];sources[0]["source_time_s"]+=.005
    with pytest.raises(ValueError):evaluate_frames([prediction(0)],labels,CALIBRATION,.1,source_sequence=sources)


def test_cli_prediction_files_unchanged_when_only_gt_changes(tmp_path):
    capture=tmp_path/"capture";capture.mkdir()
    # This is deliberately a negative image. GT cannot create detections.
    for eye in ("left","right"):Image.fromarray(np.zeros((80,120,3),dtype=np.uint8)).save(capture/f"{eye}.png")
    for kind,value in (("depth",2),("confidence",0)):np.full((80,120),value,dtype="<f4").tofile(capture/f"{kind}.f32")
    metadata={"depth_source":"zed_stereo_sdk","depth_units":"m","baseline_m":.12,
              "calibration":{"left":K,"right":K},"reported_camera_fps":10,"depth_mode":"synthetic_fixture"}
    (capture/"metadata.json").write_text(json.dumps(metadata))
    frame={"frame_index":0,"sdk_timestamp_ns":1,"width":120,"height":80,
           **{f"{kind}_path":f"{kind}."+("png"if kind in ("left","right")else"f32")for kind in ("left","right","depth","confidence")}}
    (capture/"frames.jsonl").write_text(json.dumps(frame)+"\n")
    first=tmp_path/"first.jsonl";second=tmp_path/"second.jsonl"
    a=label(0);b=label(0);b["ball_center_camera_m"]=[0,0,5];b["ball_world_position_m"]=[1,2,8]
    first.write_text(json.dumps(a)+"\n");second.write_text(json.dumps(b)+"\n")
    cli.run([capture],[first],tmp_path/"a")
    cli.run([capture],[second],tmp_path/"b")
    assert (tmp_path/"a/run_000/predictions.jsonl").read_bytes()==(tmp_path/"b/run_000/predictions.jsonl").read_bytes()
    report=json.loads((tmp_path/"b/summary.json").read_text())
    assert report["groups"][0]["visible_detection_rate"]==0
    assert report["pose_source"]=="oracle"
    assert (tmp_path/"b/metrics.csv").is_file()


def test_wrong_white_circle_is_false_positive_miss_and_continuous_ball_loss():
    wrong=prediction(0)
    wrong["left"].update(centroid_px=[100,40],bbox_xywh=[94.5,34.5,11,11])
    for method in("triangulation","sdk_sphere"):wrong[method]["center_camera_m"]=[.8,0,2]
    rows=evaluate_frames([wrong],[label(0)],CALIBRATION,.1)
    detection=rows[0]["detection"]
    assert detection["candidate_valid"]
    assert not detection["ball_matched"]
    assert detection["match_iou"]==0
    assert detection["evaluation_failure_reason"]=="wrong_object_detection"
    group=summarize(rows)["groups"][0]
    assert group["raw_candidate_valid_visible_rate"]==1
    assert group["visible_detection_rate"]==0
    assert group["wrong_object_false_positives"]==1
    assert group["ball_missed_visible_frames"]==1
    sdk=group["methods"]["sdk_sphere"]
    assert sdk["visible_3d_valid_rate"]==1
    assert sdk["ball_matched_visible_3d_valid_rate"]==0
    # Wrong-object estimates remain in all-valid errors; never hide them by
    # dropping association failures from error calculation.
    assert sdk["center_world_error_m"]["samples"]==1
    assert sdk["center_world_error_m"]["mae"]==pytest.approx(.8)
    assert sdk["ball_matched_center_world_error_m"]["samples"]==0
    assert group["losses"]["detection"]["total_duration_s"]==pytest.approx(.1)
    assert group["losses"]["sdk_sphere"]["intervals"][0]["kind"]=="wrong_object_detection"


def test_renderer_bbox_is_preferred_only_in_offline_association():
    source=label(0)
    source["renderer_ball_bbox_xyxy"]=[95,35,105,45]
    candidate=prediction(0)
    candidate["left"].update(centroid_px=[100,40],bbox_xywh=[95,35,10,10])
    original=copy.deepcopy(candidate)
    row=evaluate_frames([candidate],[source],CALIBRATION,.1)[0]
    assert row["detection"]["match_bbox_basis"]=="renderer_ball_bbox"
    assert row["detection"]["ball_matched"]
    assert candidate==original


def test_short_terminal_sequence_uses_declared_end_not_full_nominal_period():
    labels=[label(0),label(1)]
    labels[-1]["observation_interval_end_s"]=.13
    rows=evaluate_frames([prediction(0,sdk=False),prediction(1,sdk=False)],labels,CALIBRATION,.1)
    loss=summarize(rows)["groups"][0]["losses"]["sdk_sphere"]["intervals"][0]
    assert loss["duration_s"]==pytest.approx(.13)
    assert loss["observed_midpoint_span_s"]==pytest.approx(.1)
    assert not loss["terminal_support_assumed"]


def test_near_clip_visibility_evidence_is_not_mislabeled_physical_occlusion():
    source=label(0,False)
    source["visibility_reason"]="occluded"
    source["visibility_evidence"]={"blocked_mesh_paths_per_ray":[["target_behind_near_clip_plane"]]}
    rows=evaluate_frames([prediction(0,False,False)],[source],CALIBRATION,.1)
    diagnostic=rows[0]["visibility_diagnostic"]
    assert diagnostic["reason"]=="near_clip_or_camera_sphere_intersection"
    assert diagnostic["source_reason"]=="occluded"
    assert summarize(rows)["groups"][0]["visibility_reason_categories"]=={"near_clip":1}
    assert rows[0]["detection"]["reason"]=="no_ball_contour"


def test_head_pose_conditions_are_not_mixed_in_exposure_comparison():
    a=label(0);a["head_pose"]="neutral"
    b=label(0,exposure=10);b["head_pose"]="down"
    rows=evaluate_frames([prediction(0)],[a],CALIBRATION,.1,"neutral")
    rows+=evaluate_frames([prediction(0)],[b],CALIBRATION,.1,"down")
    report=summarize(rows)
    assert len(report["groups"])==2
    assert not report["exposure_comparisons"]


def test_all_subframes_some_support_does_not_imply_fully_visible_silhouette():
    source=label(0)
    source["any_visible"]=source["all_visible"]=True
    source["visibility_evidence"]={"fully_in_fov":False,"physical_mesh_blocked_ray_count":2}
    row=evaluate_frames([prediction(0)],[source],CALIBRATION,.1)[0]
    diagnostic=row["visibility_diagnostic"]
    assert diagnostic["all_subframes_have_visible_support"]
    assert not diagnostic["all_visible"]
    assert diagnostic["partial_visibility"]
    assert diagnostic["expected_visible"]


def test_phase_preroll_frames_predicted_but_excluded_from_target_denominator():
    labels=[label(0),label(1)]
    labels[0]["phase"]="preroll"
    labels[1]["phase"]="target"
    records=evaluate_frames([prediction(0),prediction(1)],labels,CALIBRATION,.1,
                            source_sequence=[labels[1]])
    assert len(records)==1
    assert records[0]["frame_index"]==1


def test_finite_window_visible_support_uses_union_when_midpoint_ball_is_elsewhere():
    source=label(0,exposure=10)
    # The midpoint center is far right of the image; only the earlier window
    # sample has visible support. Matching remains an offline image annotation.
    source["ball_center_camera_m"]=[20,0,2]
    source["ball_world_position_m"]=[21,2,5]
    source["any_visible"]=True;source["all_visible"]=False
    source["subframe_visibility"]=[
        {"expected_visible":True,"visibility_evidence":{"center_optical_m":[0,0,2]}},
        {"expected_visible":False,"visibility_evidence":{"center_optical_m":[20,0,2]}}]
    row=evaluate_frames([prediction(0)],[source],CALIBRATION,.1)[0]
    match=row["detection"]
    assert match["midpoint_iou"]==0
    assert match["union_iou"]>.9
    assert match["ball_matched"]
    assert match["match_bbox_basis"]=="analytic_visible_subframe_sphere_union"
    assert match["union_subframes_used"]==1
    # Association cannot erase the large midpoint position error.
    assert row["methods"]["sdk_sphere"]["center_world_error_m"]>19


def test_zero_exposure_keeps_midpoint_bbox_and_finite_missing_gt_is_explicit_fallback():
    zero=label(0)
    zero["subframe_visibility"]=[{"expected_visible":True,"ball_center_camera_m":[1,0,2]}]
    a=evaluate_frames([prediction(0)],[zero],CALIBRATION,.1)[0]["detection"]
    assert a["ball_matched"]
    assert a["match_bbox_basis"]=="analytic_midpoint_sphere_silhouette"
    finite=label(0,exposure=10)
    b=evaluate_frames([prediction(0)],[finite],CALIBRATION,.1)[0]["detection"]
    assert b["ball_matched"]
    assert b["union_iou"]is None
    assert b["match_bbox_basis"]=="analytic_midpoint_fallback_missing_visible_subframe_gt"


def test_finite_any_subframe_visibility_not_midpoint_flag_sets_denominator():
    source=label(0,False,10)
    source.update(any_visible=True,all_visible=False)
    source["subframe_visibility"]=[{"expected_visible":True,"ball_center_camera_m":[0,0,2]},
                                    {"expected_visible":False,"ball_center_camera_m":[0,0,2]}]
    row=evaluate_frames([prediction(0)],[source],CALIBRATION,.1)[0]
    diagnostic=row["visibility_diagnostic"]
    assert not diagnostic["source_midpoint_expected_visible"]
    assert diagnostic["expected_visible"]
    assert diagnostic["reason"]=="partial_exposure_fov"
    group=summarize([row])["groups"][0]
    assert group["visible_frames"]==1
    assert group["detection_valid_visible_frames"]==1
    assert group["partly_visible_frames"]==1


def test_explicit_partial_near_clipping_is_preserved_not_physical_occlusion():
    source=label(0)
    source["visibility_reason"]="partially_near_clipped"
    source["visibility_evidence"]={"mesh_ray_count":9,"near_clipped_ray_count":3,
        "physical_mesh_blocked_ray_count":0,
        "blocked_mesh_paths_per_ray":[["target_behind_near_clip_plane"]]*3+[[]]*6}
    row=evaluate_frames([prediction(0)],[source],CALIBRATION,.1)[0]
    assert row["visibility_diagnostic"]["reason"]=="partially_near_clipped"
    assert row["visibility_diagnostic"]["expected_visible"]
    assert row["visibility_diagnostic"]["partial_visibility"]
    assert summarize([row])["groups"][0]["visibility_reason_categories"]=={"near_clip":1}
