"""Photometric evidence is evaluator-only, independent of detector successes."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("exposure_images",Path(__file__).parents[1]/"scripts/compare_exposure_images.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_fixed_diagnostic_changes_do_not_count_as_exposure_image_change():
    zero=np.zeros((160,640,3),dtype=np.uint8);finite=zero.copy()
    finite[20:32,20:296]=255
    finite[45:57,20:296]=255
    result=m.photometric_metrics(zero,finite)
    assert result["global"]["rgb_mae_8bit"]==0
    assert result["gt_ball_union_roi"]["pixels"]==0
    assert result["gt_ball_union_roi"]["rgb_mae_8bit"]is None


def test_gt_roi_diagnostic_separates_ball_and_background_pixel_changes():
    zero=np.zeros((160,640,3),dtype=np.uint8);finite=zero.copy()
    finite[90:110,390:410]=30
    finite[130:140,500:510]=10
    result=m.photometric_metrics(zero,finite,[[390,90,410,110]])
    assert result["gt_ball_union_roi"]["rgb_mae_8bit"]==30
    assert result["gt_ball_union_roi"]["pixels"]==400
    assert 0<result["outside_ball_roi"]["rgb_mae_8bit"]<10
    assert result["roi_is_offline_gt_assisted"]


def test_roi_unions_do_not_double_count_and_clip_image_bounds():
    zero=np.zeros((160,640,3),dtype=np.uint8)
    result=m.photometric_metrics(zero,zero,[[390,90,410,110],[400,90,420,110],[-10,140,10,170]])
    assert result["gt_ball_union_roi"]["pixels"]==800


def test_mismatched_image_size_rejected():
    with pytest.raises(ValueError,match="same"):
        m.photometric_metrics(np.zeros((5,5,3)),np.zeros((5,6,3)))


def sender_fixture(directory,exposure,midpoint=.05):
    import json
    from PIL import Image
    directory.mkdir()
    image=np.zeros((160,640,3),dtype=np.uint8)
    if exposure:image[90:110,390:410]=30
    for eye in("left","right"):Image.fromarray(image).save(directory/f"{eye}.png")
    pose=np.eye(4).tolist()
    observation={"source_frame":1,"trajectory_name":"direct_goal","head_pose":"neutral",
        "trajectory_time_s":midpoint,"exposure_ms":exposure,"source_trajectory_sha256":"same-frozen-record",
        "T_world_camera":pose,"expected_visible":True}
    k=[[100,0,400],[0,100,100],[0,0,1]]
    state={"ball_center_camera_m":[0,0,2],"ball_world_position_m":[0,0,2],"ball_radius_m":.11,
        "T_world_right_optical":pose,"K_per_eye":{"left":k,"right":k},"expected_visible":True}
    acquisition={"subframes":[{"state":state}],"composite_paths":{"left":"left.png","right":"right.png"}}
    (directory/"source_observations.jsonl").write_text(json.dumps(observation)+"\n")
    (directory/"source_frames.jsonl").write_text(json.dumps({"source_frame":1,"acquisition":acquisition})+"\n")
    return directory


def test_source_pair_uses_same_physical_midpoint_and_independent_subframe_gt(tmp_path):
    a=sender_fixture(tmp_path/"zero",0)
    b=sender_fixture(tmp_path/"finite",10)
    report=m.compare(a,b)
    assert report["paired_midpoints"]==1
    assert report["aggregate"]["left"]["gt_ball_union_roi"]["rgb_mae_8bit"]==30
    assert report["aggregate"]["left"]["global"]["rgb_mae_8bit"]>0
    assert "GT-assisted offline"in report["diagnostic"]


def test_source_pair_rejects_five_ms_midpoint_shift(tmp_path):
    a=sender_fixture(tmp_path/"zero",0)
    b=sender_fixture(tmp_path/"finite",10,.055)
    with pytest.raises(ValueError,match="identical physical midpoint"):
        m.compare(a,b)


def test_visible_camera_intersection_without_finite_roi_is_not_background_noise():
    a=np.zeros((160,640,3),dtype=np.uint8);b=np.full_like(a,30)
    report=m.photometric_metrics(a,b,visible_support=True)
    assert report["global"]["rgb_mae_8bit"]==30
    assert report["gt_roi_status"]=="unavailable_for_visible_support"
    assert report["gt_ball_union_roi"]["rgb_mae_8bit"]is None
    assert report["outside_ball_roi"]["pixels"]==0
    assert report["outside_ball_roi"]["rgb_mae_8bit"]is None


def test_camera_plane_intersection_with_invisible_flag_is_not_clean_background():
    a=np.zeros((160,640,3),dtype=np.uint8);b=np.full_like(a,20)
    report=m.photometric_metrics(a,b,visible_support=False,projection_unbounded=True)
    assert report["global"]["rgb_mae_8bit"]==20
    assert report["gt_roi_status"]=="unavailable_camera_intersection"
    assert report["outside_ball_roi"]["rgb_mae_8bit"]is None
