"""Frozen G6 white detector synthetic checks; no actual-frame threshold tuning."""
import cv2
import numpy as np
import pytest

from robot_perception_sim.white_ball_perception import (
    DETECTOR_NAME, DETECTOR_THRESHOLDS, detect_white_ball, process_white_frame)


def field():
    rgb=np.zeros((160,240,3),dtype=np.uint8)
    rgb[:]=[25,105,35]
    return rgb


def ball(rgb,center=(120,80),radius=12):
    cv2.circle(rgb,center,radius,(245,245,245),-1)
    return rgb


def test_white_ball_with_larger_fieldline_and_rectangle_distractors():
    rgb=field()
    cv2.line(rgb,(5,140),(230,140),(255,255,255),5)
    cv2.rectangle(rgb,(10,15),(55,50),(230,230,230),-1)
    ball(rgb)
    result=detect_white_ball(rgb)
    assert result["valid"]
    assert result["centroid_px"]==pytest.approx([120,80])
    assert result["bbox_xywh"]==[108,68,25,25]
    assert result["detector"]==DETECTOR_NAME=="g6_white_circle_mvp"
    assert result["mask"][140,120]==0
    assert result["mask"][25,25]==0
    assert DETECTOR_THRESHOLDS["frozen_before_actual_white_capture_inspection"]


def test_comparable_white_round_distractor_returns_ambiguous_not_gt_selection():
    rgb=ball(field(),(80,80));ball(rgb,(170,80))
    result=detect_white_ball(rgb)
    assert not result["valid"]
    assert result["reason"]=="ambiguous_contours"


def test_clipped_ball_returns_invalid():
    result=detect_white_ball(ball(field(),(4,80),12))
    assert not result["valid"]
    assert result["reason"]in("clipped_contour","no_ball_contour")


def test_half_occlusion_and_full_occlusion_do_not_fill_hidden_ball():
    rgb=ball(field())
    rgb[:,:120]=[30,30,30]
    assert not detect_white_ball(rgb)["valid"]
    assert not detect_white_ball(field())["valid"]


def test_connected_white_line_remains_explicit_failure():
    rgb=ball(field())
    cv2.line(rgb,(5,80),(230,80),(255,255,255),5)
    result=detect_white_ball(rgb)
    assert not result["valid"]
    assert result["reason"]=="no_ball_contour"


def test_orange_ball_is_not_a_white_ball_positive():
    rgb=field();cv2.circle(rgb,(120,80),12,(255,130,0),-1)
    assert not detect_white_ball(rgb)["valid"]


def test_geometry_reuses_g5_invalids_without_gt_fallback():
    left=ball(field());right=ball(field(),(110,80))
    depth=np.full((160,240),np.nan,dtype=np.float32)
    k={"fx":100.,"fy":100.,"cx":120.,"cy":80.}
    result=process_white_frame(left,right,depth,np.zeros_like(depth),
                               {"left":k,"right":k,"baseline_m":.12})
    assert result["left"]["valid"]
    assert result["triangulation"]["valid"]
    assert result["triangulation"]["center_camera_m"][2]==pytest.approx(1.2)
    assert not result["sdk_sphere"]["valid"]
    assert result["sdk_sphere"]["reason"]=="insufficient_foreground_depth"
    assert result["sdk_sphere"]["center_camera_m"]is None


def test_small_white_square_glyphs_are_an_explicit_ambiguity_limit():
    # Pixel-size effects make 12px solid squares pass the frozen extent filter.
    # Preserve this known failure rather than silently selecting a glyph as GT.
    rgb=field()
    cv2.rectangle(rgb,(180,120),(191,131),(255,255,255),-1)
    cv2.rectangle(rgb,(210,120),(221,131),(255,255,255),-1)
    result=detect_white_ball(rgb)
    assert not result["valid"]
    assert result["reason"]=="ambiguous_contours"


def test_fixed_diagnostic_glyphs_excluded_without_affecting_ball_or_thresholds():
    rgb=np.zeros((180,640,3),dtype=np.uint8);rgb[:]=[25,105,35]
    for y in(20,45):
        for x in range(20,296,24):cv2.rectangle(rgb,(x,y),(x+11,y+11),(255,255,255),-1)
    cv2.line(rgb,(350,150),(620,150),(255,255,255),5)
    cv2.rectangle(rgb,(400,10),(450,50),(255,255,255),-1)
    ball(rgb,(400,100))
    result=detect_white_ball(rgb)
    assert result["valid"]
    assert result["centroid_px"]==pytest.approx([400,100])
    assert not result["mask"][18:34,18:298].any()
    assert not result["mask"][43:59,18:298].any()
    assert DETECTOR_THRESHOLDS["circularity_min"]==.70
    assert DETECTOR_THRESHOLDS["diagnostic_exclusion_margin_px"]==2


def test_ball_touching_instrumented_exclusion_returns_invalid_no_reconstruction():
    rgb=np.zeros((180,640,3),dtype=np.uint8);rgb[:]=[25,105,35]
    ball(rgb,(310,26),20)
    result=detect_white_ball(rgb)
    assert not result["valid"]
    assert result["reason"]=="diagnostic_overlap"
    assert not result["mask"][18:34,18:298].any()
