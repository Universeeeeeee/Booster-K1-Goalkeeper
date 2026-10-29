"""Separate edge-circle architecture tests, before actual-frame evaluation."""
import cv2
import numpy as np
import pytest

from robot_perception_sim.edge_ball_perception import detect_edge_ball,DETECTOR_THRESHOLDS


def shaded_sphere():
    height,width=360,640
    image=np.full((height,width,3),245,dtype=np.uint8)
    v,u=np.indices((height,width))
    inside=(u-400)**2+(v-180)**2<=35**2
    shade=np.clip(230-(v-145)*1.5,100,240).astype(np.uint8)
    image[inside]=shade[inside,None]
    cv2.line(image,(5,300),(630,300),(80,80,80),3)
    cv2.rectangle(image,(40,140),(150,220),(100,100,100),2)
    return image


def test_shaded_white_sphere_on_bright_background_with_lines():
    result=detect_edge_ball(shaded_sphere())
    assert result["valid"],result["reason"]
    assert result["centroid_px"]==pytest.approx([400,180],abs=3)
    assert result["radius_px"]==pytest.approx(35,abs=3)
    assert DETECTOR_THRESHOLDS["hough_accumulator"]==30
    assert DETECTOR_THRESHOLDS["edge_coverage_min"]==.50


def test_blank_image_and_straight_field_lines_do_not_create_circle():
    image=np.full((360,640,3),245,dtype=np.uint8)
    cv2.line(image,(20,200),(620,200),(30,30,30),4)
    result=detect_edge_ball(image)
    assert not result["valid"]
    assert result["reason"]=="no_circle_support"


def test_two_comparable_edge_circles_return_ambiguous():
    image=np.full((360,640,3),245,dtype=np.uint8)
    cv2.circle(image,(350,150),35,(100,100,100),-1)
    cv2.circle(image,(500,150),35,(100,100,100),-1)
    result=detect_edge_ball(image)
    assert not result["valid"]
    assert result["reason"]=="ambiguous_circles"


def test_clipped_circle_is_not_a_valid_center():
    image=np.full((360,640,3),245,dtype=np.uint8)
    cv2.circle(image,(15,180),35,(100,100,100),-1)
    result=detect_edge_ball(image)
    assert not result["valid"]
    assert result["reason"]in("clipped_contour","no_circle_support")


def test_circle_overlapping_instrumentation_is_invalid_and_mask_stays_excluded():
    image=np.full((360,640,3),245,dtype=np.uint8)
    cv2.circle(image,(310,40),35,(100,100,100),-1)
    result=detect_edge_ball(image)
    assert not result["valid"]
    assert result["reason"]=="diagnostic_overlap"
    assert not result["mask"][18:34,18:298].any()
