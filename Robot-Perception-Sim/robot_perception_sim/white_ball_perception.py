"""G6-only full-frame white sphere MVP, frozen before actual capture inspection.

This detector deliberately has no GT, field coordinates, pose, crop or identity
input. A plain white ball can be confused with white scene geometry; ambiguous
or merged contours remain failures rather than being selected using GT.
"""
import cv2
import numpy as np

from robot_perception_sim.ball_perception import fit_sdk_sphere, triangulate

DETECTOR_NAME = "g6_white_circle_mvp"
DIAGNOSTIC_EXCLUSIONS_XYXY = ((20,20,296,32),(20,45,296,57))
DIAGNOSTIC_EXCLUSION_MARGIN_PX = 2
DETECTOR_THRESHOLDS = {"hsv_saturation_max":60,"hsv_value_min":180,"area_min_px2":6,
    "circularity_min":.70,"bbox_aspect_min":.65,"bbox_aspect_max":1.5,
    "contour_bbox_extent_min":.45,"contour_bbox_extent_max":.90,
    "ambiguity_second_area_fraction":.5,
    "frozen_before_actual_white_capture_inspection":True,
    "diagnostic_exclusions_xyxy":DIAGNOSTIC_EXCLUSIONS_XYXY,
    "diagnostic_exclusion_margin_px":DIAGNOSTIC_EXCLUSION_MARGIN_PX,
    "diagnostic_exclusion_basis":"fixed source instrumentation coordinates; never GT"}


def detect_white_ball(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    raw=cv2.inRange(hsv,(0,0,180),(179,60,255))
    excluded=np.zeros(raw.shape,dtype=np.uint8)
    margin=DIAGNOSTIC_EXCLUSION_MARGIN_PX
    for x0,y0,x1,y1 in DIAGNOSTIC_EXCLUSIONS_XYXY:
        excluded[max(0,y0-margin):min(raw.shape[0],y1+margin),
                 max(0,x0-margin):min(raw.shape[1],x1+margin)]=255
    raw[excluded>0]=0
    exclusion_boundary=cv2.dilate(excluded,np.ones((3,3),np.uint8))>0
    contours,_=cv2.findContours(raw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    rejected_diagnostic_overlap=False
    for contour in contours:
        area=cv2.contourArea(contour)
        perimeter=cv2.arcLength(contour,True)
        if area<6 or perimeter<=0:continue
        pixels=contour.reshape(-1,2)
        if exclusion_boundary[pixels[:,1],pixels[:,0]].any():
            rejected_diagnostic_overlap=True
        x,y,w,h=cv2.boundingRect(contour)
        circularity=4*np.pi*area/perimeter**2
        aspect=w/h
        extent=area/(w*h)
        if circularity>=.70 and .65<=aspect<=1.5 and .45<=extent<=.90:
            candidates.append((area,circularity,contour,[x,y,w,h]))
    mask=np.zeros(raw.shape,dtype=np.uint8)
    if not candidates:
        return {"valid":False,"reason":"diagnostic_overlap"if rejected_diagnostic_overlap else"no_ball_contour","mask":mask,"bbox_xywh":None,
                "centroid_px":None,"confidence":0.,"mask_pixels":0,"mask_fraction":0.,
                "detector":DETECTOR_NAME}
    candidates.sort(key=lambda item:item[0],reverse=True)
    area,circularity,contour,bbox=candidates[0]
    x,y,w,h=bbox
    clipped=x==0 or y==0 or x+w==rgb.shape[1]or y+h==rgb.shape[0]
    ambiguous=len(candidates)>1 and candidates[1][0]>.5*area
    cv2.drawContours(mask,[contour],-1,255,-1)
    diagnostic_overlap=bool(np.any((mask>0)&exclusion_boundary))
    mask[excluded>0]=0
    moments=cv2.moments(contour)
    return {"valid":not(clipped or ambiguous or diagnostic_overlap),
            "reason":"diagnostic_overlap"if diagnostic_overlap else"clipped_contour"if clipped else"ambiguous_contours"if ambiguous else None,
            "mask":mask,"bbox_xywh":bbox,
            "centroid_px":[moments["m10"]/moments["m00"],moments["m01"]/moments["m00"]],
            "confidence":float(min(1,circularity)),"mask_pixels":int(np.count_nonzero(mask)),
            "mask_fraction":float(np.count_nonzero(mask)/mask.size),"detector":DETECTOR_NAME}


def process_white_frame(left_rgb,right_rgb,depth,confidence,calibration,radius_m=.11):
    if not np.isfinite(radius_m)or radius_m<=0:raise ValueError("radius_m must be finite and positive")
    if depth.shape!=left_rgb.shape[:2]or confidence.shape!=depth.shape:
        raise ValueError("SDK depth/confidence and left image sizes must agree")
    left,right=detect_white_ball(left_rgb),detect_white_ball(right_rgb)
    return {"left":left,"right":right,"triangulation":triangulate(left,right,calibration),
            "sdk_sphere":fit_sdk_sphere(depth,confidence,left,calibration["left"],radius_m)}
