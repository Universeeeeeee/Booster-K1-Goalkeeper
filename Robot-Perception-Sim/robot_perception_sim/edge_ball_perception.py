"""G6 image-only edge-circle MVP; a separate architecture from frozen white HSV.

Parameters were fixed before actual evaluation. No GT, crop, expected radius in
pixels, pose or candidate identity enters detection. Circle masks do not imply
that their 3D correspondence or object identity is correct.
"""
import cv2
import numpy as np

from robot_perception_sim.ball_perception import fit_sdk_sphere, triangulate
from robot_perception_sim.white_ball_perception import (
    DIAGNOSTIC_EXCLUSIONS_XYXY, DIAGNOSTIC_EXCLUSION_MARGIN_PX)

DETECTOR_NAME="g6_edge_circle_mvp"
DETECTOR_THRESHOLDS={"gaussian_kernel":[5,5],"gaussian_sigma":1.5,"hough_dp":1,
    "hough_min_dist_px":30,"hough_canny_high":100,"hough_accumulator":30,
    "radius_min_px":6,"radius_max":"image_height/2","edge_canny_low":50,
    "edge_canny_high":100,"edge_coverage_angles":72,"edge_radial_band_px":2,
    "edge_coverage_min":.50,"ambiguity_second_score_fraction":.90,
    "diagnostic_exclusions_xyxy":DIAGNOSTIC_EXCLUSIONS_XYXY,
    "diagnostic_exclusion_margin_px":DIAGNOSTIC_EXCLUSION_MARGIN_PX,
    "frozen_before_actual_edge_evaluation":True}


def detect_edge_ball(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),1.5)
    edges=cv2.Canny(gray,50,100)
    excluded=np.zeros(gray.shape,dtype=np.uint8)
    m=DIAGNOSTIC_EXCLUSION_MARGIN_PX
    for x0,y0,x1,y1 in DIAGNOSTIC_EXCLUSIONS_XYXY:
        excluded[max(0,y0-m):min(gray.shape[0],y1+m),max(0,x0-m):min(gray.shape[1],x1+m)]=255
    # The Hough proposal is image-only. Instrument pixels cannot support its
    # independent edge-coverage score, and an overlapping selected circle is
    # invalid rather than reconstructed through the excluded region.
    edges[cv2.dilate(excluded,np.ones((3,3),np.uint8))>0]=0
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1,minDist=30,
        param1=100,param2=30,minRadius=6,maxRadius=gray.shape[0]//2)
    candidates=[]
    angles=np.arange(72)*2*np.pi/72
    if circles is not None:
        for x,y,radius in circles[0]:
            supported=np.zeros(72,dtype=bool)
            for delta in(-2,-1,0,1,2):
                u=np.rint(x+(radius+delta)*np.cos(angles)).astype(int)
                v=np.rint(y+(radius+delta)*np.sin(angles)).astype(int)
                inside=(u>=0)&(u<gray.shape[1])&(v>=0)&(v<gray.shape[0])
                supported[inside]|=edges[v[inside],u[inside]]>0
            score=float(supported.mean())
            if score>=.50:candidates.append((score,float(x),float(y),float(radius)))
    mask=np.zeros(gray.shape,dtype=np.uint8)
    if not candidates:
        return {"valid":False,"reason":"no_circle_support","mask":mask,"bbox_xywh":None,
            "centroid_px":None,"confidence":0.,"mask_pixels":0,"mask_fraction":0.,"detector":DETECTOR_NAME}
    candidates.sort(key=lambda c:c[0],reverse=True)
    score,x,y,radius=candidates[0]
    cv2.circle(mask,(round(x),round(y)),round(radius),255,-1)
    overlap=bool(np.any((mask>0)&cv2.dilate(excluded,np.ones((3,3),np.uint8)).astype(bool)))
    mask[excluded>0]=0
    clipped=x-radius<0 or y-radius<0 or x+radius>=gray.shape[1]or y+radius>=gray.shape[0]
    ambiguous=len(candidates)>1 and candidates[1][0]>=.90*score
    x0,y0,x1,y1=int(np.floor(x-radius)),int(np.floor(y-radius)),int(np.ceil(x+radius)),int(np.ceil(y+radius))
    return {"valid":not(overlap or clipped or ambiguous),
        "reason":"diagnostic_overlap"if overlap else"clipped_contour"if clipped else"ambiguous_circles"if ambiguous else None,
        "mask":mask,"bbox_xywh":[x0,y0,x1-x0,y1-y0],"centroid_px":[x,y],"confidence":score,
        "mask_pixels":int(np.count_nonzero(mask)),"mask_fraction":float(np.count_nonzero(mask)/mask.size),
        "radius_px":radius,"circle_candidates":len(candidates),"detector":DETECTOR_NAME}


def process_edge_frame(left_rgb,right_rgb,depth,confidence,calibration,radius_m=.11):
    if not np.isfinite(radius_m)or radius_m<=0:raise ValueError("radius_m must be finite and positive")
    if depth.shape!=left_rgb.shape[:2]or confidence.shape!=depth.shape:
        raise ValueError("SDK depth/confidence and left image sizes must agree")
    left,right=detect_edge_ball(left_rgb),detect_edge_ball(right_rgb)
    return {"left":left,"right":right,"triangulation":triangulate(left,right,calibration),
        "sdk_sphere":fit_sdk_sphere(depth,confidence,left,calibration["left"],radius_m)}
