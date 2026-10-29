"""Offline GT-assisted photometric diagnostics of paired 0/10 ms composites.

This is not a detector and never supplies pixels, crops or positions to one.
RGB differences alone do not establish physical shutter accuracy or depth gain.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from robot_perception_sim.continuous_perception import sphere_bbox
from robot_perception_sim.white_ball_perception import (
    DIAGNOSTIC_EXCLUSIONS_XYXY,DIAGNOSTIC_EXCLUSION_MARGIN_PX)


def rows(path):
    return [json.loads(s)for s in path.read_text().splitlines()if s.strip()]


def instrument_mask(shape):
    keep=np.ones(shape,dtype=bool)
    margin=DIAGNOSTIC_EXCLUSION_MARGIN_PX
    for x0,y0,x1,y1 in DIAGNOSTIC_EXCLUSIONS_XYXY:
        keep[max(0,y0-margin):min(shape[0],y1+margin),max(0,x0-margin):min(shape[1],x1+margin)]=False
    return keep


def photometric_metrics(baseline,finite,ball_boxes=(),visible_support=None,projection_unbounded=False):
    if baseline.shape!=finite.shape or baseline.ndim!=3 or baseline.shape[2]!=3:
        raise ValueError("Paired RGB images must have the same HxWx3 shape")
    difference=np.abs(baseline.astype(float)-finite.astype(float))
    keep=instrument_mask(baseline.shape[:2])
    roi=np.zeros(keep.shape,dtype=bool)
    for box in ball_boxes:
        if box is None:continue
        x0,y0,x1,y1=box
        x0,y0=max(0,int(np.floor(x0))),max(0,int(np.floor(y0)))
        x1,y1=min(keep.shape[1],int(np.ceil(x1))),min(keep.shape[0],int(np.ceil(y1)))
        if x1>x0 and y1>y0:roi[y0:y1,x0:x1]=True
    roi&=keep
    def metric(mask):
        return {"pixels":int(mask.sum()),"rgb_mae_8bit":float(difference[mask].mean())if mask.any()else None,
                "rgb_max_abs_8bit":float(difference[mask].max())if mask.any()else None}
    outside=keep&~roi
    if projection_unbounded or(visible_support and not roi.any()):outside=np.zeros_like(keep)
    return {"global":metric(keep),"gt_ball_union_roi":metric(roi),"outside_ball_roi":metric(outside),
            "gt_roi_status":("partial_finite_roi_with_camera_intersection"if roi.any()else"unavailable_camera_intersection")if projection_unbounded else
                "available"if roi.any()else"unavailable_for_visible_support"if visible_support else"not_visible_or_not_supplied",
            "roi_projection_unbounded":bool(projection_unbounded),
            "gt_roi_supplied_boxes":len(ball_boxes),"roi_is_offline_gt_assisted":True}


def _intrinsics(state,eye):
    matrix=np.asarray(state["K_per_eye"][eye]if "K_per_eye"in state else state["K"],dtype=float)
    return {"fx":matrix[0,0],"fy":matrix[1,1],"cx":matrix[0,2],"cy":matrix[1,2]}


def camera_center(state,eye):
    if eye=="left":return np.asarray(state["ball_center_camera_m"],dtype=float)
    inverse=np.linalg.inv(np.asarray(state["T_world_right_optical"],dtype=float))
    return (inverse@np.r_[state["ball_world_position_m"],1])[:3]


def unbounded_projection(acquisition,eye):
    states=[sample["state"]for sample in acquisition.get("subframes",[])]or[acquisition]
    return any(-state.get("ball_radius_m",.11)<camera_center(state,eye)[2]<=state.get("ball_radius_m",.11)for state in states)


def projected_boxes(acquisition,eye):
    boxes=[]
    samples=acquisition.get("subframes",[])
    states=[sample["state"]for sample in samples]or[acquisition]
    for state in states:
        if not state.get("expected_visible",False):continue
        center=camera_center(state,eye)
        box=sphere_bbox(center,state.get("ball_radius_m",.11),_intrinsics(state,eye))
        if box is not None:boxes.append(box)
    return boxes


def _source(directory):
    observations=rows(directory/"source_observations.jsonl")
    lookup={r["source_frame"]:r for r in rows(directory/"source_frames.jsonl")}
    result={}
    for row in observations:
        key=(row["trajectory_name"],row["head_pose"],round(row["trajectory_time_s"],9))
        if key in result:raise ValueError("Duplicate physical midpoint in source observations")
        if row["source_frame"]not in lookup:raise ValueError("Intended observation missing source composite record")
        result[key]=(row,lookup[row["source_frame"]]["acquisition"])
    return result


def compare(baseline_directory,finite_directory):
    a,b=_source(baseline_directory),_source(finite_directory)
    if set(a)!=set(b):raise ValueError("Exposure conditions do not have identical physical midpoint schedules")
    measurements=[]
    for key in sorted(a):
        a_label,a_state=a[key];b_label,b_state=b[key]
        if a_label["exposure_ms"]!=0 or b_label["exposure_ms"]!=10:
            raise ValueError("Photometric comparison requires 0 ms and 10 ms sources")
        if a_label["source_trajectory_sha256"]!=b_label["source_trajectory_sha256"]:
            raise ValueError("Exposure conditions use different frozen physical trajectories")
        if not np.allclose(a_label["T_world_camera"],b_label["T_world_camera"],atol=1e-7,rtol=0):
            raise ValueError("Exposure comparison camera poses differ")
        measurement={"trajectory_name":key[0],"head_pose":key[1],"physical_midpoint_s":key[2],
            "baseline_source_frame":a_label["source_frame"],"finite_source_frame":b_label["source_frame"],
            "any_visible":a_label.get("any_visible",a_label["expected_visible"])or b_label.get("any_visible",b_label["expected_visible"]),
            "baseline_midpoint_visible":a_label["expected_visible"],"finite_midpoint_visible":b_label["expected_visible"],"eyes":{}}
        for eye in("left","right"):
            images=[np.asarray(Image.open(directory/state["composite_paths"][eye]).convert("RGB"))
                    for directory,state in((baseline_directory,a_state),(finite_directory,b_state))]
            boxes=projected_boxes(a_state,eye)+projected_boxes(b_state,eye)
            measurement["eyes"][eye]=photometric_metrics(*images,boxes,visible_support=measurement["any_visible"],
                projection_unbounded=unbounded_projection(a_state,eye)or unbounded_projection(b_state,eye))
        measurements.append(measurement)
    aggregate={}
    for eye in("left","right"):
        aggregate[eye]={}
        for region in("global","gt_ball_union_roi","outside_ball_roi"):
            values=[r["eyes"][eye][region]for r in measurements]
            count=sum(v["pixels"]for v in values)
            aggregate[eye][region]={"pixels":count,"rgb_mae_8bit":sum(v["pixels"]*v["rgb_mae_8bit"]for v in values if v["pixels"])/count if count else None,
                "paired_frames_with_support":sum(v["pixels"]>0 for v in values)}
    return {"baseline_source":str(baseline_directory.resolve()),"finite_source":str(finite_directory.resolve()),
            "paired_midpoints":len(measurements),"physical_midpoints_match":True,
            "diagnostic":"GT-assisted offline photometric comparison; no prediction input",
            "units":"absolute 8-bit RGB channel levels averaged over eligible pixels and channels",
            "instrument_exclusions_xyxy":DIAGNOSTIC_EXCLUSIONS_XYXY,"instrument_margin_px":DIAGNOSTIC_EXCLUSION_MARGIN_PX,
            "aggregate":aggregate,"per_midpoint":measurements,
            "limitations":"RGB change includes renderer noise; compare outside-ball ROI. Does not alone validate shutter physics or depth accuracy."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-sender",type=Path,action="append",required=True)
    parser.add_argument("--finite-sender",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if len(args.baseline_sender)!=len(args.finite_sender):parser.error("Condition pair counts differ")
    report={"comparisons":[compare(a,b)for a,b in zip(args.baseline_sender,args.finite_sender)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x")as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")


if __name__=="__main__":main()
