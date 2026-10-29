"""G6-A fixed-robot continuous perception on recorded SDK RGB/depth/confidence.

Repeat --capture/--evaluation/--source-sequence for the four trajectory/exposure
conditions. Prediction APIs receive no GT, source poses or visibility labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_perception_sim import white_ball_perception, edge_ball_perception
DETECTORS={"white_hsv":(white_ball_perception.DETECTOR_NAME,white_ball_perception.DETECTOR_THRESHOLDS,white_ball_perception.process_white_frame),
           "edge_circle":(edge_ball_perception.DETECTOR_NAME,edge_ball_perception.DETECTOR_THRESHOLDS,edge_ball_perception.process_edge_frame)}
from robot_perception_sim.continuous_perception import evaluate_frames, summarize


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def predict_capture(capture, output, radius_m=.11, detector="white_hsv"):
    predictor=DETECTORS[detector][2]
    metadata = json.loads((capture/"metadata.json").read_text())
    if metadata.get("depth_source") != "zed_stereo_sdk" or metadata.get("depth_units") != "m":
        raise ValueError("Requires actual declared ZED SDK stereo depth in metres")
    calibration = {**metadata["calibration"], "baseline_m": metadata["baseline_m"]}
    for eye in ("left","right"):
        k = calibration[eye]
        if not all(np.isfinite(k[field])for field in ("fx","fy","cx","cy")) or min(k["fx"],k["fy"])<=0:
            raise ValueError("Invalid SDK calibration")
    if not np.isfinite(calibration["baseline_m"]) or calibration["baseline_m"]<=0:
        raise ValueError("Invalid SDK baseline")
    output.mkdir(parents=True,exist_ok=False)
    (output/"masks").mkdir()
    predictions, seen = [], set()
    previous_stamp = None
    for frame in json_rows(capture/"frames.jsonl"):
        index, stamp = frame["frame_index"],frame["sdk_timestamp_ns"]
        if index in seen or (previous_stamp is not None and stamp<=previous_stamp):
            raise ValueError("Capture IDs must be unique and SDK timestamps strictly increase")
        seen.add(index)
        previous_stamp = stamp
        shape = (frame["height"],frame["width"])
        images = [np.asarray(Image.open(capture/frame[f"{eye}_path"]).convert("RGB"))for eye in ("left","right")]
        if any(rgb.shape[:2]!=shape for rgb in images):
            raise ValueError("SDK image size differs from frame record")
        arrays=[]
        for kind in ("depth","confidence"):
            raw=np.fromfile(capture/frame[f"{kind}_path"],dtype="<f4")
            if raw.size!=shape[0]*shape[1]:raise ValueError(f"Invalid {kind} payload size")
            arrays.append(raw.reshape(shape))
        prediction=predictor(*images,*arrays,calibration,radius_m)
        prediction.update({"frame_index":index,"sdk_timestamp_ns":stamp})
        for eye in ("left","right"):
            relative=f"masks/{index:06d}_{eye}.png"
            Image.fromarray(prediction[eye].pop("mask")).save(output/relative)
            prediction[eye]["mask_path"]=relative
        predictions.append(prediction)
    (output/"predictions.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n"for r in predictions))
    return predictions,metadata,calibration


def run(captures,evaluations,output,source_sequences=None,radius_m=.11,detector="white_hsv"):
    if len(captures)!=len(evaluations) or (source_sequences is not None and len(source_sequences)!=len(captures)):
        raise ValueError("Capture/evaluation/source-sequence counts must agree")
    output.mkdir(parents=True,exist_ok=False)
    records, provenance = [],[]
    for index,(capture,evaluation) in enumerate(zip(captures,evaluations)):
        run_id=f"run_{index:03d}"
        destination=output/run_id
        predictions,metadata,calibration=predict_capture(capture,destination,radius_m,detector)
        # Source/GT files are opened only after image predictions are persisted.
        labels=json_rows(evaluation)
        sequence=json_rows(source_sequences[index])if source_sequences is not None else None
        fps=metadata["reported_camera_fps"]
        rows=evaluate_frames(predictions,labels,calibration,1/fps,run_id,sequence)
        records.extend(rows)
        (destination/"evaluation.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n"for r in rows))
        provenance.append({"run_id":run_id,"capture":str(capture.resolve()),"evaluation":str(evaluation.resolve()),
            "source_sequence":str(source_sequences[index].resolve())if source_sequences else None,
            "sdk_mode":metadata.get("depth_mode"),"sdk_version":metadata.get("sdk_version"),
            "reported_camera_fps":fps,"calibration":calibration,"source_sequence_complete":sequence is not None})
    report=summarize(records)
    report.update({"schema_version":1,"runs":provenance,"radius_prior_m":radius_m,
                   "predictor":DETECTORS[detector][0],"detector_thresholds":DETECTORS[detector][1],
                   "runtime_versions":{"numpy":np.__version__,"opencv":cv2.__version__,"Pillow":PIL.__version__},
                   "geometry":"unchanged G5 triangulate and fit_sdk_sphere valid rules",
                   "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in(
                       "robot_perception_sim/white_ball_perception.py","robot_perception_sim/edge_ball_perception.py","robot_perception_sim/ball_perception.py",
                       "robot_perception_sim/continuous_perception.py","scripts/run_continuous_perception.py")},
                   "provenance_verified_by_this_script":False})
    (output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    fields=["trajectory_name","head_pose","exposure_ms","method","visible_frames","timing_eligible_visible_frames",
            "missing_delivery_frames","raw_candidate_valid_visible_rate","visible_detection_rate","visible_3d_valid_rate",
            "ball_matched_visible_3d_valid_rate","center_world_mae_m",
            "center_world_p95_m","center_world_max_m","maximum_loss_duration_s","unobserved_gap_duration_s"]
    with (output/"metrics.csv").open("w",newline="")as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for group in report["groups"]:
            for method,metrics in group["methods"].items():
                row={key:group[key]for key in fields if key in group}
                row.update({"method":method,"visible_3d_valid_rate":metrics["visible_3d_valid_rate"],
                    "ball_matched_visible_3d_valid_rate":metrics["ball_matched_visible_3d_valid_rate"],
                    "center_world_mae_m":metrics["center_world_error_m"]["mae"],
                    "center_world_p95_m":metrics["center_world_error_m"]["p95"],
                    "center_world_max_m":metrics["center_world_error_m"]["max"],
                    "maximum_loss_duration_s":group["losses"][method]["maximum_duration_s"],
                    "unobserved_gap_duration_s":group["losses"][method]["unobserved_gap_duration_s"]})
                writer.writerow(row)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture",type=Path,action="append",required=True)
    parser.add_argument("--evaluation",type=Path,action="append",required=True)
    parser.add_argument("--source-sequence",type=Path,action="append")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--radius-m",type=float,default=.11)
    parser.add_argument("--detector",choices=DETECTORS,default="white_hsv")
    args=parser.parse_args()
    report=run(args.capture,args.evaluation,args.output,args.source_sequence,args.radius_m,args.detector)
    print(json.dumps({"conditions":len(report["groups"]),"output":str(args.output)}))


if __name__=="__main__":main()
