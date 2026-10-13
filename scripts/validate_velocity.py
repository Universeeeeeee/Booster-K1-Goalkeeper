"""Stage C: low-to-moderate command ladders, frame checks, switches, and recovery."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch

from k1_goalkeeper.velocity_validation import Segment, run_case
from scripts.validate_walk import verify_sources


def protocol():
    groups = []
    for direction, axis, values in [
        ("forward", 0, [.05, .1, .15, .2, .3, .4, .6, .8]),
        ("backward", 0, [-.05, -.1, -.15, -.2, -.25, -.27, -.28]),
        ("left", 1, [.05, .1, .15, .2, .25]),
        ("right", 1, [-.05, -.1, -.15, -.2, -.25]),
        ("turn_left", 2, [.1, .2, .4, .6, .8]),
        ("turn_right", 2, [-.1, -.2, -.4, -.6, -.8]),
    ]:
        cases = []
        for value in values:
            command = [0., 0., 0.]
            command[axis] = value
            cases.append((f"{direction}_{abs(value):g}", [Segment("settle", 2, (0, 0, 0)),
                          Segment("move", 6, tuple(command)), Segment("recover", 5, (0, 0, 0))], 0.))
        groups.append((direction, cases))
    frames = []
    for degrees in [90, 180]:
        for name, command in [("forward", (.4, 0, 0)), ("backward", (-.25, 0, 0)),
                              ("left", (0, .2, 0)), ("right", (0, -.2, 0)),
                              ("turn_left", (0, 0, .4)), ("turn_right", (0, 0, -.4))]:
            frames.append((f"frame_{degrees}_{name}", [Segment("settle", 2, (0, 0, 0)),
                           Segment("move", 6, command), Segment("recover", 5, (0, 0, 0))], math.radians(degrees)))
    groups.append(("frames", frames))
    groups.append(("switches", [
        ("switch_forward_back_stop", [Segment("settle", 2, (0, 0, 0)),
         Segment("forward_small", 4, (.3, 0, 0)), Segment("forward_faster", 4, (.6, 0, 0)),
         Segment("stop_forward", 5, (0, 0, 0)), Segment("backward", 4, (-.25, 0, 0)),
         Segment("recover", 5, (0, 0, 0))], 0.),
        ("switch_lateral_turn_stop", [Segment("settle", 2, (0, 0, 0)),
         Segment("left", 4, (0, .2, 0)), Segment("right", 4, (0, -.2, 0)),
         Segment("turn_left", 4, (0, 0, .6)), Segment("turn_right", 4, (0, 0, -.6)),
         Segment("recover", 5, (0, 0, 0))], 0.),
    ]))
    return groups


def write_brief_report(report, output):
    rows = report["results"]
    text = ["# 阶段 C 基础速度控制验证", "",
            f"独立 K1 模型；{len(rows)} 个工况；每个工况重复 {report['criteria']['repeat_count']} 次。",
            f"跌倒 {report['fall_count']} 次；轨迹重复一致：{report['all_repeats_identical']}；"
            f"坐标方向验证：{report['all_frame_directions_correct']}；全部运动后停止恢复：{report['all_post_motion_recoveries_verified']}。", "",
            "下表仅列同时满足持续方向、平均跟踪和停止恢复判据的已测指令点；未证明点之间的连续范围。力矩饱和单独评估。", "",
            "| 机身方向 | 2 ms 通过指令 | 1 ms 通过指令 |", "|---|---|---|"]
    for prefix, label, axis in [("forward", "前进 m/s", 0), ("backward", "后退 m/s", 0),
                                ("left", "左移 m/s", 1), ("right", "右移 m/s", 1),
                                ("turn_left", "左转 rad/s", 2), ("turn_right", "右转 rad/s", 2)]:
        values = []
        for dt in [.002, .001]:
            matched = [r["segments"][1]["command"][axis] for r in rows
                       if r["physics_dt"] == dt and r["name"].startswith(prefix+"_") and r["mobility_usable"]]
            values.append(", ".join(f"{v:g}" for v in matched) or "无")
        text.append(f"| {label} | {values[0]} | {values[1]} |")
    text.extend(["", "+X 前进、+Y 左移、+Z 正向角速度为俯视逆时针；机身朝向改变时平移指令随之旋转。", "",
                 "| 工况 | dt ms | 完成 | 方向/跟踪/恢复通过 | 停止恢复延迟 s | 最大稳态单关节饱和比例 |", "|---|---:|---|---|---:|---:|"])
    for row in rows:
        delay = row["zero_recovery"]["stop_delay_s"]
        text.append(f"| {row['name']} | {row['physics_dt']*1000:g} | {row['completed']} | {row['mobility_usable']} | "
                    f"{f'{delay:.1f}' if delay is not None else '—'} | {row['maximum_steady_joint_saturation_fraction']:.1%} |")
    text.extend(["", "停止延迟指进入连续一秒低速区间的起点，分辨率0.2 s；确认该状态还需观察后续一秒。",
                 "位移、姿态、高度、瞬时RMSE、平均误差、原始PD需求及逐关节饱和比例见 results.json 与各工况 trajectory.npz。",
                 "能力限制及守门用途解释见项目 docs/velocity_validation.md；本轮没有执行场景接入或足球碰撞测试。", ""])
    (output / "report.md").write_text("\n".join(text))


def run(output, repeats=2, case_filter=None):
    manifest = verify_sources()
    protected = ["k1_goalkeeper/walk_controller.py", "scripts/validate_walk.py",
                 "assets/booster_assets/robots/K1/K1_22dof.xml",
                 "k1_goalkeeper/scene.py", "k1_goalkeeper/simulation.py", "k1_goalkeeper/detection.py",
                 "config/simulation.json", "models/futsal_goal_3x2_reference.urdf",
                 "scenes/goalkeeper.xml", "scenes/goalkeeper_fixed.xml", "scenes/goalkeeper_absent.xml"]
    protected_hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in protected}
    output.mkdir(parents=True, exist_ok=True)
    groups = protocol()
    report = {
        "scope": "Stage C only, independent K1 model; no soccer integration, impacts, or training",
        "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                     "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 1},
        "sources": manifest, "protected_files_sha256": protected_hashes,
        "protocol": [{"group": group, "cases": [{"name": name, "initial_yaw": yaw,
                      "segments": [asdict(s) for s in segments]} for name, segments, yaw in cases]}
                     for group, cases in groups],
        "criteria": {"fall_base_height_m": .3, "fall_tilt_degrees": 60,
                     "steady_discard_s": "min(2, half segment duration)",
                     "direction_gain_range": [.5, 1.5], "tracking_mean_error": "max(.05 m/s or .08 rad/s, 30% of command)",
                     "zero_recovery": "every post-motion stop: last 2s mean vx/vy <= .05 m/s, wz <= .1 rad/s, z >= .45m, tilt <= 10deg; 1s consecutive slow .2s bins",
                     "saturation": "raw PD demand reaches task effort limit, each physics substep; does not by itself label a fall",
                     "torque_quality_limit_fraction": .05, "seed": 0, "repeat_count": repeats},
        "results": [], "skipped": [], "case_filter": case_filter,
    }
    for dt in [.002, .001]:
        for group, cases in groups:
            stopped = False
            for name, segments, yaw in cases:
                if case_filter and case_filter not in name:
                    continue
                if stopped:
                    report["skipped"].append({"physics_dt": dt, "case": name, "reason": "prior larger-risk direction test failed stability"})
                    continue
                path = output / ("dt_2ms" if dt == .002 else "dt_1ms") / name
                summary, trajectory = run_case(name, segments, dt, yaw, 0, path)
                first_hash = summary["trajectory_sha256"]
                repeat_hashes = [first_hash]
                for _ in range(repeats - 1):
                    repeated, _ = run_case(name, segments, dt, yaw, 0)
                    repeat_hashes.append(repeated["trajectory_sha256"])
                summary["reproducibility"] = {"verified": repeats >= 2 and len(set(repeat_hashes)) == 1,
                                               "repeat_hashes": repeat_hashes}
                moving = [r for r in summary["segments"] if any(r["command"]) and "mean_body_velocity" in r]
                maximum_saturation = max((max(r["steady_torque_saturation_fraction"]) for r in moving), default=0)
                summary["torque_quality_ok"] = maximum_saturation <= .05
                summary["maximum_steady_joint_saturation_fraction"] = maximum_saturation
                if group == "frames":
                    phase = moving[0] if moving else None
                    correct = False
                    if phase:
                        cmd = np.asarray(phase["command"])
                        if cmd[2]:
                            correct = phase["yaw_change_radians"] * cmd[2] > 0
                        else:
                            body_expected = np.array([cmd[0], cmd[1]])
                            rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
                            correct = np.dot(np.asarray(phase["steady_displacement_world"])[:2], rotation @ body_expected) > 0
                    summary["frame_direction_correct"] = bool(correct)
                (path / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
                report["results"].append(summary)
                print(f"dt={dt:g} {name}: completed={summary['completed']} usable={summary['mobility_usable']} "
                      f"recovery={summary['zero_recovery']['verified']} sat={maximum_saturation:.1%} "
                      f"repeat={summary['reproducibility']['verified']}", flush=True)
                (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                if not summary["completed"] and group not in ["frames", "switches"]:
                    stopped = True
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name, digest in protected_hashes.items())
    verify_sources()
    report["protected_files_unchanged"] = True
    report["all_repeats_identical"] = bool(report["results"]) and repeats >= 2 and all(r["reproducibility"]["verified"] for r in report["results"])
    report["fall_count"] = sum(len(r["fall_events"]) for r in report["results"])
    frame_rows = [r for r in report["results"] if "frame_direction_correct" in r]
    report["frame_case_count"] = len(frame_rows)
    report["all_frame_directions_correct"] = all(r["frame_direction_correct"] for r in frame_rows) if frame_rows else None
    report["all_post_motion_recoveries_verified"] = all(m["verified"] for r in report["results"] for m in r["zero_recoveries"])
    report["maximum_velocity_position_consistency_error_m_s"] = max(
        (abs(v) for r in report["results"] for s in r["segments"] if "velocity_position_consistency_error" in s
         for v in s["velocity_position_consistency_error"]), default=0)
    report["assessment_code_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                        for name in ["k1_goalkeeper/velocity_validation.py", "scripts/validate_velocity.py"]}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_brief_report(report, output)
    print(f"Saved {len(report['results'])} cases to {output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/velocity_validation")
    parser.add_argument("--repeats", type=int, default=2, help="Same seed/state repeats; <2 is not a reproducibility verification")
    parser.add_argument("--case", help="Run only names containing this text")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    report = run(args.output, args.repeats, args.case)
    raise SystemExit(0 if report["results"] and not report["fall_count"] and (args.repeats < 2 or report["all_repeats_identical"]) and
                     all(r["completed"] for r in report["results"]) else 1)
