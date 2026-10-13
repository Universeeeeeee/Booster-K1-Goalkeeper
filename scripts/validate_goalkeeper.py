"""Stage D: official walking in the shared soccer scene, without shots/impacts."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import torch
from k1_goalkeeper.scene import build_scene
from k1_goalkeeper.scene_simulation import SceneSimulation
from k1_goalkeeper.velocity_validation import Segment, run_case, recovery_metrics
from scripts.validate_walk import verify_sources


def protocol():
    zero = (0, 0, 0)
    cases = [("D4.1", "stand_30s", [Segment("stand", 30, zero)])]
    for speed in [.15, .2, .25]:
        for sign in [1, -1]:
            cases.append(("D4.2", f"lateral_{sign*speed:+g}", [Segment("settle", 2, zero),
                          Segment("move", 3, (0, sign*speed, 0)), Segment("recover", 5, zero)]))
    for name, command, duration in [("forward", (.4, 0, 0), 3), ("backward", (-.27, 0, 0), 2),
                                   *[(f"turn_{s:+g}", (0, 0, s), 3) for s in [.2, -.2, .4, -.4]]]:
        cases.append(("D4.3", name, [Segment("settle", 2, zero), Segment("move", duration, command),
                                    Segment("recover", 5, zero)]))
    cases.append(("D4.4", "switch_lateral_stop_reverse_turn", [Segment("settle", 2, zero),
                  Segment("left", 4, (0, .2, 0)), Segment("stop_left", 5, zero),
                  Segment("right", 4, (0, -.2, 0)), Segment("turn_left", 4, (0, 0, .6)),
                  Segment("turn_right", 4, (0, 0, -.6)), Segment("recover", 5, zero)]))
    return cases


def assess(summary, trajectory):
    checks = []
    for phase in summary["segments"]:
        cmd = np.asarray(phase["command"])
        if not cmd.any():
            continue
        mask = trajectory["segment_index"] == summary["segments"].index(phase)
        first, last = np.flatnonzero(mask)[[0, -1]] if mask.any() else (0, 0)
        if cmd[2]:
            response = phase.get("yaw_change_radians", 0) * np.sign(cmd[2])
            correct = response >= .05
        else:
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, trajectory["base_quaternion"][max(0, first-1)])
            direction = rotation.reshape(3, 3) @ np.array([cmd[0], cmd[1], 0])
            displacement = np.asarray(phase.get("displacement_world", [0, 0, 0]))
            response = float(displacement @ direction / np.linalg.norm(direction))
            correct = response >= .02
        checks.append({"segment": phase["name"], "direction_correct": bool(correct),
                       "signed_response_m_or_rad": float(response)})
    if not checks:
        velocities = np.column_stack([trajectory["body_linear_velocity"][:, :2],
                                      trajectory["body_angular_velocity"][:, 2]])
        summary["standing_stability"] = recovery_metrics(trajectory, velocities, 0, 30, summary["completed"])
    summary["direction_checks"] = checks
    summary["scene_integration_passed"] = bool(
        summary["completed"] and not summary["scene"]["unexpected_contact_events"]
        and summary["minimum_base_height_m"] >= .45 and summary["maximum_trunk_tilt_degrees"] <= 10
        and (bool(checks) or summary["standing_stability"]["verified"])
        and all(c["direction_correct"] for c in checks)
        and all(r["verified"] for r in summary["zero_recoveries"]))
    moving = [p for p in summary["segments"] if any(p["command"])]
    summary["maximum_steady_joint_saturation_fraction"] = max(
        (max(p.get("steady_torque_saturation_fraction", [0])) for p in moving), default=0.)


def write_report(report, output):
    rows = ["# 阶段 D 场景集成验证", "", f"{len(report['results'])} 个工况，每个重复 {report['repeat_count']} 次；"
            f"全部集成判据通过：{report['passed']}；轨迹重复一致：{report['all_repeats_identical']}。", "",
            "共用完整足球场景；physics_dt=1 ms，policy_dt=20 ms，decimation=20，每物理步执行PD。",
            "机器人初始位置(10.4,0,0.55369)m、航向180°；机身+X→世界−X，机身+Y→世界−Y。",
            "球门坐标采用原URDF框架：原点(11,0,0)m，+X=世界+Y横跨门口，+Y=世界−X指向场内，+Z向上。",
            "站立及短距离运动要求高度≥0.45m、倾角≤10°、无跌倒/数值异常/意外接触；"
            "平移方向响应≥0.02m，转向响应≥0.05rad；停止沿用阶段C恢复判据。",
            "短运动的跟踪误差与阶段C同阈值单独列出，不以短距离方向通过替代稳态跟踪通过。", "",
            "| 工况 | 集成通过 | C跟踪/恢复通过 | 最低高度m | 最大倾角° | 最慢恢复s | 稳态单关节饱和 |",
            "|---|---|---|---:|---:|---:|---:|"]
    for r in report["results"]:
        delays = [x["stop_delay_s"] for x in r["zero_recoveries"] if x["stop_delay_s"] is not None]
        delay = f"{max(delays):.1f}" if delays else "—"
        rows.append(f"| {r['name']} | {r['scene_integration_passed']} | {r['mobility_usable'] if r['direction_checks'] else '—'} | "
                    f"{r['minimum_base_height_m']:.4f} | {r['maximum_trunk_tilt_degrees']:.2f} | {delay} | "
                    f"{r['maximum_steady_joint_saturation_fraction']:.1%} |")
    rows += ["", "完整位移、瞬时/平均速度误差、PD及限幅、姿态、球门和足球坐标见 results.json 和各工况 trajectory.npz。",
             "没有执行足球射门、机器人受球撞击、高层位置控制或新策略训练。", ""]
    (output/"report.md").write_text("\n".join(rows))


def run(output, repeats=2):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output.mkdir(parents=True, exist_ok=True)
    manifest = verify_sources()
    protected = ["assets/booster_assets/robots/K1/K1_22dof.xml", "models/futsal_goal_3x2_reference.urdf",
                 "k1_goalkeeper/detection.py", "k1_goalkeeper/simulation.py", "k1_goalkeeper/scene.py",
                 "config/simulation.json", "reports/velocity_validation/results.json",
                 "reports/walk_validation/validation.json"]
    protected += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"tests").glob("test_*.py"))
                  if p.name != "test_scene_control.py"]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    model = mujoco.MjModel.from_xml_path(str(build_scene(output/"scene.xml")))
    simulation = SceneSimulation(model, control_mode="k1_walk")
    report = {"stage": "D", "repeat_count": repeats, "physics_dt": .001, "policy_dt": .02,
              "decimation": 20, "sources": manifest, "protected_files_sha256": hashes,
              "versions": {"python": platform.python_version(), "mujoco": mujoco.__version__,
                           "torch": torch.__version__, "numpy": np.__version__},
              "protocol": [{"stage": stage, "name": name, "segments": [asdict(s) for s in segments]}
                           for stage, name, segments in protocol()], "results": [], "skipped": []}
    for stage, name, segments in protocol():
        path = output/name
        summary, trajectory = run_case(name, segments, .001, output=path, simulation=simulation)
        assess(summary, trajectory)
        hashes_repeat = [summary["trajectory_sha256"]]
        for _ in range(repeats-1):
            repeated, repeated_trace = run_case(name, segments, .001, simulation=simulation)
            assess(repeated, repeated_trace)
            hashes_repeat.append(repeated["trajectory_sha256"])
        summary.update(stage=stage, reproducibility={"verified": repeats >= 2 and len(set(hashes_repeat)) == 1,
                                                     "repeat_hashes": hashes_repeat})
        (path/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        report["results"].append(summary)
        print(f"{stage} {name}: integration={summary['scene_integration_passed']} "
              f"tracking={summary['mobility_usable']} recoveries={[x['verified'] for x in summary['zero_recoveries']]} "
              f"repeat={summary['reproducibility']['verified']}", flush=True)
        (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if not summary["scene_integration_passed"]:
            report["skipped"] = [n for _, n, _ in protocol()[len(report["results"]):]]
            break
    verify_sources()
    report["protected_files_unchanged"] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report["all_repeats_identical"] = all(r["reproducibility"]["verified"] for r in report["results"])
    report["passed"] = bool(not report["skipped"] and len(report["results"]) == len(protocol())
                            and report["protected_files_unchanged"] and report["all_repeats_identical"]
                            and all(r["scene_integration_passed"] for r in report["results"]))
    report["assessment_code_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ["k1_goalkeeper/walk_controller.py", "k1_goalkeeper/scene_simulation.py",
         "k1_goalkeeper/velocity_validation.py", "scripts/validate_goalkeeper.py"]}
    (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    write_report(report, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/goalkeeper_validation")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.repeats)["passed"] else 1)
