"""Stage E: graded football impacts with exact absent-scene launch controls."""
import argparse
from dataclasses import replace
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
from k1_goalkeeper.impact_validation import ImpactCase, run_impact
from k1_goalkeeper.scene import build_scene
from scripts.validate_walk import verify_sources


def protocol():
    groups = [("front", [ImpactCase(f"front_{s}", speed=s) for s in [2, 4, 6, 8, 12, 16, 20, 24, 30]])]
    for sign in [-1, 1]:
        groups.append((f"eccentric_{sign:+d}", [ImpactCase(f"eccentric_{sign:+d}_{s}", speed=s, offset_y=sign*.07) for s in [2, 4, 8]]))
    for side in ["left", "right"]:
        for region, body, index in [("foot", f"{side}_ankle_roll_link", 0), ("shank", f"{side}_knee_pitch_link", 1)]:
            groups.append((f"{side}_{region}", [ImpactCase(f"{side}_{region}_{s}", body, s, target_geom_index=index) for s in [2, 4, 8]]))
    groups.append(("moving", [ImpactCase(f"moving_{s}", speed=s, command=(0, .2, 0), moving_duration=2) for s in [2, 4, 8]]))
    groups.append(("other_coverage", [ImpactCase("head_2", "aahead_pitch_link"),
                   ImpactCase("left_forearm_2", "left_elbow_yaw_link"), ImpactCase("right_forearm_2", "right_elbow_yaw_link")]))
    groups.append(("no_contact", [ImpactCase("miss_4", speed=4, miss=True),
                  ImpactCase("moving_miss_4", speed=4, miss=True, command=(0, .2, 0), moving_duration=2)]))
    return groups


def compare(summary, trace, absent, reference):
    if not summary["launch"] or not absent["launch"]:
        return {"trajectory_changed": None, "successfully_blocked": None, "reason": "launch unavailable"}
    start = summary["launch"]["time"]
    ref_start = absent["launch"]["time"]
    selected = trace["time"] >= start-1e-9
    elapsed = trace["time"][selected]-start
    ref_times = reference["time"]-ref_start
    control_position = np.column_stack([np.interp(elapsed, ref_times, reference["ball_position"][:, i]) for i in range(3)])
    control_velocity = np.column_stack([np.interp(elapsed, ref_times, reference["ball_velocity"][:, i]) for i in range(3)])
    deflection = float(np.linalg.norm(trace["ball_position"][selected]-control_position, axis=1).max())
    velocity_change = float(np.linalg.norm(trace["ball_velocity"][selected]-control_velocity, axis=1).max())
    defended_goal = summary.get("defended_goal", summary["goal"] and bool(summary.get("exit_event") and summary["exit_event"].get("goal") == "right"))
    control_goal = absent.get("defended_goal", absent["goal"] and bool(absent.get("exit_event") and absent["exit_event"].get("goal") == "right"))
    blocked = bool(summary["effective_contact"] and not defended_goal) if control_goal and summary["completed"] else None
    return {"control": absent["case"]["name"], "launch_identical": all(summary["launch"][k] == absent["launch"][k] for k in ["position", "velocity", "angular_velocity"]),
            "maximum_position_deflection_m": deflection, "maximum_velocity_difference_m_s": velocity_change,
            "trajectory_changed": bool(deflection >= .02 or velocity_change >= .1),
            "control_goal": absent["goal"], "control_defended_goal": control_goal, "successfully_blocked": blocked,
            "observation_duration_s": float(elapsed[-1]),
            "reason": "finite observation; physical contact and matched absent right Goal required; early fall is censored"}


def assess_experiment(summary, trajectory=None):
    if trajectory is not None:
        ids = [c["geom_id"] for c in summary["contacts"] if c["kind"] == "robot"]
        contacts = trajectory["contacts"]
        robot_contacts = contacts[np.isin(contacts[:, 2], ids)]
        summary["maximum_robot_penetration_m"] = float(max(0, -robot_contacts[:, 14].min())) if len(robot_contacts) else 0.
    event = summary["exit_event"]
    summary["defended_goal"] = bool(event["outcome"] == "Goal" and event["goal"] == "right") if event else (False if summary["completed"] else None)
    summary["momentum_precision_passed"] = summary["maximum_momentum_residual_ns"] <= 1e-6
    # Finite Newton solve residual is measured separately from invalid state or an incomplete experiment.
    summary["experiment_valid"] = bool(summary["launch"] is not None and summary["numerical_error"] is None
                                        and (summary["completed"] or summary["fall_events"] or summary["policy_stop"]))


def write_report(report, output):
    rows = ["# 阶段 E 足球碰撞与抗扰验证", "",
            f"{len(report['results'])} 个工况、{len(report['results'])*report['repeats']} 次执行；重复一致：{report['all_repeats_identical']}；"
            f"数值异常：{report['numerical_error_count']}；自由机器人跌倒工况：{report['fall_case_count']}。", "",
            "1ms物理、50Hz官方策略、每物理步PD；球半径0.11m、质量0.43kg。速度为实验选点，不代表FIFA规定。",
            "发射前静止2秒；横移组再执行2秒vy=+0.2，并在发射后1秒切换零指令。发射后观察6秒；跌倒时保存并提前停止。",
            "固定模式为原零关节姿态的刚体隔离，不能用于同姿态移除控制器的因果比较。无机器人组使用对应自由组相同初始球位置、速度、旋转。", "",
            "| 工况 | 有效接触 | 轨迹改变 | 维持/恢复平衡 | 跌倒 | 任一门/右门进球 | 有效阻挡 | 法向冲量Ns | 峰值力N |",
            "|---|---|---|---|---|---|---|---:|---:|"]
    for r in report["results"]:
        if r["case"]["robot_mode"] == "absent":
            continue
        contacts = [c for c in r["contacts"] if c["kind"] == "robot"]
        comp = r.get("comparison", {})
        rows.append(f"| {r['case']['name']} | {r['effective_contact']} | {comp.get('trajectory_changed')} | "
                    f"{r['balance_maintained']}/{r['balance_recovered']} | {bool(r['fall_events'])} | {r['goal']}/{r['defended_goal']} | "
                    f"{comp.get('successfully_blocked')} | {sum(c['normal_impulse_ns'] for c in contacts):.3f} | "
                    f"{max((c['peak_force_n'] for c in contacts), default=0):.1f} |")
    rows += ["", "None为不适用/观察被提前截断，不能当作成功阻挡。有效阻挡仅为6秒观察窗口中，相同发射的absent对照进球，而有机器人工况发生接触且右门未进球。左门进球仍由原判定器记录，不计为右门失守。",
             "力从每次求解后的mj_contactForce读取，接触区间是该物理步开始到结束；ΣFdt积分为冲量。",
             f"每步球动量变化与全部球接触力+重力独立核对；严格1e-6Ns精度门槛在{report['momentum_precision_failed_case_count']}个工况中未满足，最大{report['maximum_momentum_residual_ns']:.4g}Ns。独立底层核对确认来自原Newton求解残差，接触力重建及Euler积分相符，参数未调整。详见solver_audit.json。",
             "详细接触部位、时刻、持续时间、速度变化、姿态、恢复和限幅见各summary.json及trajectory.npz。",
             "覆盖和警告来源见docs/collision_audit.md；能力限制见docs/impact_validation.md。", ""]
    (output/"report.md").write_text("\n".join(rows))


def run(output, repeats=2):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    audit_path = ROOT/"reports/collision_audit.json"
    audit = json.loads(audit_path.read_text())
    if not (audit["warning_provenance"]["native_probe"]["verified"] and audit["official_compiled_k1_geometry_identical"] and audit["protected_unchanged"]):
        raise RuntimeError("E0 preconditions not verified; run scripts/audit_collisions.py --reproduce-tests")
    for p, h in audit["protected_sha256_after"].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h:
            raise RuntimeError("E0 source audit no longer matches")
    manifest = verify_sources()
    protected = ["k1_goalkeeper/walk_controller.py", "k1_goalkeeper/scene_simulation.py", "k1_goalkeeper/velocity_validation.py",
                 "k1_goalkeeper/scene.py", "k1_goalkeeper/simulation.py", "k1_goalkeeper/detection.py", "config/simulation.json",
                 "assets/booster_assets/robots/K1/K1_22dof.xml", "models/futsal_goal_3x2_reference.urdf",
                 "reports/walk_validation/validation.json", "reports/walk_validation_1ms/validation.json",
                 "reports/velocity_validation/results.json", "reports/goalkeeper_validation/results.json"]
    protected += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"tests").glob("test_*.py")) if p.name != "test_impact_validation.py"]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    output.mkdir(parents=True, exist_ok=True)
    models = {mode: mujoco.MjModel.from_xml_path(str(build_scene(output/f"scene_{mode}.xml", robot_mode=mode)))
              for mode in ["free", "fixed", "absent"]}
    report = {"stage": "E", "repeats": repeats, "e0_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
              "sources": manifest, "versions": {"mujoco": mujoco.__version__, "python": platform.python_version(),
              "numpy": np.__version__, "torch": torch.__version__}, "protected_sha256": hashes,
              "protocol": [{"group": g, "cases": [c.__dict__ for c in cases]} for g, cases in protocol()],
              "criteria": {"fall_height_m": .3, "fall_tilt_degrees": 60, "maintained_height_m": .45,
                           "maintained_tilt_degrees": 10, "momentum_residual_tolerance_ns": 1e-6,
                           "deflection_m": .02, "velocity_change_m_s": .1, "post_launch_observation_s": 6},
              "results": [], "skipped": []}

    def execute(case, launch=None):
        path = output/case.name
        first, trace = run_impact(models[case.robot_mode], case, output=path, launch=launch)
        repeat_hashes = [first["trajectory_sha256"]]
        for _ in range(repeats-1):
            other, _ = run_impact(models[case.robot_mode], case, launch=launch)
            repeat_hashes.append(other["trajectory_sha256"])
        first["reproducibility"] = {"verified": repeats >= 2 and len(set(repeat_hashes)) == 1, "repeat_hashes": repeat_hashes}
        assess_experiment(first, trace)
        report["results"].append(first)
        return first, trace

    def persist():
        (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        for r in report["results"]:
            (output/r["case"]["name"]/"summary.json").write_text(json.dumps(r, indent=2, allow_nan=False)+"\n")

    fronts = {}
    for group, cases in protocol():
        stopped = False
        for case in cases:
            if stopped:
                report["skipped"].append({"case": case.name, "reason": "earlier lower-speed fall/instability; ladder stopped"})
                continue
            summary, trace = execute(case)
            control = replace(case, name=f"absent_{case.name}", robot_mode="absent", command=(0, 0, 0), moving_duration=0)
            absent, reference = execute(control, summary["launch"])
            summary["comparison"] = compare(summary, trace, absent, reference)
            if group == "front":
                fronts[case.speed] = (summary, absent, reference)
            print(f"{case.name}: contact={summary['effective_contact']} maintained={summary['balance_maintained']} "
                  f"recovered={summary['balance_recovered']} fall={bool(summary['fall_events'])} goal={summary['goal']} "
                  f"blocked={summary['comparison']['successfully_blocked']} repeat={summary['reproducibility']['verified']}", flush=True)
            persist()
            if summary["fall_events"] or summary["numerical_error"] or summary["policy_stop"]:
                stopped = True
    for speed in [2, 8, 20]:
        if speed not in fronts:
            continue
        free, absent, reference = fronts[speed]
        summary, trace = execute(ImpactCase(f"fixed_front_{speed}", speed=speed, robot_mode="fixed"), free["launch"])
        summary["comparison"] = compare(summary, trace, absent, reference)
        summary["fixed_pose_note"] = "original zero-joint rigid pose, not the free settled policy pose"
        print(f"fixed_front_{speed}: contact={summary['effective_contact']} goal={summary['goal']}", flush=True)
    report["protected_unchanged"] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    verify_sources()
    report["all_repeats_identical"] = bool(report["results"]) and all(r["reproducibility"]["verified"] for r in report["results"])
    report["numerical_error_count"] = sum(r["numerical_error"] is not None for r in report["results"])
    report["momentum_precision_failed_case_count"] = sum(not r["momentum_precision_passed"] for r in report["results"])
    report["fall_case_count"] = sum(bool(r["fall_events"]) for r in report["results"])
    report["maximum_momentum_residual_ns"] = max(r["maximum_momentum_residual_ns"] for r in report["results"])
    report["experiments_valid"] = bool(report["protected_unchanged"] and report["all_repeats_identical"]
                                       and all(r["experiment_valid"] for r in report["results"]))
    report["assessment_code_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ["k1_goalkeeper/impact_validation.py", "scripts/validate_impacts.py", "scripts/audit_collisions.py", "tests/test_impact_validation.py"]}
    persist()
    write_report(report, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reports/impact_validation")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.repeats)["experiments_valid"] else 1)
