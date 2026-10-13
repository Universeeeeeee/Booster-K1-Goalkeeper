"""E0: audit compiled official K1 geometry and intentional BADQVEL provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from k1_goalkeeper.scene import K1_XML, build_scene
from k1_goalkeeper.scene_simulation import SceneSimulation
from k1_goalkeeper.simulation import Shot, run_shot


def warnings(data):
    return {mujoco.mjtWarning(i).name: int(item.number)
            for i, item in enumerate(data.warning)}


def region(body):
    if body == "trunk":
        return "trunk"
    side = "left" if "left" in body else "right"
    if "head" in body:
        return "head"
    if "shoulder" in body or "elbow" in body:
        return f"{side}_arm"
    if "ankle_roll" in body:
        return f"{side}_foot"
    return f"{side}_leg"


def robot_body(model, body):
    while body > 0:
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) == "trunk":
            return True
        body = int(model.body_parentid[body])
    return False


def inventory(model, data):
    ball = model.geom("ball_collision").id
    ball_body = int(model.geom_bodyid[ball])
    excluded = []
    for signature in model.exclude_signature:
        first, second = divmod(int(signature), 65536)
        excluded.append([first, second])
    explicit_pairs = [{"geom1": int(model.pair_geom1[i]),
                       "geom2": int(model.pair_geom2[i])}
                      for i in range(model.npair)]
    records = []
    for i in range(model.ngeom):
        body = int(model.geom_bodyid[i])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        mask_match = bool((int(model.geom_contype[i]) & int(model.geom_conaffinity[ball]))
                          or (int(model.geom_contype[ball]) & int(model.geom_conaffinity[i])))
        body_excluded = sorted([body, ball_body]) in excluded
        explicit_pair = any({i, ball} == {pair["geom1"], pair["geom2"]}
                            for pair in explicit_pairs)
        independent_body = body != ball_body
        # Ball's free joint makes it neither welded to nor a parent of K1/world.
        eligible = independent_body and (explicit_pair or (mask_match and not body_excluded))
        mesh = int(model.geom_dataid[i]) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH else -1
        records.append({
            "geom_id": i, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
            "body_id": body, "body": body_name,
            "is_k1": robot_body(model, body),
            "region": region(body_name) if robot_body(model, body) else None,
            "type": mujoco.mjtGeom(int(model.geom_type[i])).name,
            "local_position_m": model.geom_pos[i].tolist(),
            "local_quaternion_wxyz": model.geom_quat[i].tolist(),
            "size_m": model.geom_size[i].tolist(),
            "world_position_initialized_m": data.geom_xpos[i].tolist(),
            "world_rotation_initialized": data.geom_xmat[i].reshape(3, 3).tolist(),
            "mesh": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh) if mesh >= 0 else None,
            "contype": int(model.geom_contype[i]), "conaffinity": int(model.geom_conaffinity[i]),
            "rgba": model.geom_rgba[i].tolist(), "group": int(model.geom_group[i]),
            "disabled_visual": bool(model.geom_contype[i] == model.geom_conaffinity[i] == 0),
            "ball_mask_match": mask_match, "ball_body_excluded": body_excluded,
            "ball_explicit_pair": explicit_pair, "ball_collision_eligible": eligible,
        })
    return records, excluded, explicit_pairs


def warning_probe(model):
    """Direct native step validates warning counters independently of run_shot."""
    data = mujoco.MjData(model)
    v = int(model.joint("ball_free").dofadr[0])
    q = int(model.joint("ball_free").qposadr[0])
    data.qpos[q:q + 3] = [0, 0, 1]
    data.qvel[v:v + 3] = [1e100, 0, 0]
    before = {"time": float(data.time), "warning_counts": warnings(data),
              "ball_velocity": data.qvel[v:v + 3].tolist()}
    mujoco.mj_forward(model, data)
    forward = {"time": float(data.time), "warning_counts": warnings(data)}
    mujoco.mj_step(model, data)
    after = {"time": float(data.time), "warning_counts": warnings(data),
             "ball_velocity": data.qvel[v:v + 3].tolist(),
             "finite_qpos": bool(np.isfinite(data.qpos).all()),
             "finite_qvel": bool(np.isfinite(data.qvel).all())}
    exception = None
    try:
        run_shot(model, Shot(position=(0, 0, 1), velocity=(1e100, 0, 0), duration=.01), "absent")
    except RuntimeError as error:
        exception = str(error)
    return {"deliberate_input": True, "robot_mode": "absent", "ball_dof_address": v,
            "before": before, "after_forward": forward, "after_step": after,
            "run_shot_exception": exception,
            "verified": after["warning_counts"]["mjWARN_BADQVEL"] == 1
                        and exception == "MuJoCo unstable state: mjWARN_BADQVEL"}


def audit(output, reproduce_tests=False):
    evidence = output.parent / "collision_audit"
    evidence.mkdir(parents=True, exist_ok=True)
    protected = [K1_XML, ROOT / "models/futsal_goal_3x2_reference.urdf",
                 ROOT / "assets/booster_deploy/tasks/locomotion/robots/k1/models/k1_walk.pt"]
    before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in protected}
    with tempfile.TemporaryDirectory(prefix="k1-e0-") as directory:
        scene_path = build_scene(Path(directory) / "full_scene.xml")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        runtime = SceneSimulation(model, control_mode="k1_walk")
        records, excluded, pairs = inventory(model, runtime.data)
        initial_counts = warnings(runtime.data)
        ball = model.geom("ball_collision").id
        robot_ball_contact_samples = 0
        for _ in range(2000):
            runtime.step()
            for contact in runtime.data.contact:
                if ball in (contact.geom1, contact.geom2):
                    other = int(contact.geom2 if contact.geom1 == ball else contact.geom1)
                    robot_ball_contact_samples += int(robot_body(model, int(model.geom_bodyid[other])))
        ball_contacts = sum(ball in [int(c.geom1), int(c.geom2)] for c in runtime.data.contact)
        quiet = {"command": [0, 0, 0], "duration_s": 2, "physics_steps": runtime.physics_steps,
                 "policy_steps": runtime.policy_steps, "final_time_s": float(runtime.data.time),
                 "initial_warning_counts": initial_counts, "final_warning_counts": warnings(runtime.data),
                 "finite_qpos": bool(np.isfinite(runtime.data.qpos).all()),
                 "finite_qvel": bool(np.isfinite(runtime.data.qvel).all()),
                 "robot_ball_contact_samples": robot_ball_contact_samples,
                 "final_ball_contact_count": ball_contacts,
                 "note": "No ball launch or robot impact; stationary ball may contact ground."}
        absent = mujoco.MjModel.from_xml_path(str(build_scene(Path(directory) / "absent.xml", "absent")))
        probe = warning_probe(absent)
        official = mujoco.MjModel.from_xml_path(str(K1_XML))
        official_geoms = [i for i in range(official.ngeom) if robot_body(official, int(official.geom_bodyid[i]))]
        source_geoms = list(ET.parse(K1_XML).getroot().find("worldbody/body[@name='trunk']").iter("geom"))
        k1_records = [record for record in records if record["is_k1"]]
        official_match = len(official_geoms) == len(k1_records)
        for source, original_id, record in zip(source_geoms, official_geoms, k1_records):
            record["source_xml_geom_attributes"] = source.attrib
            official_match &= all(np.array_equal(getattr(official, key)[original_id], getattr(model, key)[record["geom_id"]])
                                  for key in ("geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size", "geom_contype", "geom_conaffinity"))
    coverage = {}
    for record in records:
        if record["is_k1"]:
            item = coverage.setdefault(record["region"], {"eligible_geoms": [], "disabled_visual_geoms": [], "visual_only_bodies": []})
            item["eligible_geoms" if record["ball_collision_eligible"] else "disabled_visual_geoms"].append(record["name"])
    for name, item in coverage.items():
        region_records = [record for record in records if record["region"] == name]
        item["visual_only_bodies"] = sorted({record["body"] for record in region_records}
                                           - {record["body"] for record in region_records if record["ball_collision_eligible"]})
    test_evidence = {}
    for name, args in [
        ("isolated_warning_test", ["tests/test_goal_detection.py::test_mujoco_silent_unstable_reset_is_error"]),
        ("baseline_without_intentional_warning", ["-k", "not test_mujoco_silent_unstable_reset_is_error"]),
    ]:
        path = evidence / f"{name}.txt"
        command = [sys.executable, "-m", "pytest", "-q", *args]
        returncode = None
        if reproduce_tests:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            path.write_text(result.stdout + result.stderr)
            returncode = result.returncode
        contents = path.read_text() if path.exists() else ""
        passed = re.search(r"(\d+) passed", contents)
        deselected = re.search(r"(\d+) deselected", contents)
        test_evidence[name] = {"command": command, "returncode_on_this_audit_run": returncode,
                              "log": str(path.relative_to(ROOT)),
                              "passed": int(passed.group(1)) if passed else None,
                              "deselected": int(deselected.group(1)) if deselected else 0,
                              "native_warning_line_count": contents.count("WARNING:"),
                              "output": contents or None}
    after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in protected}
    report = {
        "stage": "E0", "mujoco_version": mujoco.__version__, "model": "official K1 full field scene",
        "generation": ".venv/bin/python scripts/audit_collisions.py --reproduce-tests",
        "compiled_via_filepath": True, "temporary_scene_deleted": True,
        "protected_sha256_before": before, "protected_sha256_after": after, "protected_unchanged": before == after,
        "official_compiled_k1_geometry_identical": bool(official_match),
        "ball_mask_rule": "(geom.contype & ball.conaffinity) != 0 OR (ball.contype & geom.conaffinity) != 0",
        "ball_body_independent": True, "excluded_body_pairs": excluded, "explicit_geom_pairs": pairs,
        "ngeom": len(records), "k1_ngeom": len(k1_records),
        "k1_ball_collision_eligible_count": sum(record["ball_collision_eligible"] for record in k1_records),
        "k1_disabled_visual_count": sum(record["disabled_visual"] for record in k1_records),
        "geometry_units": "m; box size=half-extents; cylinder size=[radius, half-length, unused]; mesh size is compiled bound, not a primitive",
        "mesh_contact_note": "MuJoCo convex mesh collision does not reproduce every visual surface or concavity.",
        "coverage": coverage, "geometries": records, "quiet_baseline": quiet,
        "warning_provenance": {"historical_log": "reports/goalkeeper_validation/tests.txt", "intentional_test": "tests/test_goal_detection.py::test_mujoco_silent_unstable_reset_is_error", "native_probe": probe, "tests": test_evidence},
        "coverage_gaps": ["Shoulder-pitch roots, hip-pitch roots and ankle-pitch crosses have visual-only meshes.",
                          "Trunk and legs use partial box/cylinder proxies, not continuous full visual coverage.",
                          "Head and distal arms use convex mesh collision; visual concavities are not guaranteed."],
        "official_testable_targets": {"trunk": "k1_geom_1", "left_shank": "k1_geom_29", "right_shank": "k1_geom_40", "left_foot": "k1_geom_32", "right_foot": "k1_geom_43", "left_arm": "k1_geom_12", "right_arm": "k1_geom_19", "head": "k1_geom_7"},
        "supplemental_model_required_for_these_targets": False,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/collision_audit.json")
    parser.add_argument("--reproduce-tests", action="store_true")
    args = parser.parse_args()
    result = audit(args.output, args.reproduce_tests)
    print(json.dumps({key: result[key] for key in ("k1_ngeom", "k1_ball_collision_eligible_count", "k1_disabled_visual_count", "protected_unchanged", "official_compiled_k1_geometry_identical")}, indent=2))
