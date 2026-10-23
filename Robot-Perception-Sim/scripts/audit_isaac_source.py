#!/usr/bin/env python3
"""Freeze the real MuJoCo source contract for a later Isaac migration (no steps)."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from k1_goalkeeper.scene import build_scene, K1_XML, GOAL_URDF
from scripts.audit_assets import name, COMMIT
from scripts.audit_defense_geometry import official_config, K1_CONFIG, WALK_CONFIG


def policy_names():
    tree = ast.parse(WALK_CONFIG.read_text())
    return next(ast.literal_eval(item.value) for node in ast.walk(tree)
                if isinstance(node, ast.Call) for item in node.keywords
                if item.arg == "policy_joint_names")


def audit(scene):
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config, policy = official_config(), policy_names()
    joint_names = config["joint_names"]
    if len(joint_names) != 22 or len(set(joint_names)) != 22:
        raise ValueError("Expected 22 unique configured K1 joints")
    if len(policy) != 20 or len(set(policy)) != 20 or not set(policy) <= set(joint_names):
        raise ValueError("Expected 20 unique policy joints in the 22-joint contract")

    def ident(kind, value):
        index = mujoco.mj_name2id(model, kind, value)
        if index < 0:
            raise ValueError(f"Missing source object: {value}")
        return index

    joints = []
    for index, joint_name in enumerate(joint_names):
        joint = ident(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        motor = ident(mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected hinge: {joint_name}")
        if model.actuator_trnid[motor, 0] != joint:
            raise ValueError(f"Motor does not target named joint: {joint_name}")
        joints.append({"robot_index": index, "name": joint_name,
                       "joint_id": joint, "actuator_id": motor,
                       "qpos_address": int(model.jnt_qposadr[joint]),
                       "dof_address": int(model.jnt_dofadr[joint]),
                       "body": name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[joint]),
                       "axis_body": model.jnt_axis[joint].tolist(),
                       "anchor_body_m": model.jnt_pos[joint].tolist(),
                       "axis_world_initial": data.xaxis[joint].tolist(),
                       "limited": bool(model.jnt_limited[joint]),
                       "range_rad": model.jnt_range[joint].tolist(),
                       "scene_initial_q_rad": float(data.qpos[model.jnt_qposadr[joint]]),
                       "walk_home_q_rad": config["default_joint_pos"][index],
                       "walk_effort_limit_nm": config["effort_limit"][index],
                       "motor_gear": model.actuator_gear[motor].tolist(),
                       "motor_control_limited": bool(model.actuator_ctrllimited[motor]),
                       "motor_control_range": model.actuator_ctrlrange[motor].tolist(),
                       "motor_force_limited": bool(model.actuator_forcelimited[motor]),
                       "motor_force_range": model.actuator_forcerange[motor].tolist()})
    base = ident(mujoco.mjtObj.mjOBJ_BODY, "trunk")
    free = ident(mujoco.mjtObj.mjOBJ_JOINT, "world_joint")
    if model.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("K1 source must retain its free base")
    bodies = [{"id": body, "name": name(model, mujoco.mjtObj.mjOBJ_BODY, body),
               "parent_id": int(model.body_parentid[body]),
               "mass_kg": float(model.body_mass[body]),
               "inertia_principal_kg_m2": model.body_inertia[body].tolist(),
               "inertial_position_body_m": model.body_ipos[body].tolist(),
               "inertial_quaternion_body_wxyz": model.body_iquat[body].tolist(),
               "position_parent_m": model.body_pos[body].tolist(),
               "quaternion_parent_wxyz": model.body_quat[body].tolist()}
              for body in range(model.nbody)]
    geoms = [{"id": geom, "name": name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
              "body_id": int(model.geom_bodyid[geom]),
              "type": mujoco.mjtGeom(int(model.geom_type[geom])).name,
              "size_m": model.geom_size[geom].tolist(),
              "position_body_m": model.geom_pos[geom].tolist(),
              "quaternion_body_wxyz": model.geom_quat[geom].tolist(),
              "contype": int(model.geom_contype[geom]),
              "conaffinity": int(model.geom_conaffinity[geom]),
              "group": int(model.geom_group[geom]),
              "condim": int(model.geom_condim[geom]),
              "priority": int(model.geom_priority[geom]),
              "friction": model.geom_friction[geom].tolist(),
              "solref": model.geom_solref[geom].tolist(),
              "solimp": model.geom_solimp[geom].tolist()}
             for geom in range(model.ngeom)]
    goals = {}
    for side in ("left", "right"):
        ids = [ident(mujoco.mjtObj.mjOBJ_GEOM, f"goal_{side}_{part}_collision")
               for part in ("left_post", "right_post", "crossbar")]
        left, right, bar = ids
        goals[side] = {
            "clear_width_m": float(np.linalg.norm(data.geom_xpos[right] - data.geom_xpos[left])
                                   - model.geom_size[left, 0] - model.geom_size[right, 0]),
            "clear_height_m": float(data.geom_xpos[bar, 2] - model.geom_size[bar, 0]),
            "frame_geom_world_positions_m": {geoms[g]["name"]: data.geom_xpos[g].tolist() for g in ids},
            "net_modeled": False}
    ball = ident(mujoco.mjtObj.mjOBJ_BODY, "ball")
    sphere = ident(mujoco.mjtObj.mjOBJ_GEOM, "ball_collision")
    sources = {scene, K1_XML, K1_XML.with_suffix(".urdf"), GOAL_URDF,
               ROOT / "config/simulation.json", ROOT / "k1_goalkeeper/scene.py",
               ROOT / "k1_goalkeeper/walk_controller.py", K1_CONFIG, WALK_CONFIG,
               ROOT / "scripts/audit_assets.py", ROOT / "scripts/audit_defense_geometry.py",
               Path(__file__).resolve()}
    # Include source dependencies and actual scene asset bytes, not just XML names.
    for xml in (K1_XML, scene):
        tree = ET.parse(xml).getroot()
        compiler = tree.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        for node in tree.findall("asset/mesh"):
            if node.get("file"):
                sources.add((xml.parent / meshdir / node.get("file")).resolve())
    hashes = [{"path": str(path.relative_to(ROOT)),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in sorted(sources)]
    return {"schema_version": 1, "stage": "G1 MuJoCo source contract only",
            "usd_migration_complete": False, "physics_parity_verified": False,
            "physics_steps": 0,
            "versions": {"mujoco": mujoco.__version__, "python": sys.version.split()[0]},
            "source": {"scene": str(scene.relative_to(ROOT)), "asset_pinned_commit": COMMIT,
                       "asset_checkout_commit": subprocess.check_output(
                           ["git", "-C", str(K1_XML.parents[2]), "rev-parse", "HEAD"], text=True).strip(),
                       "hashes": hashes},
            "world": {"units": "m, kg, s, rad", "handedness": "right",
                      "up_axis": "+Z", "field_length_axis": "X", "field_width_axis": "Y",
                      "quaternion_order": "wxyz", "gravity_m_s2": model.opt.gravity.tolist(),
                      "timestep_s": float(model.opt.timestep)},
            "robot": {"free_base": True, "base_position_world_m": data.xpos[base].tolist(),
                      "base_quaternion_world_wxyz": data.xquat[base].tolist(),
                      "base_velocity_initial": data.qvel[model.jnt_dofadr[free]:model.jnt_dofadr[free]+6].tolist(),
                      "joints": joints, "robot_joint_order": joint_names,
                      "policy_action_order": policy,
                      "policy_to_robot_indices": [joint_names.index(value) for value in policy],
                      "unactuated_by_policy": [value for value in joint_names if value not in policy]},
            "bodies": bodies, "geoms": geoms,
            "collision": {"exclude_signature": model.exclude_signature.tolist(),
                          "explicit_pair_geom_ids": np.column_stack((model.pair_geom1, model.pair_geom2)).tolist(),
                          "filter_rule": "MuJoCo bitmask: (contype1 & conaffinity2) or (contype2 & conaffinity1); body/parent exclusions also apply",
                          "initial_contacts": [{"geom_ids": data.contact[i].geom.tolist(),
                                                "distance_m": float(data.contact[i].dist)}
                                               for i in range(data.ncon)]},
            "ball": {"radius_m": float(model.geom_size[sphere, 0]),
                     "mass_kg": float(model.body_mass[ball]),
                     "inertia_principal_kg_m2": model.body_inertia[ball].tolist(),
                     "position_world_m": data.xpos[ball].tolist()}, "goals": goals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, help="Read an existing complete source scene")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/g1" /
                        datetime.now(timezone.utc).strftime("source_contract_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(PROJECT / "results"):
        parser.error("output-dir must be a fresh directory under Robot-Perception-Sim/results")
    output.mkdir(parents=True, exist_ok=False)
    scene = args.scene.resolve() if args.scene else build_scene(output / "source_scene.xml", robot_mode="free")
    report = audit(scene)
    path = output / "asset_manifest.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(path)
    print(f"22 joints; 20 policy actions; {len(report['bodies'])} bodies; MuJoCo {mujoco.__version__}; no physics steps")


if __name__ == "__main__":
    main()
