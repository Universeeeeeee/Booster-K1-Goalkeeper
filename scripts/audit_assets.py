#!/usr/bin/env python3
"""Audit consumed source assets and the compiled MuJoCo model."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets/booster_assets"
SOURCE = ASSETS / "robots/K1/K1_22dof.xml"
COMMIT = "12a97516f57469078707afb587a5f84c1cb110d1"


def name(model, kind, index):
    return mujoco.mj_id2name(model, kind, int(index))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def audit():
    source_xml = ET.parse(SOURCE).getroot()
    meshes = [SOURCE.parent / "meshes" / node.attrib["file"]
              for node in source_xml.findall("asset/mesh")]
    consumed = [SOURCE, SOURCE.with_suffix(".urdf"), *meshes]
    hashes = []
    for path in consumed:
        relative = path.relative_to(ASSETS).as_posix()
        upstream = subprocess.check_output(
            ["git", "-C", str(ASSETS), "show", f"{COMMIT}:{relative}"])
        digest = sha256(path.read_bytes())
        hashes.append({"path": path.relative_to(ROOT).as_posix(),
                       "sha256": digest, "matches_pinned_commit": digest == sha256(upstream)})
    goal_path = ROOT / "models/futsal_goal_3x2_reference.urdf"
    hashes.append({"path": goal_path.relative_to(ROOT).as_posix(), "sha256": sha256(goal_path.read_bytes()),
                   "source": "user-supplied original URDF"})

    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    motors = []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        motors.append({"index": actuator,
                       "name": name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator),
                       "joint": name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                       "qpos_address": int(model.jnt_qposadr[joint]),
                       "dof_address": int(model.jnt_dofadr[joint]),
                       "joint_range_rad": model.jnt_range[joint].tolist(),
                       "force_range_nm": model.actuator_forcerange[actuator].tolist(),
                       "force_limited": bool(model.actuator_forcelimited[actuator]),
                       "gear": model.actuator_gear[actuator].tolist()})
    joint_order = [name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                   for joint in range(model.njnt)
                   if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE]
    collisions = []
    missing = []
    for body in range(1, model.nbody):
        geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
        active = [geom for geom in geoms if model.geom_contype[geom] or model.geom_conaffinity[geom]]
        if model.body_geomnum[body] and not active:
            missing.append(name(model, mujoco.mjtObj.mjOBJ_BODY, body))
        for geom in active:
            collisions.append({"geom_id": int(geom),
                               "body": name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                               "type": mujoco.mjtGeom(int(model.geom_type[geom])).name,
                               "local_position": model.geom_pos[geom].tolist(),
                               "size": model.geom_size[geom].tolist(),
                               "contype": int(model.geom_contype[geom]),
                               "conaffinity": int(model.geom_conaffinity[geom])})
    soles = {}
    for side in ("left", "right"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        foot = next(geom for geom in range(model.body_geomadr[body],
                                          model.body_geomadr[body] + model.body_geomnum[body])
                    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX)
        extent = np.abs(data.geom_xmat[foot].reshape(3, 3)[2]) @ model.geom_size[foot]
        soles[side] = float(data.geom_xpos[foot, 2] - extent)
    optical = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_booster_stereo_rgb_link")
    front = data.xmat[optical].reshape(3, 3)[:, 2].tolist()
    motion_source = subprocess.check_output(
        ["git", "-C", str(ASSETS), "show", f"{COMMIT}:src/booster_assets/motions.py"], text=True)
    motion_tree = ast.parse(motion_source)
    motion_names = next(ast.literal_eval(node.value) for node in motion_tree.body
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "K1_JOINT_NAMES"
                                for target in node.targets))
    tracked_files = subprocess.check_output(
        ["git", "-C", str(ASSETS), "ls-tree", "-r", "--name-only", COMMIT], text=True).splitlines()
    controller_candidates = [path for path in tracked_files
                             if any(word in path.lower() for word in ("controller", "policy", "standing", "gain"))
                             or Path(path).suffix.lower() in (".onnx", ".pt", ".pth")]
    goal = mujoco.MjModel.from_xml_path(str(goal_path))
    goal_data = mujoco.MjData(goal)
    mujoco.mj_forward(goal, goal_data)
    goal_geoms = {name(goal, mujoco.mjtObj.mjOBJ_GEOM, geom):
                  {"position": goal_data.geom_xpos[geom].tolist(), "size": goal.geom_size[geom].tolist()}
                  for geom in range(goal.ngeom)}
    left, right, bar = [goal_geoms[key] for key in
                        ("left_post_collision", "right_post_collision", "crossbar_collision")]
    clear_width = right["position"][0] - left["position"][0] - right["size"][0] - left["size"][0]
    clear_height = bar["position"][2] - bar["size"][0]

    # This is a no-controller baseline, not a standing-controller acceptance test.
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    data.qpos[2] -= min(soles.values()) - 0.002
    mujoco.mj_forward(model, data)
    start_height = float(data.xpos[base, 2])
    minimum_height = start_height
    for _ in range(round(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        minimum_height = min(minimum_height, float(data.xpos[base, 2]))
    mujoco.mj_forward(model, data)
    baseline = {"duration_s": float(data.time), "control": "zero motor torque; no controller",
                "initial_base_z_m": start_height, "final_base_z_m": float(data.xpos[base, 2]),
                "minimum_base_z_m": minimum_height,
                "final_trunk_up_world_z": float(data.xmat[base].reshape(3, 3)[2, 2]),
                "finite_state": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
                "dynamic_balance_verified": False}
    report = {
        "versions": {"python": sys.version.split()[0], "mujoco": mujoco.__version__, "numpy": np.__version__},
        "source": {"repository": "https://github.com/BoosterRobotics/booster_assets",
                   "pinned_commit": COMMIT,
                   "checkout_commit": subprocess.check_output(
                       ["git", "-C", str(ASSETS), "rev-parse", "HEAD"], text=True).strip()},
        "asset_hashes": hashes,
        "robot": {"compiled": True, "nq": model.nq, "nv": model.nv, "nu": model.nu,
                  "referenced_mesh_count": len(meshes), "compiled_mesh_count": model.nmesh,
                  "mass_kg": float(model.body_mass.sum()),
                  "options": {"timestep_s": float(model.opt.timestep),
                              "solver": mujoco.mjtSolver(int(model.opt.solver)).name,
                              "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name},
                  "actuators": motors, "motor_order_matches_joint_order":
                  [motor["joint"] for motor in motors] == joint_order,
                  "collision_geometries": collisions, "visual_bodies_without_direct_collision": missing,
                  "collision_limitations": "Partial proxy coverage; active mesh geoms use convex hulls. Visual surface coverage is not guaranteed.",
                  "source_zero_pose_sole_z_m": soles,
                  "stereo_optical_plus_z_in_robot_frame": front,
                  "suggested_base_z_for_2mm_sole_clearance_m": 1.0 - min(soles.values()) + 0.002},
        "goal": {"compiled": True, "nq": goal.nq, "nv": goal.nv,
                 "default_import_collision_geometries": goal_geoms,
                 "clear_width_m": clear_width, "clear_height_m": clear_height,
                 "note": "Default URDF importer discards visuals. Scene conversion must retain visuals explicitly; rear support has no collision and no net is modeled."},
        "control": {"standing_controller_found": False, "pd_gains_found": False,
                    "policy_weights_found": False, "filename_candidates": controller_candidates,
                    "inspection_scope": "Pinned repository tracked filenames, model actuator definitions, Python motion metadata, and repository README.",
                    "motion_csv_files": [path for path in tracked_files if path.startswith("motions/K1/")],
                    "motion_name_mismatches": [{"index": index, "csv_metadata": csv_name, "mjcf": joint_order[index]}
                                               for index, csv_name in enumerate(motion_names)
                                               if csv_name != joint_order[index]],
                    "interface": "MjData.ctrl follows the 22 motor order; motors apply torque (gear 1), with no built-in PD or balance policy.",
                    "blocking_item": "Dynamic standing and goalkeeper behavior require an independently validated controller."},
        "no_controller_baseline": baseline,
    }
    from k1_goalkeeper.scene import build_scene

    with tempfile.TemporaryDirectory() as directory:
        scene_path = build_scene(Path(directory) / "scene.xml", robot_mode="free")
        scene = mujoco.MjModel.from_xml_path(str(scene_path))
    scene_data = mujoco.MjData(scene)
    mujoco.mj_forward(scene, scene_data)
    ball_geom = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_GEOM, "ball_collision")
    ball_body = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, "ball")
    robot_base = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    scene_optical = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, "head_booster_stereo_rgb_link")
    report["scene"] = {
        "robot_mode": "free", "compiled": True, "nq": scene.nq, "nv": scene.nv, "nu": scene.nu,
        "actuator_names_match_original":
        [name(scene, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(scene.nu)]
        == [motor["name"] for motor in motors],
        "actuator_force_ranges_match_original": bool(np.array_equal(scene.actuator_forcerange, model.actuator_forcerange)),
        "robot_base_position": scene_data.xpos[robot_base].tolist(),
        "stereo_optical_forward_world": scene_data.xmat[scene_optical].reshape(3, 3)[:, 2].tolist(),
        "ball_radius_m": float(scene.geom_size[ball_geom, 0]),
        "ball_mass_kg": float(scene.body_mass[ball_body]),
        "ball_inertia_kg_m2": scene.body_inertia[ball_body].tolist(),
        "initial_contact_count": scene_data.ncon,
        "initial_minimum_contact_distance_m": min((float(scene_data.contact[index].dist)
                                                   for index in range(scene_data.ncon)), default=None),
        "goals": {side: {"joint_count": int(scene.body_jntnum[mujoco.mj_name2id(
            scene, mujoco.mjtObj.mjOBJ_BODY, f"goal_{side}")]),
            "collision_count": sum(1 for geom in range(scene.ngeom)
                                   if (name(scene, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(f"goal_{side}_")
                                   and (scene.geom_contype[geom] or scene.geom_conaffinity[geom]))}
                  for side in ("left", "right")},
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/asset_audit.json")
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Asset audit written to {args.output}")
    print(f"K1: {report['robot']['nu']} motors; {report['robot']['compiled_mesh_count']} meshes; "
          f"{len(report['robot']['collision_geometries'])} robot collision geoms.")
    print("Standing controller absent; dynamic balance has not been verified.")


if __name__ == "__main__":
    main()
