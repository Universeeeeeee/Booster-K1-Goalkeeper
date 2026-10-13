"""H0: read-only native K1 geometry audit; no controller or simulation steps."""

import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K1_XML = ROOT / "assets/booster_assets/robots/K1/K1_22dof.xml"
K1_CONFIG = ROOT / "assets/booster_deploy/booster_deploy/robots/k1.py"
WALK_CONFIG = ROOT / "assets/booster_deploy/tasks/locomotion/robots/k1/__init__.py"
E0_PATH = ROOT / "reports/collision_audit.json"
GRID_M = .0025
REFERENCE_BASE_HEIGHT_M = .55369


def official_config():
    tree = ast.parse(K1_CONFIG.read_text())
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "RobotCfg")
    config = {item.arg: ast.literal_eval(item.value) for item in call.keywords
              if item.arg in {"joint_names", "default_joint_pos", "effort_limit"}}
    walk_tree = ast.parse(WALK_CONFIG.read_text())
    override = next(node for node in ast.walk(walk_tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "replace")
    for item in override.keywords:
        if item.arg in {"default_joint_pos", "effort_limit"}:
            config[item.arg] = ast.literal_eval(item.value)
    return config


def sha256(paths):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def region(body):
    if body == "trunk":
        return "trunk"
    if "shoulder" in body or "elbow" in body:
        return "left_arm" if "left" in body else "right_arm"
    return None


def hull(points):
    """Monotone-chain 2-D convex hull, counterclockwise."""
    points = sorted(set(map(tuple, points)))
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower, upper = [], []
    for chain, order in ((lower, points), (upper, reversed(points))):
        for point in order:
            while len(chain) > 1 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def geom_points(model, data, geom):
    kind = int(model.geom_type[geom])
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        points = np.asarray(list(itertools.product((-1, 1), repeat=3))) * size
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        angle = np.arange(256) * (2 * np.pi / 256)
        ring = np.column_stack((size[0] * np.cos(angle), size[0] * np.sin(angle)))
        points = np.vstack([np.column_stack((ring, np.full(256, z)))
                            for z in (-size[1], size[1])])
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        start = int(model.mesh_vertadr[mesh])
        points = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh])]
    else:
        raise ValueError(f"Unsupported native upper-body geometry {kind}")
    return points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def projection(model, data, records):
    polygons = {record["official_geom_id"]:
                hull(geom_points(model, data, record["official_geom_id"])[:, 1:3]
                     - data.xpos[model.body("trunk").id, 1:3])
                for record in records if record["native_collision_enabled"]}
    # Common bounds prevent pose-dependent raster alignment from biasing deltas.
    y, z = np.meshgrid(np.arange(-.75 + GRID_M / 2, .75, GRID_M),
                       np.arange(-.30 + GRID_M / 2, .50, GRID_M))
    grid = np.column_stack((y.ravel(), z.ravel()))
    masks = {name: np.zeros(len(grid), dtype=bool)
             for name in ("trunk", "left_arm", "right_arm")}
    bounds = {}
    for record in records:
        if not record["native_collision_enabled"]:
            continue
        polygon = polygons[record["official_geom_id"]]
        inside = np.ones(len(grid), dtype=bool)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            inside &= ((b[0] - a[0]) * (grid[:, 1] - a[1])
                       - (b[1] - a[1]) * (grid[:, 0] - a[0])) >= -1e-12
        masks[record["region"]] |= inside
    for name in masks:
        points = np.vstack([polygons[r["official_geom_id"]] for r in records
                            if r["region"] == name and r["native_collision_enabled"]])
        bounds[name] = {"y_min": float(points[:, 0].min()), "y_max": float(points[:, 0].max()),
                        "z_min": float(points[:, 1].min()), "z_max": float(points[:, 1].max())}
    union = masks["trunk"] | masks["left_arm"] | masks["right_arm"]
    return {"region_projected_area_m2": {name: float(mask.sum() * GRID_M ** 2)
                                         for name, mask in masks.items()},
            "upper_body_union_projected_area_m2": float(union.sum() * GRID_M ** 2),
            "arms_outside_trunk_projected_area_m2": float(
                ((masks["left_arm"] | masks["right_arm"]) & ~masks["trunk"]).sum() * GRID_M ** 2),
            "region_projection_bounds_trunk_relative_m": bounds}


def audit(output):
    source = ET.parse(K1_XML).getroot()
    protected = [K1_XML, K1_CONFIG, WALK_CONFIG, E0_PATH, ROOT / "k1_goalkeeper/walk_controller.py",
                 ROOT / "assets/booster_deploy/tasks/locomotion/robots/k1/models/k1_walk.pt"]
    protected.extend(K1_XML.parent / "meshes" / mesh.attrib["file"]
                     for mesh in source.findall("asset/mesh"))
    before = sha256(protected)
    config = official_config()
    model = mujoco.MjModel.from_xml_path(str(K1_XML))
    e0 = json.loads(E0_PATH.read_text())
    e0_geoms = [record for record in e0["geometries"] if record["is_k1"]]
    official_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0]
    mapping = dict(zip(official_geoms, e0_geoms))
    records = []
    for geom in official_geoms:
        body = model.body(int(model.geom_bodyid[geom])).name
        if region(body) is None:
            continue
        previous = mapping[geom]
        matches = (previous["body"] == body
                   and previous["type"] == mujoco.mjtGeom(int(model.geom_type[geom])).name
                   and previous["contype"] == int(model.geom_contype[geom])
                   and previous["conaffinity"] == int(model.geom_conaffinity[geom])
                   and all(np.array_equal(previous[key], getattr(model, field)[geom])
                           for key, field in (("local_position_m", "geom_pos"),
                                              ("local_quaternion_wxyz", "geom_quat"),
                                              ("size_m", "geom_size"))))
        mesh = int(model.geom_dataid[geom]) if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH else -1
        records.append({"official_geom_id": geom, "e0_alias": previous["name"],
                        "body": body, "region": region(body), "type": previous["type"],
                        "mesh": model.mesh(mesh).name if mesh >= 0 else None,
                        "contype": int(model.geom_contype[geom]),
                        "conaffinity": int(model.geom_conaffinity[geom]),
                        "native_collision_enabled": bool(model.geom_contype[geom] or model.geom_conaffinity[geom]),
                        "local_position_m": model.geom_pos[geom].tolist(),
                        "local_quaternion_wxyz": model.geom_quat[geom].tolist(),
                        "size_m": model.geom_size[geom].tolist(), "matches_existing_e0": bool(matches)})
    if len(e0_geoms) != len(official_geoms) or not all(r["matches_existing_e0"] for r in records):
        raise RuntimeError("Official upper-body inventory differs from existing E0 evidence")
    joints = []
    for name, default, effort in zip(config["joint_names"], config["default_joint_pos"], config["effort_limit"]):
        joint = model.joint(name)
        actuator = model.actuator(name)
        joints.append({"name": name, "default_angle_rad": default,
                       "limited": bool(joint.limited[0]), "range_rad": joint.range.tolist(),
                       "axis_local": joint.axis.tolist(),
                       "actuator_force_limited": bool(actuator.forcelimited[0]),
                       "actuator_force_range": actuator.forcerange.tolist(),
                       "actuator_gear": actuator.gear.tolist(), "deploy_effort_limit_nm": effort,
                       "is_arm": "shoulder" in name or "elbow" in name})
    excluded = [list(divmod(int(value), 65536)) for value in model.exclude_signature]
    active = [r for r in records if r["native_collision_enabled"]]
    pairs = []
    parent_filter = not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    for first, second in itertools.combinations(active, 2):
        a, b = first["official_geom_id"], second["official_geom_id"]
        body_a, body_b = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
        reasons = []
        if model.body_weldid[body_a] == model.body_weldid[body_b]:
            reasons.append("same_or_welded_body")
        if parent_filter and (model.body_parentid[body_a] == body_b or model.body_parentid[body_b] == body_a):
            reasons.append("default_parent_child_filter")
        if sorted([body_a, body_b]) in excluded:
            reasons.append("explicit_body_exclude")
        if not ((int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
                or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))):
            reasons.append("mask")
        pairs.append({"geom1": first["e0_alias"], "geom2": second["e0_alias"],
                      "official_ids": [a, b], "filtered_reasons": reasons,
                      "default_collision_eligible": not reasons})
    baseline = model.qpos0.copy()
    baseline[model.joint("world_joint").qposadr[0] + 2] = REFERENCE_BASE_HEIGHT_M
    for joint in joints:
        baseline[model.joint(joint["name"]).qposadr[0]] = joint["default_angle_rad"]
    candidates = [("native_default", {})]
    for amount in (.15, .30):
        for action in ("raise", "extend"):
            for side in ("left", "right", "both"):
                delta = {}
                for arm in (("left", "right") if side == "both" else (side,)):
                    name = f"{arm}_shoulder_roll_joint" if action == "raise" else f"aa{arm}_shoulder_pitch_joint"
                    delta[name] = amount * (1 if arm == "left" else -1) if action == "raise" else -amount
                candidates.append((f"{side}_{action}_{amount:.2f}", delta))
    # A separate computation backend cross-checks native GJK distance queries.
    # This diagnostic copy never supplies dynamics, controls, or report geometry.
    diagnostic = mujoco.MjModel.from_xml_path(str(K1_XML))
    diagnostic.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    poses = []
    for name, delta in candidates:
        data = mujoco.MjData(model)
        diagnostic_data = mujoco.MjData(diagnostic)
        minima, alternative_minima, contacts = {}, {}, []
        for fraction in np.linspace(0, 1, 11):
            data.qpos[:] = baseline
            for joint, angle in delta.items():
                data.qpos[model.joint(joint).qposadr[0]] += fraction * angle
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_collision(model, data)
            diagnostic_data.qpos[:] = data.qpos
            mujoco.mj_kinematics(diagnostic, diagnostic_data)
            mujoco.mj_comPos(diagnostic, diagnostic_data)
            for pair in pairs:
                if pair["default_collision_eligible"]:
                    distance = float(mujoco.mj_geomDistance(model, data, *pair["official_ids"], 2., None))
                    key = f'{pair["geom1"]}:{pair["geom2"]}'
                    minima[key] = min(minima.get(key, 2.), distance)
                    alternative = float(mujoco.mj_geomDistance(
                        diagnostic, diagnostic_data, *pair["official_ids"], 2., None))
                    alternative_minima[key] = min(alternative_minima.get(key, 2.), alternative)
            contacts.append({"interpolation_fraction": float(fraction),
                             "robot_self_contacts": [{"geom_ids": [int(c.geom1), int(c.geom2)],
                                                       "distance_m": float(c.dist)}
                                                      for c in data.contact
                                                      if model.geom_bodyid[c.geom1] != 0
                                                      and model.geom_bodyid[c.geom2] != 0]})
        margins = {j["name"]: float(min(data.qpos[model.joint(j["name"]).qposadr[0]] - j["range_rad"][0],
                                         j["range_rad"][1] - data.qpos[model.joint(j["name"]).qposadr[0]]))
                   for j in joints if j["is_arm"]}
        coverage = projection(model, data, records)
        poses.append({"name": name, "joint_delta_from_default_rad": delta,
                      "arm_joint_angles_rad": {j["name"]: float(data.qpos[model.joint(j["name"]).qposadr[0]])
                                                for j in joints if j["is_arm"]},
                      "arm_limit_margin_rad": margins, "all_arm_angles_within_limits": min(margins.values()) >= 0,
                      "static_samples": contacts, "eligible_pair_min_distance_over_samples_m": minima,
                      "libccd_diagnostic_pair_min_distance_over_samples_m": alternative_minima,
                      "distance_backend_disagreement_over_1mm_pairs": [key for key in minima
                                                                      if abs(minima[key] - alternative_minima[key]) > .001],
                      "static_self_contact_count": sum(len(row["robot_self_contacts"]) for row in contacts),
                      "kinematic_time_s": float(data.time), "coverage": coverage,
                      "distal_elbow_yaw_body_origin_trunk_relative_m": {
                          side: (data.xpos[model.body(f"{side}_elbow_yaw_link").id]
                                 - data.xpos[model.body("trunk").id]).tolist()
                          for side in ("left", "right")}})
    reference = poses[0]["coverage"]["upper_body_union_projected_area_m2"]
    for pose in poses:
        pose["coverage"]["union_delta_from_default_m2"] = pose["coverage"]["upper_body_union_projected_area_m2"] - reference
        pose["native_arm_max_height_at_reference_base_m"] = REFERENCE_BASE_HEIGHT_M + max(
            pose["coverage"]["region_projection_bounds_trunk_relative_m"][side]["z_max"]
            for side in ("left_arm", "right_arm"))
    reach = []
    rest = mujoco.MjData(model)
    mujoco.mj_kinematics(model, rest)
    for record in active:
        if record["region"] == "trunk":
            continue
        geom = record["official_geom_id"]
        body = int(model.geom_bodyid[geom])
        offsets, current = [], body
        while model.body(current).name != "trunk":
            offsets.append({"body": model.body(current).name,
                            "offset_length_m": float(np.linalg.norm(model.body_pos[current]))})
            current = int(model.body_parentid[current])
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape_radius = float(np.linalg.norm(model.geom_pos[geom])
                                 + np.linalg.norm(model.geom_size[geom, :2]))
        else:
            shape_radius = float(np.linalg.norm(geom_points(model, rest, geom) - rest.xpos[body], axis=1).max())
        upper_radius = sum(item["offset_length_m"] for item in offsets) + shape_radius + 1e-6
        reach.append({"geom": record["e0_alias"], "body_offsets_to_trunk": list(reversed(offsets)),
                      "native_shape_radius_about_body_origin_m": shape_radius,
                      "conservative_radius_from_trunk_origin_m": upper_radius})
    reach_radius = max(row["conservative_radius_from_trunk_origin_m"] for row in reach)
    after = sha256(protected)
    report = {"stage": "H0", "mujoco_version": mujoco.__version__,
              "generation": ".venv/bin/python scripts/audit_defense_geometry.py",
              "model": str(K1_XML.relative_to(ROOT)), "compiled_directly_from_official_filepath": True,
              "method": {"dynamics_steps": 0, "policy_inferences": 0, "controls_applied": 0,
                         "static_calls": ["mj_kinematics", "mj_comPos", "mj_collision", "mj_geomDistance"],
                         "base_pose": "upright trunk at fixed reference height .55369 m; all joints use active k1_walk task default_joint_pos",
                         "joint_defaults_source": str(WALK_CONFIG.relative_to(ROOT)),
                         "reference_base_height_m": REFERENCE_BASE_HEIGHT_M,
                         "projection": "Orthographic along trunk +x; y-z convex native-geom silhouettes; trunk-relative coordinates",
                         "mesh_projection": "2-D convex hull of compiled mesh vertices, matching convex collision silhouette; no visual-only geometry included",
                         "cylinder_projection": "256-segment approximation", "grid_cell_m": GRID_M,
                         "grid_bounds_yz_m": [[-.75, .75], [-.30, .50]],
                         "interpolation_samples_per_pose": 11,
                         "scope": "trunk and both arms only; projected solid geometry is not ball-inflated, contact area, interception probability, or dynamic safety"},
              "protected_sha256_before": before, "protected_sha256_after": after,
              "protected_unchanged": before == after,
              "existing_e0_reconciled": all(r["matches_existing_e0"] for r in records),
              "geometry": records, "joints": joints,
              "self_collision": {"native_enabled_geom_masks": "all audited collision geoms are 1/1",
                                 "parent_child_filter_enabled": parent_filter,
                                 "explicit_body_excludes": excluded,
                                 "explicit_geom_pairs": model.npair, "upper_body_pairs": pairs,
                                 "distance_cross_check": "Separate compiled diagnostic copy disables nativeccd only for libccd distance queries; official source and primary model unchanged, no dynamics steps. Backend disagreement means signed distance cannot be treated as certified clearance.",
                                 "note": "Filtered pairs are not monitored by default self collision; zero detected contacts is not proof against visual-shell overlap or unsampled path collision."},
              "poses": poses,
              "static_reach_bound": {"method": "Triangle inequality: norm(shoulder anchor) + sum descendant body offset lengths + native shape circumradius + 1 micrometre numerical slack; independent of root orientation and arm joint angles",
                                     "per_geom": reach, "native_arm_max_radius_from_trunk_m": reach_radius,
                                     "reference_base_height_m": REFERENCE_BASE_HEIGHT_M,
                                     "max_arm_height_at_reference_base_m": REFERENCE_BASE_HEIGHT_M + reach_radius,
                                     "air_high_ball_center_height_m": 1.4, "ball_radius_m": .11,
                                     "air_high_ball_bottom_m": 1.29,
                                     "base_height_threshold_for_possible_vertical_overlap_m": 1.29 - reach_radius,
                                     "air_high_vertical_overlap_impossible_at_fixed_reference_base": REFERENCE_BASE_HEIGHT_M + reach_radius < 1.29,
                                     "scope": "Conditional static exclusion only when instantaneous trunk origin z stays below threshold. No statement about base rise, jumping, ball-flight height changes, contact timing, or dynamic safety."},
              "gaps": {"visual_only_shoulder_pitch_bodies": ["aaleft_shoulder_pitch_link", "aaright_shoulder_pitch_link"],
                       "supplement_required_for_native_torso_distal_arm_candidates": False,
                       "supplement_required_for_continuous_visual_shoulder_coverage_claim": True,
                       "decision": "native first; no collision geometry added; shoulder-root contact remains unverified"},
              "recommendation": {"sequence": ["native_default", "left_raise_0.15", "right_raise_0.15",
                                               "both_raise_0.15", "left_extend_0.15", "right_extend_0.15",
                                               "both_extend_0.15", "0.30 rad candidates after H0 decision and controlled trial checks"],
                                 "approval_status": "H0 informs already-authorized controlled H1 trials; static report alone is not execution or dynamic safety evidence",
                                 "required_next_evidence": ["controller joint ownership and transition protocol settled by H0 decision",
                                                            "free-base balance, actual tracking, torque saturation, warnings and self contacts",
                                                            "actual native geom IDs in ball contacts and matched torso-only baselines"],
                                 "static_is_not_dynamic_safety_proof": True}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["protected_unchanged"]:
        raise RuntimeError("Protected input changed during audit")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/defense_validation/geometry_audit.json")
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({"poses": len(result["poses"]), "protected_unchanged": result["protected_unchanged"],
                      "existing_e0_reconciled": result["existing_e0_reconciled"],
                      "dynamics_steps": 0, "output": str(args.output)}))
