#!/usr/bin/env python3
"""Derive an explicitly static robot collision fixture from the G1 free scene.

No controller/stability claim: frozen source link world poses stay unchanged.
Run with Isaac's USD Python environment (no SimulationApp required).
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stage = Usd.Stage.Open(str(args.source))
    disabled = []
    joints = []
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdPhysics.Joint):
            joints.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.GetName().endswith('_ball'):
            disabled.append(str(prim.GetPath()))
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    for path in joints:
        stage.RemovePrim(path)
    if len(disabled) != 23:
        raise ValueError(f'Expected 23 frozen robot links, got {len(disabled)}')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bounds = []
    for prim in stage.Traverse():
        if (prim.HasAPI(UsdPhysics.CollisionAPI)
                and any(str(prim.GetPath()).startswith(path + "/") for path in disabled)):
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            bounds.append((str(prim.GetPath()), box.GetMin(), box.GetMax()))
    if not bounds:
        raise ValueError("No static robot collision bounds found")
    initial_checks = []
    for shot in json.loads(args.catalog.read_text())["cases"]:
        radius = .11
        distances = [(sum(max(lo[i]-shot["position"][i], 0, shot["position"][i]-hi[i])**2
                          for i in range(3))**.5, path) for path, lo, hi in bounds]
        distance, path = min(distances)
        if distance < radius:
            raise ValueError(f"Launch overlaps conservative robot collider AABB: {shot['name']}")
        initial_checks.append({"case": shot["name"], "minimum_robot_aabb_distance_m": distance,
                               "nearest_collider": path, "ball_radius_m": radius})
    stage.GetRootLayer().Export(str(args.output))
    audit = {'initial_robot_overlap_checks': initial_checks, 'robot_collider_count': len(bounds), 'robot_mode': 'static_collision_fixture', 'disabled_rigid_body_paths': disabled,
             'removed_joint_count': len(joints), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
             'scene_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
             'limitations': 'No dynamics/control/stability claim; full-scene G1 materials retained'}
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()
