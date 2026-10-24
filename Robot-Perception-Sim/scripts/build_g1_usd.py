#!/usr/bin/env python3
"""Build an independent USD/PhysX scene from a frozen MuJoCo export (CPU only)."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema


def quat(values):
    return Gf.Quatf(float(values[0]), Gf.Vec3f(*values[1:]))


def pose(prim, position, rotation):
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddOrientOp().Set(quat(rotation))


def body_path(body):
    return body.get('usd_path', '/World/Bodies/b%d_%s' % (body['id'], body['name']))


def geom_path(geom, bodies):
    return body_path(bodies[geom['body_id']]) + '/g%d_%s' % (geom['id'], geom['name'])


def build(source_path, output_path, robot='free', restitution=0.0):
    source = json.loads(Path(source_path).read_text())
    contract, meshes = source['contract'], source['meshes']
    output_path = Path(output_path).resolve()
    script_dir = Path(__file__).resolve().parent
    allowed = script_dir if script_dir.name == 'g1_migration' else script_dir.parent / 'results'
    if not output_path.is_relative_to(allowed):
        raise ValueError('output must be under project results or isolated remote g1_migration')
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
    gravity = Gf.Vec3f(*contract['world']['gravity_m_s2'])
    scene.CreateGravityDirectionAttr(gravity.GetNormalized())
    scene.CreateGravityMagnitudeAttr(gravity.GetLength())
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    ps.CreateTimeStepsPerSecondAttr(round(1 / contract['world']['timestep_s']))
    ps.CreateEnableGPUDynamicsAttr(False)
    ps.CreateBroadphaseTypeAttr('MBP')
    ps.CreateSolverTypeAttr('TGS')
    bodies = contract['bodies']
    robot_ids = {b['id'] for b in bodies if b['name'] not in ('world', 'ball', 'goal_left', 'goal_right')}
    included = {b['id'] for b in bodies if robot != 'absent' or b['id'] not in robot_ids}
    for b in bodies:
        if b['id'] not in included:
            continue
        # Separate rigid bodies have world poses; massless camera frames are
        # parented to their welded rigid body and retain their local transforms.
        path = body_path(b)
        welded = b['mass_kg'] == 0 and b['id'] in robot_ids
        if welded:
            path = body_path(bodies[b['parent_id']]) + '/b%d_%s' % (b['id'], b['name'])
        b['usd_path'] = path
        node = UsdGeom.Xform.Define(stage, path)
        pose(node, b['position_parent_m'] if welded else b['position_world_initial_m'],
             b['quaternion_parent_wxyz'] if welded else b['quaternion_world_initial_wxyz'])
        if b['dynamic']:
            UsdPhysics.RigidBodyAPI.Apply(node.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(node.GetPrim())
            mass.CreateMassAttr(b['mass_kg'])
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*b['inertial_position_body_m']))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*b['inertia_principal_kg_m2']))
            mass.CreatePrincipalAxesAttr(quat(b['inertial_quaternion_body_wxyz']))
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(node.GetPrim())
            rb.CreateLinearDampingAttr(0.0)
            rb.CreateAngularDampingAttr(0.0)
            rb.CreateMaxAngularVelocityAttr(100000.0)
        if b['name'] == 'trunk':
            UsdPhysics.ArticulationRootAPI.Apply(node.GetPrim())
            PhysxSchema.PhysxArticulationAPI.Apply(node.GetPrim()).CreateEnabledSelfCollisionsAttr(True)
    geoms = [g for g in contract['geoms'] if g['body_id'] in included]
    limitations = ['MuJoCo solref/solimp and condim have no exact PhysX mapping; restitution is an explicit uncalibrated input.',
                   'Plane collision uses the native USD Plane schema; runtime contact behavior must be verified.',
                   'Dynamic mesh colliders use PhysX convex hull cooking; MuJoCo mesh collision also uses convex hulls, but cooking algorithms differ.',
                   'Texture checker/skybox and MuJoCo reflectance are not transferred; compiled surface geometry and material colors are retained.',
                   'MuJoCo torsional/rolling friction and actuator/controller behavior are not reproduced by USD materials/drives.']
    if robot == 'absent':
        limitations.append('Robot-absent materials bake the source ball priority-1 sliding friction 0.6 into every collider, matching ball-ground/frame coefficients only.')
    else:
        limitations.append('Robot scene keeps source per-geom friction with max combine; ball-goal/robot sliding friction is 1.0 versus source priority-1 value 0.6. Full-scene material priority parity is unresolved.')
    for g in geoms:
        path, typ, size = geom_path(g, bodies), g['type'], g['size_m']
        if typ == 'mjGEOM_MESH':
            obj = UsdGeom.Mesh.Define(stage, path)
            mesh = meshes[str(g['mesh_id'])]
            obj.CreatePointsAttr(mesh['vertices'])
            obj.CreateFaceVertexCountsAttr([3] * len(mesh['faces']))
            obj.CreateFaceVertexIndicesAttr([i for face in mesh['faces'] for i in face])
            obj.CreateSubdivisionSchemeAttr('none')
        elif typ == 'mjGEOM_PLANE':
            obj = UsdGeom.Plane.Define(stage, path)
            obj.CreateAxisAttr('Z')
            obj.CreateWidthAttr(200.0)
            obj.CreateLengthAttr(200.0)
        elif typ == 'mjGEOM_BOX':
            obj = UsdGeom.Cube.Define(stage, path)
            obj.CreateSizeAttr(2.0)
        elif typ == 'mjGEOM_SPHERE':
            obj = UsdGeom.Sphere.Define(stage, path)
            obj.CreateRadiusAttr(size[0])
        elif typ == 'mjGEOM_CYLINDER':
            obj = UsdGeom.Cylinder.Define(stage, path)
            obj.CreateRadiusAttr(size[0])
            obj.CreateHeightAttr(2 * size[1])
            obj.CreateAxisAttr('Z')
        else:
            raise ValueError('Unsupported source geom: ' + typ)
        pose(obj, g['position_body_m'], g['quaternion_body_wxyz'])
        if typ == 'mjGEOM_BOX':
            obj.AddScaleOp().Set(Gf.Vec3f(*size))
        obj.CreateDisplayColorAttr([Gf.Vec3f(*g['rgba'][:3])])
        obj.CreateDisplayOpacityAttr([g['rgba'][3]])
        if g['contype'] or g['conaffinity']:
            UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
            if typ == 'mjGEOM_MESH':
                UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr('convexHull')
            contact = PhysxSchema.PhysxCollisionAPI.Apply(obj.GetPrim())
            contact.CreateContactOffsetAttr(0.001)
            contact.CreateRestOffsetAttr(0.0)
            # Each collider owns its source sliding coefficient. Combine mode is
            # max because MuJoCo uses maximum per-geom coefficients at equal priority.
            material_path = '/World/Materials/g%d' % g['id']
            from pxr import UsdShade
            material = UsdShade.Material.Define(stage, material_path)
            mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            friction = 0.6 if robot == 'absent' else g['friction'][0]
            mat.CreateStaticFrictionAttr(friction)
            mat.CreateDynamicFrictionAttr(friction)
            mat.CreateRestitutionAttr(restitution)
            pm = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
            pm.CreateFrictionCombineModeAttr('max')
            pm.CreateRestitutionCombineModeAttr('max')
            UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(material, materialPurpose='physics')
    colliders = [g for g in geoms if g['contype'] or g['conaffinity']]
    filtered = []
    explicit = {tuple(sorted(p)) for p in contract['collision']['explicit_pair_geom_ids']}
    excluded = set(contract['collision']['exclude_signature'])
    for a, b in itertools.combinations(colliders, 2):
        ba, bb = bodies[a['body_id']], bodies[b['body_id']]
        wa, wb = ba['weld_id'], bb['weld_id']
        pair = tuple(sorted((a['id'], b['id'])))
        signature = (min(ba['id'], bb['id']) << 16) + max(ba['id'], bb['id'])
        same = wa == wb
        parent = wa != 0 and wb != 0 and (bodies[wa]['parent_id'] == wb or bodies[wb]['parent_id'] == wa)
        mask = (a['contype'] & b['conaffinity']) or (b['contype'] & a['conaffinity'])
        if pair not in explicit and (same or parent or signature in excluded or not mask):
            UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(geom_path(a, bodies))).CreateFilteredPairsRel().AddTarget(geom_path(b, bodies))
            filtered.append(list(pair))
    if robot != 'absent':
        for j in contract['robot']['joints']:
            b = next(b for b in bodies if b['name'] == j['body'])
            if abs(j['scene_initial_q_rad']) > 1e-12:
                raise ValueError('Nonzero reference pose requires joint reference-angle handling')
            joint = UsdPhysics.RevoluteJoint.Define(stage, '/World/Joints/' + j['name'])
            joint.CreateBody0Rel().SetTargets([body_path(bodies[b['parent_id']])])
            joint.CreateBody1Rel().SetTargets([body_path(b)])
            local1 = Gf.Rotation(Gf.Vec3d(1,0,0), Gf.Vec3d(*j['axis_body'])).GetQuat()
            relq = Gf.Quatd(b['quaternion_parent_wxyz'][0], Gf.Vec3d(*b['quaternion_parent_wxyz'][1:]))
            local0 = relq * local1
            anchor = Gf.Vec3d(*j['anchor_body_m'])
            parent_anchor = Gf.Vec3d(*b['position_parent_m']) + relq.Transform(anchor)
            PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateArmatureAttr(j['armature_kg_m2'])
            joint.CreateAxisAttr('X')
            joint.CreateLocalPos0Attr(Gf.Vec3f(parent_anchor))
            joint.CreateLocalPos1Attr(Gf.Vec3f(anchor))
            joint.CreateLocalRot0Attr(Gf.Quatf(local0))
            joint.CreateLocalRot1Attr(Gf.Quatf(local1))
            if j['limited']:
                joint.CreateLowerLimitAttr(math.degrees(j['range_rad'][0]))
                joint.CreateUpperLimitAttr(math.degrees(j['range_rad'][1]))
            joint.CreateCollisionEnabledAttr(True)  # Explicit source pair filtering above.
    world.GetPrim().CreateAttribute('g1:robotJointOrder', Sdf.ValueTypeNames.StringArray).Set(contract['robot']['robot_joint_order'])
    world.GetPrim().CreateAttribute('g1:policyActionOrder', Sdf.ValueTypeNames.StringArray).Set(contract['robot']['policy_action_order'])
    world.GetPrim().CreateAttribute('g1:policyToRobotIndices', Sdf.ValueTypeNames.IntArray).Set(contract['robot']['policy_to_robot_indices'])
    stage.GetRootLayer().Save()
    report = {'stage': str(output_path), 'robot': robot, 'free_base': robot == 'free',
              'ball_prim': body_path(next(b for b in bodies if b['name'] == 'ball')),
              'physics_scene': '/World/PhysicsScene', 'joint_count': 22 if robot == 'free' else 0,
              'collider_count': len(colliders), 'filtered_geom_pairs': filtered,
              'restitution': restitution, 'physics_parity_verified': False,
              'frozen_manifest_sha256': source['frozen_manifest_sha256'], 'limitations': limitations}
    readback = audit_saved_stage(output_path, contract, included, geoms, robot, meshes, filtered)
    output_path.with_suffix('.audit.json').write_text(json.dumps(readback, indent=2) + '\n')
    report['saved_stage_audit_passed'] = readback['passed']
    output_path.with_suffix('.build.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def audit_saved_stage(path, contract, included, geoms, robot, meshes, filtered):
    """Reopen serialized USD and compare real schema values/world transforms."""
    for body in contract['bodies']:
        if body['mass_kg'] == 0 and body['parent_id'] != 0:
            parent = contract['bodies'][body['parent_id']]
            body['usd_path'] = body_path(parent) + '/b%d_%s' % (body['id'], body['name'])
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    errors, body_rows, joint_rows, geom_rows = [], [], [], []

    def check(label, actual, expected, tolerance=2e-6):
        if isinstance(expected, (float, int)):
            delta = abs(float(actual) - expected)
        else:
            delta = max((abs(float(a)-float(b)) for a,b in zip(actual, expected)), default=0)
        if delta > tolerance:
            errors.append({'field': label, 'max_abs_error': delta})

    def qlist(q):
        return [q.GetReal(), *q.GetImaginary()]

    for b in contract['bodies']:
        if b['id'] not in included:
            continue
        prim = stage.GetPrimAtPath(body_path(b))
        matrix = cache.GetLocalToWorldTransform(prim)
        position = list(matrix.ExtractTranslation())
        rotation = qlist(matrix.ExtractRotationQuat())
        check(b['name'] + ':world_position', position, b['position_world_initial_m'])
        expected = b['quaternion_world_initial_wxyz']
        if sum(a*v for a,v in zip(rotation, expected)) < 0:
            rotation = [-v for v in rotation]
        check(b['name'] + ':world_rotation', rotation, expected)
        row = {'name': b['name'], 'path': body_path(b), 'position_world_m': position,
               'quaternion_world_wxyz': rotation, 'rigid_body': prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        if b['dynamic']:
            mass = UsdPhysics.MassAPI(prim)
            row.update(mass_kg=mass.GetMassAttr().Get(),
                       com_body_m=list(mass.GetCenterOfMassAttr().Get()),
                       inertia_principal_kg_m2=list(mass.GetDiagonalInertiaAttr().Get()),
                       inertia_axes_wxyz=qlist(mass.GetPrincipalAxesAttr().Get()))
            check(b['name'] + ':mass', row['mass_kg'], b['mass_kg'])
            check(b['name'] + ':com', row['com_body_m'], b['inertial_position_body_m'])
            check(b['name'] + ':inertia', row['inertia_principal_kg_m2'], b['inertia_principal_kg_m2'], max(1e-10, max(b['inertia_principal_kg_m2'])*1e-7))
            check(b['name'] + ':principal_axes', row['inertia_axes_wxyz'], b['inertial_quaternion_body_wxyz'])
        body_rows.append(row)
    actual_filtered = set()
    for g in geoms:
        prim = stage.GetPrimAtPath(geom_path(g, contract['bodies']))
        transform = cache.GetLocalToWorldTransform(prim)
        body_transform = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body_path(contract['bodies'][g['body_id']])))
        expected = body_transform.Transform(Gf.Vec3d(*g['position_body_m']))
        check(g['name'] + ':world_center', list(transform.ExtractTranslation()), expected)
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpType(): op.Get() for op in xf.GetOrderedXformOps()}
        localq = ops[UsdGeom.XformOp.TypeOrient]
        check(g['name'] + ':local_orientation', qlist(localq), g['quaternion_body_wxyz'])
        world_rotation = qlist(transform.ExtractRotationQuat())
        check(g['name'] + ':collision', int(prim.HasAPI(UsdPhysics.CollisionAPI)), int(bool(g['contype'] or g['conaffinity'])), 0)
        dims = {}
        typ = g['type']
        if typ == 'mjGEOM_SPHERE':
            dims['radius_m'] = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
            check(g['name'] + ':radius', dims['radius_m'], g['size_m'][0], 1e-9)
        elif typ == 'mjGEOM_CYLINDER':
            obj = UsdGeom.Cylinder(prim)
            dims = {'radius_m': obj.GetRadiusAttr().Get(), 'height_m': obj.GetHeightAttr().Get(), 'axis': obj.GetAxisAttr().Get()}
            check(g['name'] + ':radius', dims['radius_m'], g['size_m'][0], 1e-9)
            check(g['name'] + ':height', dims['height_m'], 2*g['size_m'][1], 1e-9)
            if dims['axis'] != 'Z':
                errors.append({'field': g['name'] + ':cylinder_axis'})
        elif typ == 'mjGEOM_BOX':
            dims = {'cube_size': UsdGeom.Cube(prim).GetSizeAttr().Get(), 'scale': list(ops[UsdGeom.XformOp.TypeScale])}
            check(g['name'] + ':cube_size', dims['cube_size'], 2.0, 0)
            check(g['name'] + ':half_extents', dims['scale'], g['size_m'], 1e-7)
        elif typ == 'mjGEOM_MESH':
            mesh = UsdGeom.Mesh(prim)
            points = [list(v) for v in mesh.GetPointsAttr().Get()]
            faces = list(mesh.GetFaceVertexIndicesAttr().Get())
            counts = list(mesh.GetFaceVertexCountsAttr().Get())
            expected_mesh = meshes[str(g['mesh_id'])]
            if faces != [i for face in expected_mesh['faces'] for i in face] or counts != [3]*len(expected_mesh['faces']):
                errors.append({'field': g['name'] + ':mesh_topology'})
            if len(points) != len(expected_mesh['vertices']):
                errors.append({'field': g['name'] + ':mesh_vertex_count'})
            for i, (point, expected_point) in enumerate(zip(points, expected_mesh['vertices'])):
                check(g['name'] + ':vertex_' + str(i), point, expected_point, 1e-8)
            dims = {'point_count': len(points), 'triangle_count': len(counts),
                    'points_sha256': hashlib.sha256(json.dumps(points, separators=(',', ':')).encode()).hexdigest(),
                    'indices_sha256': hashlib.sha256(json.dumps(faces, separators=(',', ':')).encode()).hexdigest()}
        pairs = [str(p) for p in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()]
        for target in pairs:
            other = next(item for item in geoms if geom_path(item, contract['bodies']) == target)
            actual_filtered.add(tuple(sorted((g['id'], other['id']))))
        geom_rows.append({'name': g['name'], 'path': str(prim.GetPath()), 'type': prim.GetTypeName(),
                          'collision': prim.HasAPI(UsdPhysics.CollisionAPI),
                          'world_center_m': list(transform.ExtractTranslation()),
                          'world_quaternion_wxyz': world_rotation,
                          'dimensions': dims, 'filtered_pairs': pairs})
    if actual_filtered != {tuple(pair) for pair in filtered}:
        errors.append({'field': 'serialized_collision_filter_set'})
    if robot != 'absent':
        for j in contract['robot']['joints']:
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/Joints/' + j['name']))
            paths = [joint.GetBody0Rel().GetTargets()[0], joint.GetBody1Rel().GetTargets()[0]]
            frames = [cache.GetLocalToWorldTransform(stage.GetPrimAtPath(p)) for p in paths]
            anchors = [list(frames[0].Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))),
                       list(frames[1].Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get())))]
            rots = [joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()]
            axes = [list(frames[i].TransformDir(Gf.Quatd(rots[i]).Transform(Gf.Vec3d(1,0,0)))) for i in range(2)]
            check(j['name'] + ':anchors_coincide', anchors[0], anchors[1])
            expected_anchor = frames[1].Transform(Gf.Vec3d(*j['anchor_body_m']))
            check(j['name'] + ':source_anchor', anchors[1], expected_anchor)
            for i in range(2):
                check(j['name'] + ':axis' + str(i), axes[i], j['axis_world_initial'])
            limits = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
            check(j['name'] + ':limits', limits, [math.degrees(v) for v in j['range_rad']], 2e-5)
            armature = PhysxSchema.PhysxJointAPI(joint.GetPrim()).GetArmatureAttr().Get()
            check(j['name'] + ':armature', armature, j['armature_kg_m2'])
            joint_rows.append({'name': j['name'], 'body_paths': [str(p) for p in paths],
                               'anchors_world_m': anchors, 'axes_world': axes,
                               'limits_deg': limits, 'armature_kg_m2': armature})
    result = {'passed': not errors, 'errors': errors, 'bodies': body_rows,
              'joints': joint_rows, 'geoms': geom_rows,
              'physics_parity_verified': False}
    if errors:
        raise ValueError('Saved USD source audit failed: ' + json.dumps(errors))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--robot', choices=('free', 'absent'), default='free')
    parser.add_argument('--restitution', type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.robot, args.restitution)))


if __name__ == '__main__':
    main()
