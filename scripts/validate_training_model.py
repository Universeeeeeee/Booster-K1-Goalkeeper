"""I2 geometry evidence and paired native/training dynamics; no policy training."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from k1_goalkeeper.scene import K1_XML, build_scene
from k1_goalkeeper.simulation import Shot
from k1_goalkeeper.training_actions import JointTargetBackend
from k1_goalkeeper.training_env import K1GoalkeeperEnv
from k1_goalkeeper.training_model import (MODEL_VERSION, SHANK_FROMTO, SHANK_RADIUS,
                                         build_training_scene)
from scripts.validate_training_env import (array_hash, case_definitions, file_hash,
                                          prefix_proofs, run_case, write_json)


INVARIANTS = ('qpos0', 'body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
              'body_pos', 'body_quat', 'jnt_range', 'jnt_axis', 'jnt_pos',
              'dof_armature', 'dof_damping', 'dof_frictionloss', 'actuator_gear',
              'actuator_forcerange', 'actuator_ctrlrange', 'actuator_gainprm',
              'actuator_biasprm')


def model_audit(native_path, training_path):
    native, training = [mujoco.MjModel.from_xml_path(str(p))
                        for p in (native_path, training_path)]
    a, b = [ET.parse(p).getroot() for p in (native_path, training_path)]
    additions = []
    for body in b.findall('.//body'):
        for geom in list(body.findall('geom')):
            if geom.get('name', '').startswith('i2_'):
                additions.append({'body': body.get('name'), **geom.attrib})
                body.remove(geom)
    b.set('model', a.get('model'))
    # Ignore formatting only. All original physics and visual XML must match.
    for root in (a, b):
        for node in root.iter():
            node.text = node.tail = None
    unchanged = ET.tostring(a) == ET.tostring(b)
    checks = {field: bool(np.array_equal(getattr(native, field), getattr(training, field)))
              for field in INVARIANTS}
    return {'version': MODEL_VERSION, 'only_xml_changes_are_two_capsules_and_model_name': unchanged,
            'additions': additions, 'compiled_invariants': checks,
            'native_inertia_sha256': array_hash({f: getattr(native, f) for f in INVARIANTS}),
            'training_inertia_sha256': array_hash({f: getattr(training, f) for f in INVARIANTS}),
            'physics_dt_s': training.opt.timestep,
            'native_enabled_robot_geoms': int(sum(native.geom_contype[g] != 0 for g in range(native.ngeom)
                                               if native.geom(g).name.startswith('k1_'))),
            'training_enabled_robot_geoms': int(sum(training.geom_contype[g] != 0 for g in range(training.ngeom)
                                                 if training.geom(g).name.startswith(('k1_', 'i2_')))),
            'passed': bool(unchanged and len(additions) == 2 and all(checks.values()))}


def hull_proofs():
    """Check capsule support against every official convex-hull plane.

    Use raw STL vertices as a second check on compiler coordinates and planes.
    This proves containment in the convex envelope, not in every mesh cavity.
    """
    root = ET.parse(K1_XML).getroot()
    root.find('compiler').set('meshdir', str(K1_XML.parent/'meshes'))
    for side in ('left', 'right'):
        geom = root.find(f".//body[@name='{side}_knee_pitch_link']/geom[@type='mesh']")
        geom.set('contype', '1')
        geom.set('conaffinity', '1')
        geom.set('name', f'hull_{side}')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    proofs = []
    for side in ('left', 'right'):
        geom = model.geom(f'hull_{side}').id
        mid = int(model.geom_dataid[geom])
        adr = int(model.mesh_graphadr[mid])
        if adr < 0:
            raise RuntimeError('Convex hull was not compiled')
        nv, nf = map(int, model.mesh_graph[adr:adr+2])
        start = adr+2+3*nv+3*nf
        faces = model.mesh_graph[start:start+3*nf].reshape(-1, 3)
        va = int(model.mesh_vertadr[mid])
        vertices = model.mesh_vert[va:va+model.mesh_vertnum[mid]].astype(float)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[geom]
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals[np.sum(normals*(triangles[:, 0]-vertices.mean(axis=0)), axis=1) < 0] *= -1
        offsets = np.sum(normals*triangles[:, 0], axis=1)
        endpoints = np.asarray(SHANK_FROMTO).reshape(2, 3)
        margin = offsets - (normals @ endpoints.T).max(axis=1) - SHANK_RADIUS
        mesh_path = K1_XML.parent/'meshes'/f'{side.title()}_Shank.STL'
        raw = mesh_path.read_bytes()
        count = int.from_bytes(raw[80:84], 'little')
        dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
        stl = np.unique(np.frombuffer(raw, dtype=dtype, count=count, offset=84)['vertices'].reshape(-1, 3), axis=0)
        # Chunk the independent raw-asset check to keep memory bounded.
        violation = max(float((points @ normals.T-offsets).max()) for points in np.array_split(stl, 16))
        bounds_error = float(max(np.abs(stl.min(axis=0)-vertices.min(axis=0)).max(),
                                np.abs(stl.max(axis=0)-vertices.max(axis=0)).max()))
        proofs.append({'side': side, 'source': str(mesh_path.relative_to(ROOT)),
                       'source_sha256': file_hash(mesh_path), 'hull_faces': nf,
                       'minimum_capsule_inside_hull_margin_m': float(margin.min()),
                       'raw_stl_max_hull_plane_violation_m': violation,
                       'raw_stl_vs_compiled_bounds_error_m': bounds_error,
                       'passed': bool(margin.min() > .001 and violation < 1e-6 and bounds_error < 1e-6)})
    return proofs


def self_contacts(model, data):
    rows = []
    for c in data.contact:
        names = [model.geom(g).name for g in (c.geom1, c.geom2)]
        if all(n.startswith(('k1_', 'i2_')) for n in names) and c.dist < -1e-6:
            rows.append({'geoms': names, 'distance_m': float(c.dist)})
    return rows


def static_audit(model):
    backend = JointTargetBackend(model)
    d, mapping = backend.data, backend.mapping
    original = d.qpos.copy()
    poses = [('default', mapping.default_positions.copy())]
    for kind in ('raise', 'extend', 'crouch'):
        for fraction in np.linspace(.1, 1., 10):
            angles = mapping.default_positions.copy()
            for side, sign in [('left', 1), ('right', -1)]:
                changes = ({f'{side}_shoulder_roll_joint': sign*.3} if kind == 'raise' else
                           {f'aa{side}_shoulder_pitch_joint': -.3} if kind == 'extend' else
                           {f'{side}_hip_pitch_joint': -.1, f'{side}_knee_pitch_joint': .2,
                            f'{side}_ankle_pitch_joint': -.1})
                for name, change in changes.items():
                    angles[mapping.joint_names.index(name)] += fraction*change
            poses.append((f'{kind}_{fraction:.1f}', angles))
    moderate = []
    for name, angles in poses:
        d.qpos[:] = original
        d.qpos[mapping.qpos_addresses] = angles
        mujoco.mj_forward(model, d)
        moderate.append({'name': name, 'self_contacts': self_contacts(model, d)})
    # Diagnostic only: individual joint limits are not a collision-free action set.
    limits = []
    for index, joint in enumerate(mapping.joint_names):
        for value in np.linspace(*mapping.joint_ranges[index], 9):
            d.qpos[:] = original
            d.qpos[mapping.qpos_addresses[index]] = value
            mujoco.mj_forward(model, d)
            contacts = self_contacts(model, d)
            if contacts:
                limits.append({'joint': joint, 'angle_rad': float(value), 'self_contacts': contacts})
    backend.reset()
    initial = [{'geoms': [model.geom(c.geom1).name, model.geom(c.geom2).name],
                'distance_m': float(c.dist)} for c in d.contact if c.dist < -1e-6]
    return {'initial_penetrations': initial, 'moderate_poses': moderate,
            'joint_limit_samples': len(mapping.joint_names)*9,
            'joint_limit_colliding_samples': limits,
            'passed': not initial and all(not p['self_contacts'] for p in moderate)}


def coverage_probes(models):
    """Same sphere placement in each compiled model; analytic segment distance too."""
    results = []
    for side, y in [('left', -.096), ('right', .096)]:
        position = np.array([10.24862374556, y, .171])
        record = {'side': side, 'ball_position': position.tolist(), 'models': {}}
        for variant, model in models.items():
            backend = JointTargetBackend(model)
            d, mapping = backend.data, backend.mapping
            d.qpos[mapping.ball_qpos_address:mapping.ball_qpos_address+3] = position
            mujoco.mj_forward(model, d)
            ball = model.geom('ball_collision').id
            contacts = [{'geom': model.geom(c.geom2 if c.geom1 == ball else c.geom1).name,
                         'distance_m': float(c.dist)} for c in d.contact
                        if ball in (c.geom1, c.geom2) and c.dist < 0]
            record['models'][variant] = contacts
            if variant == 'training':
                body = model.body(f'{side}_knee_pitch_link').id
                endpoints = np.asarray(SHANK_FROMTO).reshape(2, 3) @ d.xmat[body].reshape(3, 3).T+d.xpos[body]
                segment = endpoints[1]-endpoints[0]
                fraction = np.clip((position-endpoints[0]) @ segment/(segment @ segment), 0, 1)
                distance = float(np.linalg.norm(position-(endpoints[0]+fraction*segment))-.11-SHANK_RADIUS)
                record['analytic_sphere_capsule_distance_m'] = distance
                matched = [c for c in contacts if c['geom'] == f'i2_{side}_shank']
                record['analytic_matches_solver'] = bool(matched and abs(matched[0]['distance_m']-distance) < 1e-9)
        record['passed'] = bool(not record['models']['native'] and record['analytic_matches_solver'])
        results.append(record)
    return results


def cases():
    result = case_definitions()
    for side, y in [('left', -.096), ('right', .096)]:
        result.append({'name': f'{side}_shank_impact', 'mode': 'joint_targets',
                       'task': 'single_shot', 'launch_time': 0., 'time_limit': 3.,
                       'shot': asdict(Shot(position=(10.12862374556, y, .1754145), velocity=(4., 0., 0.)))})
    for name, position, velocity in [('torso_impact', (9.8, 0., .72), (3., 0., 0.)),
                                      ('miss', (10., 2., .5), (4., 0., 0.))]:
        result.append({'name': name, 'mode': 'joint_targets', 'task': 'single_shot',
                       'launch_time': 0., 'time_limit': 3.,
                       'shot': asdict(Shot(position=position, velocity=velocity))})
    return result


def delivered_result(model, directory, summary):
    """Independently reopen the delivered 1 ms archive, not just the env getter."""
    with np.load(directory/'evaluation.npz') as archive:
        contacts = archive['native_contacts']
        robot = {g for g in range(model.ngeom) if model.geom(g).name.startswith(('k1_', 'i2_'))}
        self_rows = [row for row in contacts if int(row[1]) in robot and int(row[2]) in robot]
        added = [row for row in contacts if row[4] and any(model.geom(int(g)).name.startswith('i2_') for g in row[1:3])]
        impact = None
        ball_rows = contacts[contacts[:, 4] == 1]
        if len(ball_rows):
            first = int(np.searchsorted(archive['time'], ball_rows[0, 0]))
            last = int(np.searchsorted(archive['time'], ball_rows[-1, 0]))
            before = archive['ball_velocity'][max(0, first-1)]
            after = archive['ball_velocity'][min(last+1, len(archive['time'])-1)]
            impact = {'before_velocity_m_s': before.tolist(), 'after_velocity_m_s': after.tolist(),
                      'velocity_change_norm_m_s': float(np.linalg.norm(after-before))}
        checks = {'archive_hash_matches': file_hash(directory/'evaluation.npz') == summary['archive_sha256']['evaluation'],
                  'finite_delivered_state': all(np.isfinite(archive[k]).all() for k in ('qpos', 'qvel', 'ctrl')),
                  'no_positive_force_self_contacts': not self_rows,
                  'no_fall': summary['final_info']['termination_reason'] != 'fall',
                  'no_bad_warnings': not any(summary['warning_counts'].values()),
                  'within_official_torque_limits': bool(np.all(np.abs(archive['applied_torque']) <= JointTargetBackend(model).mapping.effort+1e-5))}
    return {'checks': checks, 'added_geom_force_samples': len(added),
            'first_added_geom_force_time_s': float(added[0][0]) if added else None,
            'maximum_added_geom_normal_force_n': max((float(row[3]) for row in added), default=0.),
            'ball_velocity_change': impact, 'passed': all(checks.values())}


def validate(output):
    output.mkdir(parents=True, exist_ok=False)
    protected_path = ROOT/'reports/training_model_validation/protected_manifest.json'
    protected = json.loads(protected_path.read_text())
    def verify_sources():
        changed = [p for p, digest in protected.items() if file_hash(ROOT/p) != digest]
        if changed:
            raise RuntimeError(f'Existing files changed: {changed}')
        return {'file_count': len(protected), 'all_match': True}
    before = verify_sources()
    paths = {'native': build_scene(output/'native.xml'),
             'training': build_training_scene(output/'training.xml')}
    models = {name: mujoco.MjModel.from_xml_path(str(path)) for name, path in paths.items()}
    audit = model_audit(paths['native'], paths['training'])
    hull = hull_proofs()
    static = {name: static_audit(model) for name, model in models.items()}
    probes = coverage_probes(models)
    write_json(output/'geometry.json', {'model': audit, 'hull': hull, 'static': static, 'probes': probes})
    passed = audit['passed'] and all(p['passed'] for p in hull+probes) and all(p['passed'] for p in static.values())
    if not passed:
        raise RuntimeError('Geometry gates failed; inspect geometry.json before dynamic tests')
    protocol = {'version': MODEL_VERSION, 'cases': cases(), 'repeats': 2,
                'physics_dt_s': .001, 'control_dt_s': .02,
                'package_versions': {name: importlib.metadata.version(name) for name in ('mujoco', 'numpy', 'gymnasium', 'torch')},
                'source_sha256': {name: file_hash(ROOT/name) for name in
                                  ('k1_goalkeeper/training_model.py', 'scripts/validate_training_model.py',
                                   'scripts/validate_training_env.py', 'k1_goalkeeper/training_env.py')},
                'model_hashes': {k: file_hash(p) for k, p in paths.items()},
                'scope': 'Fixed reference controls; no learning, recovery or save-rate claim'}
    write_json(output/'protocol.json', protocol)
    results = []
    for case in protocol['cases']:
        for variant, model in models.items():
            repeats = []
            for repeat in range(2):
                directory = output/variant/case['name']/f'repeat_{repeat}'
                summary = run_case(model, case, directory)
                delivery = delivered_result(model, directory, summary)
                repeats.append({'summary': summary, 'delivery': delivery})
            deterministic = repeats[0]['summary']['channel_array_sha256'] == repeats[1]['summary']['channel_array_sha256']
            case_pass = deterministic and all(r['summary']['interface_passed'] and r['delivery']['passed'] for r in repeats)
            if case['name'].endswith('shank_impact') and variant == 'training':
                case_pass &= all(r['delivery']['added_geom_force_samples'] > 0 for r in repeats)
            if case['name'] == 'miss':
                case_pass &= all(r['summary']['actual_outcome'] == 'Miss' and not r['summary']['robot_contact_seen'] for r in repeats)
            results.append({'case': case['name'], 'variant': variant, 'deterministic': deterministic,
                            'passed': bool(case_pass), 'repeats': repeats})
            print(case['name'], variant, 'PASS' if case_pass else 'FAIL', flush=True)
    isolation = {mode: prefix_proofs(models['training'], mode) for mode in ('joint_targets', 'body_velocity')}
    write_json(output/'reset_isolation.json', isolation)
    report = {'version': MODEL_VERSION, 'geometry_passed': passed, 'results': results,
              'protected_before': before, 'protected_after': verify_sources(),
              'reset_isolation_passed': all(r['passed'] for r in isolation.values()),
              'all_dynamic_cases_passed': all(r['passed'] for r in results) and all(r['passed'] for r in isolation.values())}
    write_json(output/'results.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite evidence')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    result = validate(args.output.resolve())
    if not result['all_dynamic_cases_passed']:
        raise SystemExit('Some paired cases failed; see results.json')
