#!/usr/bin/env python3
"""Export frozen compiled MuJoCo geometry for the independent USD builder; no steps."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import mujoco

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
DEFAULT = PROJECT / 'results/g1/source_contract_20261023T172323_398380Z/asset_manifest.json'


def export(manifest_path, output_path):
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    output_path = output_path.resolve()
    if not output_path.is_relative_to(PROJECT / 'results'):
        raise ValueError('output must be a new file under Robot-Perception-Sim/results')
    contract = json.loads(manifest_path.read_text())
    for item in contract['source']['hashes']:
        path = ROOT / item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Frozen source hash mismatch: {path}')
    model = mujoco.MjModel.from_xml_path(str(ROOT / contract['source']['scene']))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # geom_pos/quat already include MuJoCo mesh recentering; use compiled vertices,
    # not original STL vertices, so this transformation is applied exactly once.
    meshes = {}
    for mesh in range(model.nmesh):
        va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        meshes[str(mesh)] = {'vertices': model.mesh_vert[va:va+vn].tolist(),
                             'faces': model.mesh_face[fa:fa+fn].tolist()}
    for geom in contract['geoms']:
        i = geom['id']
        geom['mesh_id'] = int(model.geom_dataid[i]) if geom['type'] == 'mjGEOM_MESH' else None
        geom['rgba'] = model.geom_rgba[i].tolist()
        if model.geom_matid[i] >= 0:
            geom['rgba'] = model.mat_rgba[model.geom_matid[i]].tolist()
    for joint in contract['robot']['joints']:
        dof = joint['dof_address']
        joint['armature_kg_m2'] = float(model.dof_armature[dof])
        joint['damping_nm_s_rad'] = float(model.dof_damping[dof])
        joint['frictionloss_nm'] = float(model.dof_frictionloss[dof])
    for body in contract['bodies']:
        i = body['id']
        body['position_world_initial_m'] = data.xpos[i].tolist()
        body['quaternion_world_initial_wxyz'] = data.xquat[i].tolist()
        body['weld_id'] = int(model.body_weldid[i])
        body['dynamic'] = bool(model.body_dofnum[i] > 0 and body['mass_kg'] > 0)
    export_data = {'schema_version': 1, 'contract': contract, 'meshes': meshes,
                   'frozen_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   'physics_steps': 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x') as handle:
        json.dump(export_data, handle, separators=(',', ':'))
    return export_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.manifest, args.output)
    print(json.dumps({'output': str(args.output), 'meshes': len(result['meshes']), 'physics_steps': 0}))


if __name__ == '__main__':
    main()
