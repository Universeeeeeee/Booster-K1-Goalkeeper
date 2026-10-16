"""Reproduce rejected I2 mesh candidates using two collision backends."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from k1_goalkeeper.scene import build_scene
from k1_goalkeeper.training_actions import JointTargetBackend
from scripts.validate_training_env import write_json


def audit():
    with tempfile.TemporaryDirectory(prefix='i2-candidate-audit-') as directory:
        path = build_scene(Path(directory)/'candidate.xml')
        tree = ET.parse(path)
        names = []
        for body in tree.findall('.//body'):
            if body.get('name', '').endswith(('shoulder_pitch_link', 'hip_pitch_link', 'ankle_pitch_link')):
                visual = body.find("geom[@type='mesh']")
                collision = deepcopy(visual)
                name = f"candidate_{visual.get('mesh')}"
                collision.attrib.update(name=name, contype='1', conaffinity='1', group='3')
                body.append(collision)
                names.append(name)
        tree.write(path, encoding='unicode')
        records = []
        for name in names:
            record = {'geom': name, 'backends': {}}
            for backend_name in ('nativeccd', 'libccd'):
                model = mujoco.MjModel.from_xml_path(str(path))
                if backend_name == 'libccd':
                    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
                backend = JointTargetBackend(model)
                data = backend.data
                geom = model.geom(name).id
                ball = model.geom('ball_collision').id
                q = backend.mapping.ball_qpos_address
                data.qpos[q:q+3] = data.geom_xpos[geom].copy()
                mujoco.mj_forward(model, data)
                mesh = int(model.geom_dataid[geom])
                start = int(model.mesh_vertadr[mesh])
                vertices = model.mesh_vert[start:start+model.mesh_vertnum[mesh]]
                vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom]
                distances = np.linalg.norm(vertices-data.qpos[q:q+3], axis=1)
                record['backends'][backend_name] = {
                    'ball_center_m': data.qpos[q:q+3].tolist(),
                    'ball_radius_m': float(model.geom_size[ball, 0]),
                    'maximum_distance_to_mesh_vertex_m': float(distances.max()),
                    'ball_contains_every_mesh_vertex': bool(distances.max() < model.geom_size[ball, 0]),
                    'target_contact_distances_m': [float(c.dist) for c in data.contact
                                                  if {int(c.geom1), int(c.geom2)} == {ball, geom}],
                }
            records.append(record)
        return {'mujoco_version': mujoco.__version__, 'candidates': records,
                'purpose': 'Deep-overlap diagnostic, not a legal shot or dynamic safety test',
                'decision': 'Rejected direct mesh copies for this I2 version; main model keeps nativeccd'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new output path')
    write_json(args.output, audit())
