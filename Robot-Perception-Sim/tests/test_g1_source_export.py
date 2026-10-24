"""The portable export must retain compiled mesh frames and joint dynamics."""
import importlib.util
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_g1_source', PROJECT / 'scripts/export_g1_source.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture(scope='module')
def exported():
    with tempfile.TemporaryDirectory(dir=PROJECT / 'results/g1') as directory:
        result = exporter.export(exporter.DEFAULT, Path(directory) / 'source.json')
        yield result


def test_compiled_mesh_vertices_and_dynamics(exported):
    model = mujoco.MjModel.from_xml_path(str(exporter.ROOT / exported['contract']['source']['scene']))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for geom in exported['contract']['geoms']:
        if geom['mesh_id'] is not None:
            mesh = geom['mesh_id']
            vertices = np.asarray(exported['meshes'][str(mesh)]['vertices'])
            actual = vertices @ data.geom_xmat[geom['id']].reshape(3, 3).T + data.geom_xpos[geom['id']]
            start = model.mesh_vertadr[mesh]
            expected = model.mesh_vert[start:start+model.mesh_vertnum[mesh]] @ data.geom_xmat[geom['id']].reshape(3, 3).T + data.geom_xpos[geom['id']]
            np.testing.assert_array_equal(actual, expected)
    for joint in exported['contract']['robot']['joints']:
        assert joint['armature_kg_m2'] == model.dof_armature[joint['dof_address']]
        assert joint['damping_nm_s_rad'] == 0
        assert joint['frictionloss_nm'] == 0
    dynamic = [b for b in exported['contract']['bodies'] if b['dynamic']]
    assert len(dynamic) == 24  # 23 K1 rigid links and ball; no massless camera payload.
    assert sum(b['mass_kg'] for b in dynamic if b['name'] != 'ball') == pytest.approx(19.666)
    assert exported['physics_steps'] == 0


def test_changed_source_hash_rejected(tmp_path):
    contract = json.loads(exporter.DEFAULT.read_text())
    contract['source']['hashes'][0]['sha256'] = '0' * 64
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(contract))
    with tempfile.TemporaryDirectory(dir=PROJECT / 'results/g1') as directory:
        with pytest.raises(ValueError, match='Frozen source hash mismatch'):
            exporter.export(manifest, Path(directory) / 'source.json')


def test_output_outside_results_rejected(tmp_path):
    with pytest.raises(ValueError, match='under Robot-Perception-Sim/results'):
        exporter.export(exporter.DEFAULT, tmp_path / 'source.json')
