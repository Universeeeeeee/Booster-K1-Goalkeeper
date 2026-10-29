import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from k1_goalkeeper.detection import GoalDetector
from trajectory.events import classify_record
from trajectory.shot_catalog import load_catalog, measured_csv_samples, validate_shot


def test_catalog_initial_states_and_count():
    catalog = load_catalog(ROOT / "config/scenarios/g4_golden.json")
    catalogs = [load_catalog(ROOT / "config/scenarios" / name) for name in
                ("g4_golden.json", "g4_frame_glance.json", "g4_robot_contacts.json")]
    shots = [shot for entry in catalogs for shot in entry["cases"]]
    assert len(shots) == 30
    assert len({shot["name"] for shot in shots}) == len(shots)
    assert catalog["repeats"] >= 2
    for shot in catalog["cases"]:
        validate_shot(shot)


@pytest.mark.parametrize('field,value', [('position',[0,0,.1]), ('velocity',[float('nan'),0,0]), ('duration',0), ('position',[10.9,1.54,.5])])
def test_bad_initial_state_rejected(field, value):
    shot = dict(position=[0,0,1],velocity=[1,0,0],angular_velocity=[0,0,0],duration=1)
    shot[field] = value
    with pytest.raises(ValueError):validate_shot(shot)


def test_contact_does_not_imply_goal_or_save():
    contact=dict(time=.1,kind='goal',geom='goal_right_left_post_collision',position=[11,1.54,1],ball_velocity_before=[8,0,0],ball_velocity_after=[-3,0,0])
    record=dict(shot={'name':'direct_goal'},trajectory=[dict(time=0,position=[9,1.54,1]),dict(time=.2,position=[10,1.54,1])],contacts=[contact])
    result=classify_record(record,GoalDetector)
    assert result['outcome']=='Timeout'
    assert result['frame_contact']
    assert result['robot_contact_is_save'] is None
    contact.update(kind='robot',geom='trunk_collision')
    result=classify_record(record,GoalDetector)
    assert result['robot_contact'] and result['outcome']=='Timeout'


def test_whole_ball_crossing_and_initial_outside_latched():
    def classify(points):
        return classify_record({'trajectory':[dict(time=i,position=p) for i,p in enumerate(points)]},GoalDetector)
    assert classify([[10,0,1],[11.05,0,1]])['outcome']=='Timeout'
    assert classify([[10,0,1],[11.2,0,1]])['outcome']=='Goal'
    assert classify([[12,3,1],[12,0,1]])['outcome']=='Miss'


def test_measured_csv_provenance_and_invalid_clock(tmp_path):
    path=tmp_path/'measured.csv'
    path.write_text('time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s\n0,9,0,1,8,0,0\n.3,11.4,0,1,8,0,0\n')
    samples,provenance=measured_csv_samples(path)
    assert len(samples)==2 and not provenance['physical_collision_response']
    assert len(provenance['sha256'])==64
    path.write_text(path.read_text().replace('.3,','0,'))
    with pytest.raises(ValueError):measured_csv_samples(path)


def test_contact_onset_and_bounce_are_independent():
    record = {'trajectory': [dict(time=0,position=[0,0,.111],velocity=[0,0,-1]),
                             dict(time=.001,position=[0,0,.111],velocity=[0,0,.3]),
                             dict(time=.002,position=[0,0,.111],velocity=[0,0,.29])],
              'contacts':[dict(time=0,kind='ground',geom='ground',position=[0,0,0],
                               ball_velocity_before=[0,0,-1],ball_velocity_after=[0,0,.3])],
              'raw_contact_events':[dict(type='CONTACT_FOUND',other_collider='/World/g0_ground',time_after_step=.001,points=[]),
                                    dict(type='CONTACT_LOST',other_collider='/World/g0_ground',time_after_step=.002,points=[])]}
    result=classify_record(record,GoalDetector)
    assert result['ground_contact_count']==result['ground_bounce_count']==1
    assert [e['type'] for e in result['events']]==['contact_begin','ground_bounce','contact_end','timeout']
    record['trajectory'][1]['velocity'][2] = -.9
    assert classify_record(record,GoalDetector)['ground_bounce_count']==0


def test_swept_frame_crossing_requires_native_evidence():
    spec=importlib.util.spec_from_file_location('g4_export',ROOT/'scripts/export_g4_trajectories.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record={'trajectory':[dict(time=0,position=[10.9,1.54,1]),dict(time=.001,position=[11.1,1.54,1])]}
    assert module.frame_crossing_audit(record)
    record['raw_contact_events']=[dict(other_collider='/World/g91_goal_right_right_post_collision',type='CONTACT_FOUND',time_after_step=.001)]
    assert not module.frame_crossing_audit(record)

    record['raw_contact_events'][0]['type']='CONTACT_LOST'
    assert module.frame_crossing_audit(record)
    record['raw_contact_events'][0].update(type='CONTACT_FOUND',other_collider='/World/g75_goal_left_left_post_collision')
    assert module.frame_crossing_audit(record)
