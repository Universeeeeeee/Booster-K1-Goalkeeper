#!/usr/bin/env python3
"""Independently read actual USD world transforms for all G6-A replay samples.

USD-only: no SimulationApp, renderer, physics stepping, or detector. Expected
samples are chosen independently from the frozen JSON; actual translation and
quaternion come from a new USD XformCache after author_state, plus serialized
stage reloads. The frozen helper is imported without modification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acquisition.g6_scene import G6Scene,HEAD_POSES,OPTICAL_FROM_USD


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intended_requests(duration, exposure_ms, fps=60, samples=5):
    """Independent schedule from the frozen G6-A protocol, not helper code."""
    centers=[]
    index=0
    while .005+index/fps <= duration-.005+1e-12:
        centers.append(.005+index/fps)
        index+=1
    if duration-.005-centers[-1] > 1e-9:
        centers.append(duration-.005)
    for target,center in enumerate(centers):
        yield target,'midpoint',None,center
        if exposure_ms==0:
            yield target,'subsample',0,center
        else:
            for sample in range(samples):
                yield target,'subsample',sample,center-exposure_ms/2000+(sample+.5)*exposure_ms/1000/samples


def main():
    from pxr import Gf,Usd,UsdGeom
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g1-usd',type=Path,required=True)
    parser.add_argument('--camera-usd',type=Path,required=True)
    parser.add_argument('--direct',type=Path,required=True)
    parser.add_argument('--bounce',type=Path,required=True)
    parser.add_argument('--mount-config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    stage=Usd.Stage.CreateInMemory()
    scene=G6Scene(stage,args.g1_usd,args.camera_usd,{'direct_goal':args.direct,'one_bounce':args.bounce},args.mount_config)
    def actual_world(current_stage,path):
        return np.asarray(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(current_stage.GetPrimAtPath(path))).T.copy()
    def read_quaternion(column_matrix):
        q=Gf.Transform(Gf.Matrix4d(column_matrix.T.tolist())).GetRotation().GetQuat()
        result=np.r_[q.GetReal(),np.asarray(q.GetImaginary())]
        return result/np.linalg.norm(result)
    static_paths=[p for p in scene.frozen_body_paths if p not in (scene.ball_path,scene.yaw_path,scene.pitch_path)]
    assert len(static_paths)==21,static_paths
    fixed_reference={p:actual_world(stage,p) for p in static_paths}
    summary={'checks_passed':False,'runtime_domain':'USD-only schema/world-transform and serialized-stage reload',
             'renderer_used':False,'physics_stepped':False,'source_helper_path':str(sys.modules[G6Scene.__module__].__file__),
             'source_helper_sha256':digest(sys.modules[G6Scene.__module__].__file__),
             'validator_sha256':digest(__file__),'scene_provenance':scene.provenance(),
             'static_robot_body_paths':static_paths,'conditions':[]}
    trace=args.output_dir/'states.jsonl'
    totals={'translation_error_m':0.,'quaternion_sign_invariant_l2_error':0.,'rotation_matrix_error':0.,'other_body_matrix_change':0.,
            'camera_state_matrix_error':0.,'stereo_matrix_error':0.,'serialization_matrix_error':0.}
    with trace.open('x') as stream:
        for pose in HEAD_POSES:
            scene.select_head_pose(pose)
            for name in ('direct_goal','one_bounce'):
                scene.select_trajectory(name)
                # Parse expected data again: no use of helper's nearest_sample or
                # its cached samples to select the independent expected state.
                source=json.loads(scene.trajectory_path.read_text())['trajectory']
                for exposure_ms in (0,10):
                    condition={'trajectory':name,'head_pose':pose,'exposure_ms':exposure_ms,
                               'midpoints':0,'subsamples':0,'checked_requests':0,'maximum_quantization_error_s':0.}
                    max_errors={key:0. for key in totals}
                    for target,kind,subsample,requested in intended_requests(source[-1]['time'],exposure_ms):
                        ordinal,expected=min(enumerate(source),key=lambda item:(abs(item[1]['time']-requested),item[0]))
                        returned=scene.author_state(requested)
                        actual=actual_world(stage,scene.ball_path)
                        q_actual=read_quaternion(actual)
                        q_expected=np.asarray(expected.get('quaternion_wxyz',[1.,0.,0.,0.]),dtype=float)
                        q_expected/=np.linalg.norm(q_expected)
                        translation_error=float(np.max(abs(actual[:3,3]-expected['position'])))
                        rotation_error=float(min(np.linalg.norm(q_actual-q_expected),np.linalg.norm(q_actual+q_expected)))
                        w,x,y,z=q_expected
                        expected_rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                        matrix_rotation_error=float(np.max(abs(actual[:3,:3]-expected_rotation)))
                        static_change=max(float(np.max(abs(actual_world(stage,p)-fixed_reference[p]))) for p in static_paths)
                        left=actual_world(stage,scene.left_camera_path)@OPTICAL_FROM_USD
                        right=actual_world(stage,scene.right_camera_path)@OPTICAL_FROM_USD
                        camera_error=float(np.max(abs(left-np.asarray(returned['T_world_camera']))))
                        stereo=np.linalg.inv(left)@right
                        target_stereo=np.eye(4);target_stereo[0,3]=.12
                        stereo_error=float(np.max(abs(stereo-target_stereo)))
                        assert returned['source_sample_index']==ordinal
                        assert returned['realized_source_time_s']==expected['time']
                        assert np.array_equal(np.asarray(returned['ball_world_position_m']),np.asarray(expected['position']))
                        assert translation_error<1e-10,(pose,name,requested,translation_error)
                        assert rotation_error<1e-6,(pose,name,requested,rotation_error)
                        assert matrix_rotation_error<2e-6,(pose,name,requested,matrix_rotation_error)
                        assert static_change<1e-10,(pose,name,requested,static_change)
                        assert camera_error<1e-10
                        assert stereo_error<2e-6
                        for key,error in (('translation_error_m',translation_error),('quaternion_sign_invariant_l2_error',rotation_error),('rotation_matrix_error',matrix_rotation_error),
                                          ('other_body_matrix_change',static_change),('camera_state_matrix_error',camera_error),('stereo_matrix_error',stereo_error)):
                            max_errors[key]=max(max_errors[key],error)
                        quantization=abs(expected['time']-requested)
                        condition['maximum_quantization_error_s']=max(condition['maximum_quantization_error_s'],quantization)
                        condition['checked_requests']+=1
                        condition['midpoints'if kind=='midpoint'else'subsamples']+=1
                        stream.write(json.dumps({'trajectory':name,'head_pose':pose,'exposure_ms':exposure_ms,
                            'target_index':target,'kind':kind,'subsample_index':subsample,'requested_time_s':requested,
                            'realized_time_s':expected['time'],'expected_source_index':ordinal,
                            'expected_position_world_m':expected['position'],'actual_usd_ball_world_matrix':actual.tolist(),
                            'expected_quaternion_wxyz':q_expected.tolist(),'actual_usd_quaternion_wxyz':q_actual.tolist(),
                            'translation_error_m':translation_error,'quaternion_sign_invariant_l2_error':rotation_error,'rotation_matrix_error':matrix_rotation_error,
                            'other_body_max_matrix_change':static_change,'actual_usd_T_world_left_optical':left.tolist(),
                            'camera_state_matrix_error':camera_error,'stereo_matrix_error':stereo_error})+'\n')
                    snapshot=args.output_dir/f'{name}_{pose}_e{exposure_ms}.usda'
                    stage.GetRootLayer().Export(str(snapshot))
                    reopened=Usd.Stage.Open(str(snapshot))
                    serialized_error=max(float(np.max(abs(actual_world(reopened,p)-actual_world(stage,p))))
                                         for p in [scene.ball_path,scene.left_camera_path,scene.right_camera_path,*static_paths])
                    assert serialized_error<1e-10
                    max_errors['serialization_matrix_error']=serialized_error
                    condition.update({'max_errors':max_errors,'serialized_stage':snapshot.name,
                                      'serialized_stage_sha256':digest(snapshot),'source_trajectory_sha256':digest(scene.trajectory_path)})
                    summary['conditions'].append(condition)
                    for key,value in max_errors.items():totals[key]=max(totals[key],value)
                    print('READBACK_CONDITION',pose,name,exposure_ms,condition['checked_requests'],max_errors,flush=True)
    summary.update({'checks_passed':True,'max_errors':totals,'checked_requests':sum(c['checked_requests']for c in summary['conditions']),
                    'condition_count':len(summary['conditions']),'trace_sha256':digest(trace),
                    'source_schedule':'60Hz centers .005+j/60 plus duration-.005 endpoint;0ms midpoint,10ms5 midpoint-quadrature subsamples',
                    'limitations':'USD acceptance proof; independent rendered/SDK image evidence is separate'})
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('READBACK_PASS',json.dumps({'requests':summary['checked_requests'],'max_errors':totals}),flush=True)


if __name__=='__main__':main()
