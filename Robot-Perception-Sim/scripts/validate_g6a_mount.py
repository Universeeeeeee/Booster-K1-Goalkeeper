#!/usr/bin/env python3
"""USD-only two-pose full scene / official ZED mount and frozen replay audit."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acquisition.g6_scene import G6Scene,HEAD_POSES


def main():
    from pxr import Usd
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g1-usd',type=Path,required=True)
    parser.add_argument('--camera-usd',type=Path,required=True)
    parser.add_argument('--direct',type=Path,required=True)
    parser.add_argument('--bounce',type=Path,required=True)
    parser.add_argument('--mount-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    stage=Usd.Stage.CreateInMemory()
    scene=G6Scene(stage,args.g1_usd,args.camera_usd,{'direct_goal':args.direct,'one_bounce':args.bounce},args.mount_config)
    poses={}
    for pose in HEAD_POSES:
        scene.select_head_pose(pose)
        camera=scene.camera_state()
        stereo=np.asarray(camera['T_left_from_right'])
        assert np.allclose(stereo[:3,:3],np.eye(3),atol=2e-6)
        assert np.allclose(stereo[:3,3],[.12,0,0],atol=2e-6)
        assert abs(camera['baseline_m']-.12)<2e-6
        for key in ('T_world_left_optical','T_world_right_optical'):
            transform=np.asarray(camera[key]);assert np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=2e-6)
        trajectories={}
        for name in ('direct_goal','one_bounce'):
            scene.select_trajectory(name)
            trajectories[name]=[scene.author_state(t) for t in (0.,.05,.1,scene.duration_s)]
        poses[pose]={'camera':camera,'trajectory_samples':trajectories}
    result={'provenance':scene.provenance(),'poses':poses,'checks_passed':True,
            'validation_runtime':'USD-only; no GPU renderer or new physical simulation',
            'visibility_limit':'finite sampled Mesh triangle rays; renderer confirmation remains independent'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checks_passed':True,'poses':{k:v['camera'] for k,v in poses.items()}}),flush=True)


if __name__=='__main__':main()
