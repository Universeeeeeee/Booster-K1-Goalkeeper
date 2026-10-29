#!/usr/bin/env python3
"""Counterfactual renderer audit of G1 zero-opacity ground in a G6-only layer.

Preserves source assets and white ball. A is original G1 visual data; B changes
only ground visibility; C, only if necessary, binds the source turf RGB to an
explicit PreviewSurface to distinguish hidden geometry from ignored primvars.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--g1-usd',type=Path,required=True)
    parser.add_argument('--camera-usd',type=Path,required=True)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--mount-config',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    ctypes.CDLL('libcuda.so.1',mode=ctypes.RTLD_GLOBAL).cuInit(0)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':0,'physics_gpu':0,'multi_gpu':False,'fast_shutdown':True})
    completed=False
    try:
        import carb.settings
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from PIL import Image
        from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdShade
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from acquisition.g6_scene import G6Scene
        stage=omni.usd.get_context().get_stage()
        world=World(physics_dt=1/120,rendering_dt=1/60,stage_units_in_meters=1)
        scene=G6Scene(stage,args.g1_usd,args.camera_usd,{'direct_goal':args.trajectory},args.mount_config)
        light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(1000)
        world.reset()
        scene.author_state(.005)
        carb.settings.get_settings().set('/rtx/post/aa/op',0)
        product=rep.create.render_product(scene.left_camera_path,(960,600))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
        ground=stage.GetPrimAtPath('/World/G1/Bodies/b0_world/g0_ground')
        turf=stage.GetPrimAtPath('/World/G1/Bodies/b0_world/g1_turf')
        source_zero=[]
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/G1')):
            if prim.IsA(UsdGeom.Gprim):
                opacity=UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get()
                if opacity and max(opacity)==0:
                    source_zero.append(str(prim.GetPath()))
        helper_path=Path(sys.modules[G6Scene.__module__].__file__)
        report={'runtime_helper_path':str(helper_path), 'runtime_helper_sha256':hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                'diagnostic_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scene_provenance':scene.provenance(),'ground_display_opacity':list(UsdGeom.Gprim(ground).GetDisplayOpacityAttr().Get()),
                'ground_display_color':[list(c) for c in UsdGeom.Gprim(ground).GetDisplayColorAttr().Get()],
                'turf_display_color':[list(c) for c in UsdGeom.Gprim(turf).GetDisplayColorAttr().Get()],
                'zero_opacity_gprim_paths':source_zero,'variants':{}}
        def capture(name):
            for _ in range(40):world.render()
            image=np.array(rgb.get_data()).copy()
            Image.fromarray(image).save(args.output/f'{name}.png')
            roi=image[360:500,100:850,:3].astype(float)
            green=(roi[...,1]>roi[...,0]*1.15)&(roi[...,1]>roi[...,2]*1.15)&(roi[...,1]>20)
            value={'roi_bounds_xyxy':[100,360,850,500],'roi_median_rgb':np.median(roi,axis=(0,1)).tolist(),
                   'roi_green_fraction':float(green.mean())}
            report['variants'][name]=value
            print(name,json.dumps(value),flush=True)
            return value
        # Explicit A/B counterfactual remains reproducible after the G6 helper
        # learns the visibility adaptation: restore original inherited values.
        for path in source_zero:
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set('inherited')
        capture('A_original')
        UsdGeom.Imageable(ground).CreateVisibilityAttr().Set('invisible')
        b=capture('B_ground_invisible')
        if b['roi_green_fraction']<.5:
            material=UsdShade.Material.Define(stage,'/World/G6OpacityDiagnostic/TurfSourceColor')
            shader=UsdShade.Shader.Define(stage,'/World/G6OpacityDiagnostic/TurfSourceColor/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*report['turf_display_color'][0]))
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(turf).Bind(material)
            capture('C_explicit_source_turf_material')
        for path in source_zero:
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set('invisible')
        capture('D_all_source_alpha_zero_invisible')
        (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
        stage.GetRootLayer().Export(str(args.output/'diagnostic_layer.usda'))
        rgb.detach();product.destroy()
        completed=True
    except Exception:
        import traceback
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if not completed:
            import omni.kit.app
            omni.kit.app.get_app().post_quit(1)
        app.close()


if __name__=='__main__':main()
