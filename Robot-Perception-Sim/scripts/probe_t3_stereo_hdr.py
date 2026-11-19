"""Park-only native stereo HDR smoke; not exposure/performance acceptance."""
import argparse
import json
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--ae-frames',type=int,default=0);p.add_argument('--keep-center',action='store_true')
a,_=p.parse_known_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
import ctypes
cuda=ctypes.CDLL("libcuda.so.1",mode=ctypes.RTLD_GLOBAL)
if cuda.cuInit(0): raise RuntimeError("CUDA init failed")
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':1,'multi_gpu':False,'limit_cpu_threads':2,'extra_args':['--/rtx/hydra/supportMultiTickRate=false']})
try:
    import numpy as np
    import omni.replicator.core as rep
    from omni.replicator.core.scripts.utils import viewport_manager
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import World
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acquisition'))
    from t3_park import load_scene
    c=json.loads(Path(a.config).read_text());s=c['stereo'];w,h=s['width'],s['height'];k=s['K']
    stage,center,set_state,provenance=load_scene(a.scene,s['fps'])
    if not a.keep_center:
        stage.RemovePrim(center)
    products=[];annotators=[];readback={}
    center_cam=UsdGeom.Camera(stage.GetPrimAtPath(center))
    center_cam.CreateFocalLengthAttr(1.79)
    center_cam.CreateHorizontalApertureAttr(1.79*800/518.84)
    center_cam.CreateVerticalApertureAttr(1.79*600/518.84)
    readback['CenterRGB']={'path':center,'focal_length_mm':float(center_cam.GetFocalLengthAttr().Get()),'resolution':[800,600]}
    for name,y in [('Left',s['baseline_m']/2),('Right',-s['baseline_m']/2)]:
        path='/World/Cleaner/T3/'+name;cam=UsdGeom.Camera.Define(stage,path)
        cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.6,y,.78),Gf.Vec3d(1.6,y,.78),Gf.Vec3d(0,0,1)).GetInverse())
        focal_mm=s['focal_length_mm'];cam.CreateFocalLengthAttr(focal_mm);cam.CreateHorizontalApertureAttr(focal_mm*w/k[0][0]);cam.CreateVerticalApertureAttr(focal_mm*h/k[1][1]);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
        prim=cam.GetPrim();prim.ApplyAPI('OmniLensDistortionOpenCvPinholeAPI');prim.GetAttribute('omni:lensdistortion:model').Set('opencvPinhole')
        values={'fx':k[0][0],'fy':k[1][1],'cx':float(k[0][2]),'cy':float(k[1][2]),'imageSize':Gf.Vec2i(w,h)}
        values.update(zip(('k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4'),s['D']+[0.]*7))
        for key,value in values.items():
            if not prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+key).Set(value):raise RuntimeError(key)
        product=viewport_manager.get_render_product(path,(w,h),False,'T3'+name);products.append(product)
        annot=rep.AnnotatorRegistry.get_annotator('HdrColor',device='cuda');annot.attach(product.hydra_texture.get_render_product_path());annotators.append(annot)
        readback[name]={key:str(prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+key).Get()) for key in values}
    world=World(physics_dt=.01,rendering_dt=.01);world.reset();world.play()
    for _ in range(100):world.step(render=True)
    if a.ae_frames:
        from t3_exposure import integrate_linear, next_shared_exposure
        e=c['exposure']; exposure=e['initial_s']; records=[]
        # Static park fixture: repeated fresh renders approximate integration only
        # for a time-invariant scene. Not dynamic exposure/timestamp acceptance.
        for index in range(a.ae_frames):
            samples=[[],[]]
            for sample in range(e['subsamples']):
                world.step(render=True)
                for eye,annot in enumerate(annotators):
                    samples[eye].append(annot.get_data().numpy()[...,:3].astype(np.float32).copy())
            exposed=[integrate_linear(x,exposure,e['reference_exposure_s']) for x in samples]
            measured=[float((x @ np.array([.2126,.7152,.0722])).mean()) for x in exposed]
            updated=next_shared_exposure(*exposed,exposure,target=e['target'],minimum_s=e['min_s'],maximum_s=e['max_s'],max_step_ratio=e['max_step_ratio'])
            records.append({'index':index,'exposure_s':exposure,'eye_linear_mean':measured,'next_exposure_s':updated})
            if index in (0,a.ae_frames-1):
                for eye,x in zip(('left','right'),exposed):np.save(out/f'ae_{index}_{eye}.npy',x)
            exposure=updated
        (out/'ae_static.json').write_text(json.dumps({'scope':'static park shared AE only; no physical exposure timing validation','records':records},indent=2))
    results=[]
    for name,annot in zip(('left','right'),annotators):
        image=annot.get_data().numpy();np.save(out/(name+'_hdr.npy'),image)
        results.append({'eye':name,'shape':list(image.shape),'dtype':str(image.dtype),'finite':bool(np.isfinite(image).all()),'min':float(image.min()),'max':float(image.max()),'std':float(image[...,:3].astype(np.float32).std())})
    (out/'summary.json').write_text(json.dumps({'config':c,'provenance':provenance,'readback':readback,'images':results,'center_rgb_retained':bool(a.keep_center),'scope':'static HDR smoke only; no exposure integration, pixel geometry or timing acceptance'},indent=2))
finally:
    app.close()
