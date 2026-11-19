"""Park-only center HDR smoke; not exposure/performance acceptance."""
import argparse
import json
import hashlib
import os
import time
from pathlib import Path
import sys
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--ae-frames',type=int,default=0);p.add_argument('--light-steps',action='store_true');p.add_argument('--fast',action='store_true');p.add_argument('--fabric',action='store_true');p.add_argument('--pipeline',action='store_true');p.add_argument('--gpu-color',action='store_true');p.add_argument('--aa',type=int,choices=[0,3],default=3)
p.add_argument('--disable-viewports',action='store_true');p.add_argument('--async-write',action='store_true');p.add_argument('--async-render',action='store_true');p.add_argument('--gate-updates',action='store_true')
a,_=p.parse_known_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
import ctypes
cuda=ctypes.CDLL("libcuda.so.1",mode=ctypes.RTLD_GLOBAL)
if cuda.cuInit(0): raise RuntimeError("CUDA init failed")
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'active_gpu':int(os.environ.get('URBAN_GPU',1)),'multi_gpu':False,'limit_cpu_threads':8,'physics_gpu':0,'extra_args':['--/rtx/hydra/supportMultiTickRate=false','--/app/audio/enabled=false']})
try:
    import numpy as np
    import omni.replicator.core as rep
    from omni.replicator.core.scripts.utils import viewport_manager
    from pxr import Gf, UsdGeom
    from isaacsim.core.api import World
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acquisition'))
    from t3_park import load_scene
    c=json.loads(Path(a.config).read_text());s=c;w,h=s['width'],s['height'];k=[[s['fx'],0,s['cx']],[0,s['fy'],s['cy']],[0,0,1]]
    stage,center,set_state,provenance=load_scene(a.scene,s['fps']);stage.RemovePrim(center)
    products=[];annotators=[];readback={}
    for name,y in [('CenterRGB',0)]:
        path='/World/Cleaner/T3/'+name;cam=UsdGeom.Camera.Define(stage,path)
        cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.6,y,.78),Gf.Vec3d(1.6,y,.78),Gf.Vec3d(0,0,1)).GetInverse())
        focal_mm=c['focal_length_mm'];cam.CreateFocalLengthAttr(focal_mm);cam.CreateHorizontalApertureAttr(focal_mm*w/k[0][0]);cam.CreateVerticalApertureAttr(focal_mm*h/k[1][1]);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
        prim=cam.GetPrim();prim.ApplyAPI('OmniLensDistortionOpenCvPinholeAPI');prim.GetAttribute('omni:lensdistortion:model').Set('opencvPinhole')
        values={'fx':k[0][0],'fy':k[1][1],'cx':float(k[0][2]),'cy':float(k[1][2]),'imageSize':Gf.Vec2i(w,h)}
        values.update(zip(('k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4'),s['D']+[0.]*7))
        for key,value in values.items():
            if not prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+key).Set(value):raise RuntimeError(key)
        product=viewport_manager.get_render_product(path,(w,h),False,'T3'+name);products.append(product)
        annot=rep.AnnotatorRegistry.get_annotator('HdrColor',device='cuda');annot.attach(product.hydra_texture.get_render_product_path());annotators.append(annot)
        readback[name]={key:str(prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+key).Get()) for key in values}
    import cv2
    import carb.settings
    from pxr import Sdf
    settings=carb.settings.get_settings()
    settings.set('/rtx/post/motionblur/enabled',False)
    settings.set('/rtx/post/aa/op',a.aa);settings.set('/rtx-transient/post/aa/limitedOps',False)
    rp=stage.GetPrimAtPath(products[0].hydra_texture.get_render_product_path())
    rp.CreateAttribute('omni:rtx:post:aa:op',Sdf.ValueTypeNames.Token).Set('none' if a.aa==0 else 'dlss')
    settings.set('/rtx/dlss/frameGeneration',False)
    if a.fabric:
        settings.set('/app/hydra/renderSettings/useUsdAttributes',False)
        settings.set('/app/hydra/renderSettings/useFabricAttributes',True)
    from t3_park import FrameMarkers
    markers=FrameMarkers(stage,[path],(w,h))
    xy=(np.array(markers.centres)-[s['cx'],s['cy']])/[s['fx'],s['fy']]
    rr=(xy*xy).sum(1);rad=1+s['D'][0]*rr+s['D'][1]*rr**2
    centres=np.rint(xy*rad[:,None]*[s['fx'],s['fy']]+[s['cx'],s['cy']]).astype(int)
    render_settings={key:settings.get(key) for key in ['/rtx/rendermode','/rtx/dlss/frameGeneration','/rtx/post/aa/op','/rtx/directLighting/sampledLighting/denoisingTechnique','/rtx/indirectDiffuse/denoiser/enabled','/rtx/reflections/denoiser/enabled']}
    (out/'render_settings.json').write_text(json.dumps(render_settings,indent=2))
    world=World(physics_dt=.01,rendering_dt=.01);world.reset();world.play()
    import omni.kit.viewport.window as vw
    windows=list(vw.get_viewport_window_instances())
    if a.disable_viewports:
        for window in windows:window.viewport_api.updates_enabled=False
    if a.async_render:
        settings.set('/app/asyncRendering',True)
        settings.set('/omni/replicator/asyncRendering',True)
    def inventory():
        return {'optics':{'focal_mm':cam.GetFocalLengthAttr().Get(),'horizontal_aperture_mm':cam.GetHorizontalApertureAttr().Get(),'vertical_aperture_mm':cam.GetVerticalApertureAttr().Get()},'render_products':[str(x.GetPath()) for x in stage.Traverse() if x.GetTypeName()=='RenderProduct'],'viewports':[{'name':str(x.name),'updates':bool(x.viewport_api.updates_enabled)} for x in windows], 'settings':{k:settings.get(k) for k in ['/app/asyncRendering','/omni/replicator/asyncRendering','/omni/replicator/RTSubframes','/rtx/post/aa/op','/rtx/dlss/frameGeneration']},'streaming_extensions':[x['id'] for x in app._app.get_extension_manager().get_extensions() if ('webrtc' in x['id'].lower() or 'livestream' in x['id'].lower()) and x.get('enabled')]}
    for _ in range(100):world.step(render=True)
    (out/'inventory_start.json').write_text(json.dumps(inventory(),indent=2))
    import warp as wp
    if a.gpu_color:
        from t3_gpu_color import convert
    e=c['exposure'];exposure=e['initial_s'];rows=[];start=time.monotonic();raw=(out/'frames.rgb').open('wb')
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    writer=ThreadPoolExecutor(max_workers=1);futures=deque();queue_peak=0;step_calls=0
    def write_frame(payload,row):
        begin=time.monotonic_ns();raw.write(payload);row['sha256']=hashlib.sha256(payload).hexdigest();row['write_complete_ns']=time.monotonic_ns();row['worker_ms']=(row['write_complete_ns']-begin)/1e6
    target=a.ae_frames or 300;pending={}
    for i in range(target+(4 if a.pipeline else 0)):
        t0=time.perf_counter();state=set_state(i);markers.set(i);pending[i]=state;t1=time.perf_counter()
        # Fixed same-state flushing in all variants; never silently equate renderer tick to sensor time.
        for _ in range(1 if a.pipeline else 3):
            if a.gate_updates:products[0].hydra_texture.set_updates_enabled(True)
            world.step(render=True);step_calls+=1
            if a.gate_updates:products[0].hydra_texture.set_updates_enabled(False)
        t2=time.perf_counter()
        gpu=annotators[0].get_data()
        if a.gpu_color:
            dst=wp.empty((h,w,3),dtype=wp.uint8,device=gpu.device);meter=wp.zeros(1,dtype=float,device=gpu.device)
            wp.launch(convert,dim=(h,w),inputs=[gpu,dst,meter,exposure/e['reference_exposure_s']],device=gpu.device)
            image=dst.numpy();brightness=float(meter.numpy()[0])/((h-120)*w)
        else:
            image=gpu.numpy().astype(np.float32)
        t3=time.perf_counter()
        cells=np.array([image[v-2:v+3,u-2:u+3,:3].mean() for u,v in centres]);bits=cells>(cells[0]+cells[1])/2
        token=int(np.dot(bits[4:20],2**np.arange(15,-1,-1)))
        if a.pipeline and token==65535 and not rows:continue
        if token!=len(rows) or int(bits[20])!=int(bits[4:20].sum()%2):raise RuntimeError(f'Pixel mismatch expected {len(rows)}, got {token}')
        state=pending.pop(token)
        if a.gpu_color:
            payload=image.tobytes()
        else:
            linear=np.ascontiguousarray(image[...,:3])*(exposure/e['reference_exposure_s'])
            crop=linear[120:]
            if a.fast:
                means=cv2.mean(crop);brightness=means[0]*.2126+means[1]*.7152+means[2]*.0722
                v=np.clip(linear,0,1);powered=cv2.pow(v,1/2.4);srgb=1.055*powered-.055;np.copyto(srgb,12.92*v,where=v<=.0031308)
            else:
                brightness=float((crop @ np.array([.2126,.7152,.0722])).mean())
                v=np.clip(linear,0,1);srgb=np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)
            payload=np.rint(srgb*255).astype(np.uint8).tobytes()
        next_exposure=float(np.clip(exposure*np.clip(.45/max(brightness,1e-12),.5,2),.0001,.008));t4=time.perf_counter()
        t5=time.perf_counter();digest=None
        rows.append({'index':token,'pixel_id':token,'simulation_s':token/s['fps'],'exposure_s':exposure,'brightness':brightness,'sha256':digest,'state_ms':(t1-t0)*1000,'render_ms':(t2-t1)*1000,'read_ms':(t3-t2)*1000,'process_ms':(t4-t3)*1000,'write_hash_ms':(t5-t4)*1000,'state':state})
        if a.async_write:
            if len(futures)>=32:futures.popleft().result()
            futures.append(writer.submit(write_frame,payload,rows[-1]));queue_peak=max(queue_peak,len(futures))
        else:write_frame(payload,rows[-1])
        rows[-1]['arrival_ns']=time.monotonic_ns()
        exposure=next_exposure
        if len(rows)==target:break
    if len(rows)!=target:raise RuntimeError('Missing output frames')
    for future in futures:future.result()
    writer.shutdown(wait=True)
    (out/'inventory_end.json').write_text(json.dumps(inventory(),indent=2))
    raw.flush();os.fsync(raw.fileno());raw.close();elapsed=time.monotonic()-start
    (out/'benchmark.json').write_text(json.dumps({'step_calls':step_calls,'queue_peak':queue_peak,'async_write':a.async_write,'async_render_requested':a.async_render,'frames':len(rows),'elapsed_s':elapsed,'fps':len(rows)/elapsed,'fast':a.fast,'fabric':a.fabric,'pipeline':a.pipeline,'gpu_color':a.gpu_color,'config':c,'rows':rows,'scope':'single HDR; exposure applied in source-ordered CPU processing; diagnostic strip; no temporal integration; explicit source times separate from renderer clock'},indent=2))
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
