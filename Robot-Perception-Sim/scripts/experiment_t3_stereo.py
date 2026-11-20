"""Three-camera park capture; sensor time is distinct from renderer flush ticks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--center-config', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['A', 'B', 'C'], required=True)
    p.add_argument('--frames', type=int, default=1000)
    a, _ = p.parse_known_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import ctypes
    cuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    if cuda.cuInit(0): raise RuntimeError("CUDA initialization failed")
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': 1, 'physics_gpu': 0,
        'multi_gpu': False, 'limit_cpu_threads': 8,
        'extra_args': ['--/rtx/hydra/supportMultiTickRate=false', '--/app/audio/enabled=false']})
    try:
        capture(app, a)
    finally:
        app.close()


def capture(app, a):
    import cv2
    import numpy as np
    import carb.settings
    import omni.replicator.core as rep
    from omni.replicator.core.scripts.utils import viewport_manager
    from isaacsim.core.api import World
    from pxr import Gf, Sdf, UsdGeom
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'acquisition'))
    from t3_park import load_scene, FrameMarkers
    c = json.loads(Path(a.config).read_text())['stereo']
    center = json.loads(Path(a.center_config).read_text())
    stage, center_path, _, provenance = load_scene(a.scene, 100)
    body = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Cleaner'))
    body_op = body.GetOrderedXformOps()[0]
    settings = carb.settings.get_settings()
    settings.set('/rtx/post/motionblur/enabled', False)
    settings.set('/rtx/post/aa/op', 3)
    settings.set('/rtx/dlss/frameGeneration', False)
    settings.set('/rtx-transient/post/aa/limitedOps', False)
    products, annotations, markers, centres, optics = {}, {}, {}, {}, {}
    for name, offset in [('CenterRGB', 0), ('Left', .035), ('Right', -.035)]:
        w, h = (center['width'], center['height']) if name == 'CenterRGB' else (c['width'], c['height'])
        k = [[center['fx'], 0, center['cx']], [0, center['fy'], center['cy']], [0, 0, 1]] if name == 'CenterRGB' else c['K']
        d = center['D'] if name == 'CenterRGB' else c['D']
        path = '/World/Cleaner/T3/' + name
        cam = UsdGeom.Camera.Define(stage, path)
        xf = UsdGeom.Xformable(cam); xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.6, offset, .78), Gf.Vec3d(1.6, offset, .78), Gf.Vec3d(0, 0, 1)).GetInverse())
        cam.CreateFocalLengthAttr(1.79)
        cam.CreateHorizontalApertureAttr(1.79*w/k[0][0]); cam.CreateVerticalApertureAttr(1.79*h/k[1][1])
        cam.CreateClippingRangeAttr(Gf.Vec2f(.01, 100))
        prim = cam.GetPrim(); prim.ApplyAPI('OmniLensDistortionOpenCvPinholeAPI')
        prim.GetAttribute('omni:lensdistortion:model').Set('opencvPinhole')
        values = {'fx': k[0][0], 'fy': k[1][1], 'cx': float(k[0][2]), 'cy': float(k[1][2]), 'imageSize': Gf.Vec2i(w,h)}
        values.update(zip(('k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4'), d+[0.]*7))
        for key, value in values.items(): prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+key).Set(value)
        product = viewport_manager.get_render_product(path, (w,h), False, 'ABC'+name)
        rp = stage.GetPrimAtPath(product.hydra_texture.get_render_product_path())
        rp.CreateAttribute('omni:rtx:post:aa:op', Sdf.ValueTypeNames.Token).Set('dlss')
        annot = rep.AnnotatorRegistry.get_annotator('HdrColor', device='cuda'); annot.attach(product.hydra_texture.get_render_product_path())
        products[name], annotations[name] = product, annot
        markers[name] = FrameMarkers(stage, [path], (w,h))
        xy = (np.array(markers[name].centres)-[k[0][2],k[1][2]])/[k[0][0],k[1][1]]
        rr = (xy*xy).sum(1); radial = 1+d[0]*rr+d[1]*rr**2+d[4]*rr**3
        centres[name] = np.rint(xy*radial[:,None]*[k[0][0],k[1][1]]+[k[0][2],k[1][2]]).astype(int).tolist()
        optics[name] = {'resolution':[w,h], 'K':k, 'D':d, 'focal_attribute':cam.GetFocalLengthAttr().Get(),
            'local_matrix':np.array(xf.GetLocalTransformation()).T.tolist()}
    world = World(physics_dt=.01, rendering_dt=.01); world.reset(); world.play()
    for _ in range(100): world.step(render=True)
    stage.GetRootLayer().Export(str(a.output/'scene.usda'))
    meta = {'mode':a.mode, 'frames':a.frames, 'optics':optics, 'centres':centres, 'provenance':provenance,
        'sample_period_ns':10000000, 'gray':'linear HDR luminance quantized as round(clip(Y,0,1)*255); fixed gain=1',
        'sd':'signed int16 up/right',
        'sd_encoding':'half_trunc',
        'timing':'explicit sensor sample clock; renderer ticks include same-state flushing, not sensor samples',
        'render_settings':{k:settings.get(k) for k in ['/rtx/rendermode','/rtx/post/aa/op','/rtx/dlss/frameGeneration']}}
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    streams = {n:(a.output/(n+'.rgb')).open('wb') for n in annotations}
    if a.mode != 'A':
        for n in ['Left','Right']:
            streams[n+'_gray']=(a.output/(n+'.gray')).open('wb')
            streams[n+'_sd']=(a.output/(n+'.sd')).open('wb')
    start=time.perf_counter()
    with (a.output/'frames.jsonl').open('w') as log:
        for i in range(a.frames):
            begin=time.perf_counter(); t=i*.01
            xyz=[-48+1.5*t if a.mode=='C' else -48,-8.375,0]
            body_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*xyz)))
            for m in markers.values():m.set(i)
            ids={}; images={}
            for attempt in range(12):
                world.step(render=True)
                for n,annot in annotations.items():
                    img=annot.get_data().numpy().astype(np.float32)
                    cells=np.array([img[y-2:y+3,x-2:x+3,:3].mean() for x,y in centres[n]])
                    bits=cells>(cells[0]+cells[1])/2
                    token=int(np.dot(bits[4:20],2**np.arange(15,-1,-1)))
                    valid=list(bits[[0,1,2,3,21,22]])==[True,False,True,False,False,True] and bits[20]==bool(bits[4:20].sum()%2)
                    ids[n]=token if valid else -1; images[n]=img
                if all(v==i for v in ids.values()):break
            else:raise RuntimeError(f'frame identity failed {i}: {ids}')
            rendered=time.perf_counter(); row={'index':i,'simulation_ns':i*10000000,'pixel_ids':ids,
                'eye_timestamp_ns':{n:ids[n]*10000000 for n in ids},'position':list(UsdGeom.XformCache().GetLocalToWorldTransform(body.GetPrim()).ExtractTranslation()),
                'renderer_s':world.current_time,'render_ticks':attempt+1,'render_read_ms':(rendered-begin)*1000,'hashes':{}}
            payloads={}; gray_ms=0.; sd_ms=0.
            for n,img in images.items():
                v=np.clip(img[...,:3],0,1);rgb=np.rint(np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)*255).astype(np.uint8)
                payloads[n]=rgb.tobytes()
                if a.mode!='A' and n!='CenterRGB':
                    b=time.perf_counter(); gray=np.rint(np.clip(img[...,:3] @ np.array([.2126,.7152,.0722],np.float32),0,1)*255).astype(np.uint8);gray_ms+=(time.perf_counter()-b)*1000
                    b=time.perf_counter()
                    up=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,-1,0],[0,1,0],[0,0,0]],np.float32));up[0]=0
                    right=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,0,0],[0,1,-1],[0,0,0]],np.float32));right[:,-1]=0
                    diff=np.trunc(np.stack([up, right]) * np.float32(0.5)).astype(np.int16)
                    sd_ms+=(time.perf_counter()-b)*1000
                    payloads[n+'_gray']=gray.tobytes()
                    payloads[n+'_sd']=diff.tobytes()
            processed=time.perf_counter()
            for n,payload in payloads.items():streams[n].write(payload);row['hashes'][n]=hashlib.sha256(payload).hexdigest()
            row.update(gray_ms=gray_ms,sd_ms=sd_ms,process_ms=(processed-rendered)*1000,total_ms=(time.perf_counter()-begin)*1000)
            log.write(json.dumps(row)+'\n')
            if i%100==0:print(a.mode,i,flush=True)
    for stream in streams.values():stream.flush();os.fsync(stream.fileno());stream.close()
    meta['elapsed_s']=time.perf_counter()-start
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
