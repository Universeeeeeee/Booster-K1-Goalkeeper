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
    p.add_argument('--frames', type=int, default=300)
    p.add_argument('--stage', choices=['render','gpu','cpu','sync_copy','sd','save'], required=True)
    p.add_argument('--eyes', type=int, choices=[2,3], required=True)
    p.add_argument('--gpu-process', action='store_true')
    p.add_argument('--ticks', type=int, default=3)
    p.add_argument('--no-viewport', action='store_true')
    p.add_argument('--async-save', action='store_true')
    p.add_argument('--clock-mode', choices=['legacy','C1'], default='legacy')
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
    except BaseException:
        import traceback
        (a.output/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        app.close()


def capture(app, a):
    import cv2
    import numpy as np
    import carb.settings
    import omni.replicator.core as rep
    from omni.replicator.core.scripts.utils import viewport_manager
    from isaacsim.core.api import World
    import omni.timeline
    from isaacsim.core.rendering_manager import RenderingManager
    from pxr import Gf, Sdf, UsdGeom
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'acquisition'))
    from t3_park import load_scene, FrameMarkers
    c = json.loads(Path(a.config).read_text())['stereo']
    center = json.loads(Path(a.center_config).read_text())
    stage, center_path, _, provenance = load_scene(a.scene, 100)
    body = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Cleaner'))
    body_op = body.GetOrderedXformOps()[0]
    timeline = omni.timeline.get_timeline_interface()
    settings = carb.settings.get_settings()
    settings.set('/rtx/post/motionblur/enabled', False)
    settings.set('/rtx/post/aa/op', 3)
    settings.set('/rtx/dlss/frameGeneration', False)
    settings.set('/rtx-transient/post/aa/limitedOps', False)
    products, annotations, markers, centres, optics = {}, {}, {}, {}, {}
    for name, offset in ([('CenterRGB', 0)] if a.eyes==3 else []) + [('Left', .035), ('Right', -.035)]:
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
        annot = None
        if a.stage != 'render':
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
    base_simulation_s = float(world.current_time)
    base_timeline_s = float(timeline.get_current_time())
    stage.GetRootLayer().Export(str(a.output/'scene.usda'))
    meta = {'mode':a.mode, 'frames':a.frames, 'optics':optics, 'centres':centres, 'provenance':provenance,
        'sample_period_ns':10000000, 'gray':'linear HDR luminance quantized as round(clip(Y,0,1)*255); fixed gain=1',
        'sd':'signed int16 up/right; same gray input as supplied NumPy reference',
        'timing':'explicit sensor sample clock; renderer ticks include same-state flushing, not sensor samples',
        'render_settings':{k:settings.get(k) for k in ['/rtx/rendermode','/rtx/post/aa/op','/rtx/dlss/frameGeneration']}}
    (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
    import warp as wp
    from omni.hydra.engine.stats import HydraEngineStats
    hydra = HydraEngineStats()
    import omni.kit.viewport.window as windows
    viewport_inventory = [{'name':str(v.name),'updates':v.viewport_api.updates_enabled} for v in windows.get_viewport_window_instances()]
    # Keep the prior B viewport workload; inventory it rather than changing it silently.
    streams = {}
    timing_log = (a.output / 'timing.jsonl').open('w')
    if a.stage=='save':
        for n in annotations:
            streams[n]=(a.output/(n+'.rgb')).open('wb')
            if n!='CenterRGB':
                streams[n+'_gray']=(a.output/(n+'.gray')).open('wb')
                streams[n+'_sd']=(a.output/(n+'.sd')).open('wb')
    if a.no_viewport:
        for v in windows.get_viewport_window_instances():v.viewport_api.updates_enabled=False
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    writer=ThreadPoolExecutor(max_workers=1);pending=deque()
    def save(payloads,hashes):
        for n,payload in payloads.items():streams[n].write(payload);hashes[n]=hashlib.sha256(payload).hexdigest()
    from t3_gpu_sd import images as gpu_images, differences
    allocated={}
    if a.gpu_process:
        for n,ann in annotations.items():
            buf=ann.get_data();h,w=buf.shape[:2]
            allocated[n]=(wp.empty((h,w,3),dtype=wp.uint8,device=buf.device),wp.empty((h,w),dtype=wp.uint8,device=buf.device),wp.empty((2,h,w),dtype=wp.int16,device=buf.device))
            wp.launch(gpu_images,(h,w),inputs=[buf,*allocated[n][:2]],device=buf.device)
            wp.launch(differences,(h,w),inputs=[allocated[n][1],allocated[n][2]],device=buf.device)
        wp.synchronize()
    samples=[]; gpu_samples=[]; example=None
    start=time.time(); begin_all=time.perf_counter()
    for i in range(a.frames):
        begin=time.perf_counter()
        source_time_s = i * 0.01
        body_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(-48 + 1.5 * source_time_s, -8.375, 0)))
        simulation_before_s = float(world.current_time) - base_simulation_s
        timeline_before_s = float(timeline.get_current_time()) - base_timeline_s
        for m in markers.values():m.set(i)
        tick_times=[]
        b=time.perf_counter();world.step(render=True);tick_times.append((time.perf_counter()-b)*1000)
        nodes=hydra.get_gpu_profiler_result()
        gpu_samples.append([sum(n.get('duration',0) for n in dev if n.get('indent',0)==0) for dev in nodes])
        if nodes and example is None:example=nodes
        refresh_count=0; refresh_sim_deltas=[]; refresh_timeline_deltas=[]
        if a.clock_mode == 'C1' and a.stage != 'render':
            # RenderingManager refreshes the products while the Timeline is paused.
            # Physics has already advanced exactly once for this sensor sample.
            for _ in range(8):
                before_refresh_sim=float(world.current_time)-base_simulation_s
                before_refresh_t=float(timeline.get_current_time())-base_timeline_s
                timeline.pause(); b=time.perf_counter(); RenderingManager.render(); tick_times.append((time.perf_counter()-b)*1000); timeline.play()
                refresh_count += 1
                refresh_sim_deltas.append(float(world.current_time)-base_simulation_s-before_refresh_sim)
                refresh_timeline_deltas.append(float(timeline.get_current_time())-base_timeline_s-before_refresh_t)
                # The final image-ID check below is performed after GPU conversion.
                # Continue flushing until the annotator has a non-stale buffer.
                probe={n:ann.get_data() for n,ann in annotations.items()}
                if a.gpu_process:
                    probe_ids={}
                    for n,buf in probe.items():
                        rgb_gpu=allocated[n][0]; wp.launch(gpu_images,(buf.shape[0],buf.shape[1]),inputs=[buf,rgb_gpu,allocated[n][1]],device=buf.device); wp.synchronize_device(buf.device)
                        rgb_probe=rgb_gpu.numpy(); cells=np.array([rgb_probe[y-2:y+3,x-2:x+3].mean() for x,y in centres[n]]);bits=cells>(cells[0]+cells[1])/2; probe_ids[n]=int(np.dot(bits[4:20],2**np.arange(15,-1,-1)))
                else:
                    probe_ids={}
                    for n,buf in probe.items():
                        img=buf.numpy().astype(np.float32);cells=np.array([img[y-2:y+3,x-2:x+3,:3].mean() for x,y in centres[n]]);bits=cells>(cells[0]+cells[1])/2;probe_ids[n]=int(np.dot(bits[4:20],2**np.arange(15,-1,-1)))
                if all(v == i for v in probe_ids.values()): break
        else:
            for _ in range(a.ticks - 1):
                b=time.perf_counter();world.step(render=True, step_sim=False);tick_times.append((time.perf_counter()-b)*1000)
            nodes=hydra.get_gpu_profiler_result()
            gpu_samples.append([sum(n.get('duration',0) for n in dev if n.get('indent',0)==0) for dev in nodes])
            if nodes and example is None:example=nodes
        after_render=time.perf_counter()
        buffers={}; arrays={}; sync_ms=0.; copy_ms=0.; gray_ms=0.; sd_ms=0.; color_ms=0.; save_ms=0.
        if a.stage!='render':
            buffers={n:ann.get_data() for n,ann in annotations.items()}
            assert all(b.device.is_cuda for b in buffers.values())
            if i == 0:
                # The first Render Product buffer can still contain the diagnostic
                # initialization ID. Flush only the renderer; never advance physics.
                for _ in range(20):
                    world.step(render=True, step_sim=False)
                buffers={n:ann.get_data() for n,ann in annotations.items()}
        after_get=time.perf_counter()
        if a.stage in ['cpu','sync_copy','sd','save'] and not a.gpu_process:
            if a.stage=='sync_copy':
                b=time.perf_counter();wp.synchronize_device(next(iter(buffers.values())).device);sync_ms=(time.perf_counter()-b)*1000
            b=time.perf_counter();arrays={n:buf.numpy() for n,buf in buffers.items()};copy_ms=(time.perf_counter()-b)*1000
        payloads={}; ids={}; cast_ms=0.;id_ms=0.;srgb_ms=0.;pack_ms=0.;gpu_process_ms=0.
        if a.gpu_process:
            b=time.perf_counter()
            for n,buf in buffers.items():
                rgb_gpu,gray_gpu,sd_gpu=allocated[n];h,w=buf.shape[:2]
                wp.launch(gpu_images,(h,w),inputs=[buf,rgb_gpu,gray_gpu],device=buf.device)
                if n!='CenterRGB':wp.launch(differences,(h,w),inputs=[gray_gpu,sd_gpu],device=buf.device)
            wp.synchronize_device(next(iter(buffers.values())).device)
            gpu_process_ms=(time.perf_counter()-b)*1000
            for n,(rgb_gpu,gray_gpu,sd_gpu) in allocated.items():
                b=time.perf_counter();rgb=rgb_gpu.numpy();gray=gray_gpu.numpy() if n!='CenterRGB' else None;diff=sd_gpu.numpy() if n!='CenterRGB' else None;copy_ms+=(time.perf_counter()-b)*1000
                b=time.perf_counter();cells=np.array([rgb[y-2:y+3,x-2:x+3].mean() for x,y in centres[n]]);bits=cells>(cells[0]+cells[1])/2
                ids[n]=int(np.dot(bits[4:20],2**np.arange(15,-1,-1)))
                if ids[n]!=i or bits[20]!=bool(bits[4:20].sum()%2):raise RuntimeError(f'pixel mismatch {i}: {ids}')
                id_ms+=(time.perf_counter()-b)*1000
                b=time.perf_counter();payloads[n]=rgb.tobytes()
                if n!='CenterRGB':payloads[n+'_gray']=gray.tobytes();payloads[n+'_sd']=diff.tobytes()
                pack_ms+=(time.perf_counter()-b)*1000
                if i in [0,100]:
                    np.savez(a.output/f'probe_{n}_{i}.npz',hdr=buffers[n].numpy(),rgb=rgb,gray=gray if gray is not None else np.zeros(0))
        if a.stage in ['sd','save'] and not a.gpu_process:
            for n,raw in arrays.items():
                b=time.perf_counter();img=raw.astype(np.float32);cast_ms+=(time.perf_counter()-b)*1000
                b=time.perf_counter()
                cells=np.array([img[y-2:y+3,x-2:x+3,:3].mean() for x,y in centres[n]])
                bits=cells>(cells[0]+cells[1])/2
                ids[n]=int(np.dot(bits[4:20],2**np.arange(15,-1,-1)))
                if ids[n]!=i:raise RuntimeError(f'fixed 3 ticks stale frame {i}: {ids}')
                id_ms+=(time.perf_counter()-b)*1000;b=time.perf_counter()
                v=np.clip(img[...,:3],0,1);rgb=np.rint(np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)*255).astype(np.uint8)
                srgb_ms+=(time.perf_counter()-b)*1000;b=time.perf_counter();payloads[n]=rgb.tobytes();pack_ms+=(time.perf_counter()-b)*1000
                if n!='CenterRGB':
                    b=time.perf_counter();gray=np.rint(np.clip(img[...,:3] @ np.array([.2126,.7152,.0722],np.float32),0,1)*255).astype(np.uint8);gray_ms+=(time.perf_counter()-b)*1000
                    b=time.perf_counter()
                    up=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,-1,0],[0,1,0],[0,0,0]],np.float32));up[0]=0
                    right=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,0,0],[0,1,-1],[0,0,0]],np.float32));right[:,-1]=0
                    sd_ms+=(time.perf_counter()-b)*1000
                    payloads[n+'_gray']=gray.tobytes();payloads[n+'_sd']=np.stack([up,right]).tobytes()
        b=time.perf_counter();hashes={}
        if a.stage=='save':
            if a.async_save:
                if len(pending)>=8:pending.popleft().result()
                pending.append(writer.submit(save,payloads,hashes))
            else:save(payloads,hashes)
        save_ms=(time.perf_counter()-b)*1000
        simulation_after_s = float(world.current_time) - base_simulation_s
        timeline_after_s = float(timeline.get_current_time()) - base_timeline_s
        timing_log.write(json.dumps({'index':i,'source_time_s':source_time_s,'simulation_before_s':simulation_before_s,'simulation_after_s':simulation_after_s,'timeline_before_s':timeline_before_s,'timeline_after_s':timeline_after_s,'pixel_ids':ids})+'\n'); timing_log.flush()
        samples.append({'cast_ms':cast_ms,'id_ms':id_ms,'srgb_ms':srgb_ms,'pack_ms':pack_ms,'gpu_process_ms':gpu_process_ms,'index':i,'simulation_ns':i*10000000,'source_time_s':source_time_s,'simulation_before_s':simulation_before_s,'simulation_after_s':simulation_after_s,'timeline_before_s':timeline_before_s,'timeline_after_s':timeline_after_s,'refresh_count':refresh_count,'refresh_sim_deltas':refresh_sim_deltas,'refresh_timeline_deltas':refresh_timeline_deltas,'render_ms':sum(tick_times),'ticks_ms':tick_times,
            'get_ms':(after_get-after_render)*1000,'sync_ms':sync_ms,'copy_ms':copy_ms,'color_ms':color_ms,
            'gray_ms':gray_ms,'sd_ms':sd_ms,'save_ms':save_ms,'total_ms':(time.perf_counter()-begin)*1000,
            'pixel_ids':ids,'hashes':hashes})
        if i%100==0:print(a.eyes,a.stage,i,flush=True)
    loop_end=time.time();b=time.perf_counter()
    for future in pending:future.result()
    writer.shutdown()
    for stream in streams.values():stream.flush();os.fsync(stream.fileno());stream.close()
    timing_log.close()
    fsync_ms=(time.perf_counter()-b)*1000
    meta.update(stage=a.stage,eyes=a.eyes,start_epoch=start,end_epoch=loop_end,fsync_ms=fsync_ms,
        elapsed_s=time.perf_counter()-begin_all,rows=samples,gpu_frametime_samples=gpu_samples,gpu_profiler_example=example,
        viewports=viewport_inventory,render_products=[str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName()=='RenderProduct'],
        options=vars(a)|{"output":str(a.output)},fixed_render_ticks_per_sample=a.ticks,buffer_devices={n:str(b.device) for n,b in buffers.items()})
    (a.output/'profile.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
