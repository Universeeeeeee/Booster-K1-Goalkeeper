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
    p.add_argument('--clock-mode', choices=['legacy','C1'], default='C1')
    p.add_argument('--physics-hz', type=float, default=100.0)
    p.add_argument('--physical-steps-per-sample', type=int, default=1)
    p.add_argument('--substep-render', choices=['every', 'last'])
    a, _ = p.parse_known_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import ctypes
    cuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    if cuda.cuInit(0): raise RuntimeError("CUDA initialization failed")
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'active_gpu': int(os.environ.get('URBAN_GPU', '1')), 'physics_gpu': 0,
        'multi_gpu': False, 'limit_cpu_threads': 8,
        'extra_args': ['--/rtx/hydra/supportMultiTickRate=false', '--/app/audio/enabled=false']})
    code = 0
    try:
        capture(app, a)
    except BaseException:
        code = 1
        import traceback
        (a.output/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        app.app.post_quit(code)
        app.close(exit_code=code)


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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'acquisition'))
    from robot_perception_sim.t3_sdtd import TD_ENCODING, td
    from t3_park import load_scene, FrameMarkers
    from verify_t3_abc import decode_frame_id
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
    marker_strips = {}
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
        # Keep diagnostic tiles inside the central projection for the strong barrel preset.
        strong_barrel = d[0] <= -0.3 and d[1] == 0
        rows = (128, 192) if strong_barrel else (32, 96)
        start_x = w // 2 - 276 if strong_barrel else 20
        markers[name] = [FrameMarkers(stage, [path], (w, h), row=row, start_x=start_x) for row in rows]
        marker_strips[name] = []
        for marker in markers[name]:
            xy = (np.array(marker.centres)-[k[0][2],k[1][2]])/[k[0][0],k[1][1]]
            x, y = xy.T
            rr = x*x + y*y
            radial = 1+d[0]*rr+d[1]*rr**2+d[4]*rr**3
            distorted = xy * radial[:, None]
            distorted[:, 0] += 2*d[2]*x*y + d[3]*(rr + 2*x*x)
            distorted[:, 1] += d[2]*(rr + 2*y*y) + 2*d[3]*x*y
            marker_strips[name].append(np.rint(distorted * [k[0][0],k[1][1]] + [k[0][2],k[1][2]]).astype(int).tolist())
        centres[name] = marker_strips[name][0]
        optics[name] = {'resolution':[w,h], 'K':k, 'D':d, 'focal_attribute':cam.GetFocalLengthAttr().Get(),
            'local_matrix':np.array(xf.GetLocalTransformation()).T.tolist()}
    physics_dt = 1.0 / a.physics_hz
    if a.physical_steps_per_sample < 1 or abs(a.physics_hz * physics_dt - 1.0) > 1e-12:
        raise ValueError('invalid physical timing configuration')
    if abs(a.physics_hz / a.physical_steps_per_sample - 100.0) > 1e-9:
        raise ValueError('physics_hz / physical_steps_per_sample must equal 100 Hz sample rate')
    grouped = a.substep_render == 'last'
    render_dt = physics_dt * a.physical_steps_per_sample if grouped else physics_dt
    world = World(physics_dt=physics_dt, rendering_dt=render_dt); world.reset(); world.play()
    # Keep the timeline horizon beyond the requested sensor capture.  At 800 Hz
    # the timeline advances on every physical substep, so the old scene end time
    # can be reached before a 10 s, 100 Hz sensor run completes.
    stage.SetStartTimeCode(0)
    warmup_updates = (100 + a.physical_steps_per_sample - 1) // a.physical_steps_per_sample * a.physical_steps_per_sample
    if grouped:
        warmup_updates //= a.physical_steps_per_sample
    horizon_s = a.frames / 100.0 + warmup_updates * render_dt + 1.0
    stage.SetEndTimeCode(horizon_s * stage.GetTimeCodesPerSecond())
    timeline.set_end_time(horizon_s)
    timeline.set_looping(False)
    for _ in range(warmup_updates): world.step(render=True)
    physics_steps = []
    if a.substep_render:
        world.add_physics_callback('t3_substep_audit', physics_steps.append)
    base_simulation_s = float(world.current_time)
    base_timeline_s = float(timeline.get_current_time())
    stage.GetRootLayer().Export(str(a.output/'scene.usda'))
    meta = {'mode':a.mode, 'frames':a.frames, 'optics':optics, 'centres':centres, 'provenance':provenance,
        'sample_period_ns':10000000, 'gray':'linear HDR luminance quantized as round(clip(Y,0,1)*255); fixed gain=1',
        'sd':'int16 up/right, values 1..255, neutral 128',
        'sd_encoding':'half_trunc_plus_128', 'marker_strips':marker_strips,
        'timing':'explicit sensor sample clock; renderer ticks include same-state flushing, not sensor samples',
        'render_settings':{k:settings.get(k) for k in ['/rtx/rendermode','/rtx/post/aa/op','/rtx/dlss/frameGeneration']}}
    save_td = a.stage == 'save' and a.mode != 'A'
    if save_td:
        meta['td_encoding'] = TD_ENCODING
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
                if save_td:
                    streams[n + '_td'] = (a.output / (n + '.td')).open('wb')
    if a.no_viewport:
        for v in windows.get_viewport_window_instances():v.viewport_api.updates_enabled=False
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    writer=ThreadPoolExecutor(max_workers=1);pending=deque()
    def save(payloads,hashes):
        for n,payload in payloads.items():streams[n].write(payload);hashes[n]=hashlib.sha256(payload).hexdigest()
    from t3_gpu_sd import images as gpu_images, differences, TemporalDifference
    allocated={}
    gpu_td = {}
    previous_gray = {}
    if a.gpu_process:
        for n,ann in annotations.items():
            buf=ann.get_data();h,w=buf.shape[:2]
            allocated[n]=(wp.empty((h,w,3),dtype=wp.uint8,device=buf.device),wp.empty((h,w),dtype=wp.uint8,device=buf.device),wp.empty((2,h,w),dtype=wp.int16,device=buf.device))
            if save_td and n != 'CenterRGB':
                gpu_td[n] = TemporalDifference(allocated[n][1])
            wp.launch(gpu_images,(h,w),inputs=[buf,*allocated[n][:2]],device=buf.device)
            wp.launch(differences,(h,w),inputs=[allocated[n][1],allocated[n][2]],device=buf.device)
        wp.synchronize()
    def refresh_c1(expected_id):
        refresh_count=0; refresh_sim_deltas=[]; refresh_timeline_deltas=[]; refresh_times=[]
        attempts = []
        timeline.pause()
        try:
            for _ in range(16):
                before_refresh_sim=float(world.current_time)-base_simulation_s
                before_refresh_t=float(timeline.get_current_time())-base_timeline_s
                b=time.perf_counter(); RenderingManager.render(); refresh_times.append((time.perf_counter()-b)*1000)
                refresh_count += 1
                refresh_sim_deltas.append(float(world.current_time)-base_simulation_s-before_refresh_sim)
                refresh_timeline_deltas.append(float(timeline.get_current_time())-base_timeline_s-before_refresh_t)
                probe={n:ann.get_data() for n,ann in annotations.items()}; probe_ids={}
                for n,buf in probe.items():
                    if a.gpu_process:
                        rgb_gpu=allocated[n][0]; wp.launch(gpu_images,(buf.shape[0],buf.shape[1]),inputs=[buf,rgb_gpu,allocated[n][1]],device=buf.device); wp.synchronize_device(buf.device); img=rgb_gpu.numpy()
                    else: img=buf.numpy().astype(np.float32)
                    probe_ids[n] = decode_frame_id(img, marker_strips[n])
                attempts.append(probe_ids)
                if all(v == expected_id for v in probe_ids.values()): break
            else:
                from PIL import Image
                for name, buffer in probe.items():
                    if a.gpu_process:
                        image = allocated[name][0].numpy()
                    else:
                        image = np.rint(np.clip(buffer.numpy()[..., :3], 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(image).save(a.output / f'frame_mismatch_{name}.png')
                raise RuntimeError(f'C1 camera frame mismatch for requested ID {expected_id}: {attempts}')
        finally:
            timeline.play()
        return refresh_count, refresh_sim_deltas, refresh_timeline_deltas, refresh_times
    samples=[]; gpu_samples=[]; example=None
    start=time.time(); begin_all=time.perf_counter()
    for i in range(a.frames):
        begin=time.perf_counter()
        source_time_s = i * 0.01
        body_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(-48 + 1.5 * source_time_s, -8.375, 0)))
        simulation_before_s = float(world.current_time) - base_simulation_s
        timeline_before_s = float(timeline.get_current_time()) - base_timeline_s
        for camera_markers in markers.values():
            for marker in camera_markers:
                marker.set(i)
        tick_times=[]
        physics_steps.clear()
        for substep in range(1 if grouped else a.physical_steps_per_sample):
            if grouped:
                substep = a.physical_steps_per_sample - 1
            sub_time = source_time_s + substep * physics_dt
            body_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(-48 + 1.5 * sub_time, -8.375, 0)))
            # Advance every physical substep through the Isaac render loop so
            # Timeline advances by the full sample interval; only the final
            # substep's published buffer is consumed below.
            b=time.perf_counter();world.step(render=True);tick_times.append((time.perf_counter()-b)*1000)
        # The final physical substep produced the sensor sample; C1 refreshes below
        # do not advance either clock.
        nodes=hydra.get_gpu_profiler_result()
        gpu_samples.append([sum(n.get('duration',0) for n in dev if n.get('indent',0)==0) for dev in nodes])
        if nodes and example is None:example=nodes
        refresh_count=0; refresh_sim_deltas=[]; refresh_timeline_deltas=[]
        if a.clock_mode == 'C1' and a.stage != 'render':
            refresh_count, refresh_sim_deltas, refresh_timeline_deltas, refresh_times = refresh_c1(i)
            tick_times.extend(refresh_times)
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
            if i == 0 and a.clock_mode != 'C1':
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
                if n in gpu_td:
                    gpu_td[n].sample(gray_gpu, i)
            wp.synchronize_device(next(iter(buffers.values())).device)
            gpu_process_ms=(time.perf_counter()-b)*1000
            for n,(rgb_gpu,gray_gpu,sd_gpu) in allocated.items():
                b=time.perf_counter();rgb=rgb_gpu.numpy();gray=gray_gpu.numpy() if n!='CenterRGB' else None;diff=sd_gpu.numpy() if n!='CenterRGB' else None;copy_ms+=(time.perf_counter()-b)*1000
                b=time.perf_counter()
                ids[n] = decode_frame_id(rgb, marker_strips[n])
                if ids[n]!=i:raise RuntimeError(f'pixel mismatch {i}: {ids}')
                id_ms+=(time.perf_counter()-b)*1000
                b=time.perf_counter();payloads[n]=rgb.tobytes()
                if n!='CenterRGB':payloads[n+'_gray']=gray.tobytes();payloads[n+'_sd']=diff.tobytes()
                if n in gpu_td:
                    payloads[n + '_td'] = gpu_td[n].output.numpy().tobytes()
                pack_ms+=(time.perf_counter()-b)*1000
                if i in [0,100]:
                    np.savez(a.output/f'probe_{n}_{i}.npz',hdr=buffers[n].numpy(),rgb=rgb,gray=gray if gray is not None else np.zeros(0))
        if a.stage in ['sd','save'] and not a.gpu_process:
            for n,raw in arrays.items():
                b=time.perf_counter();img=raw.astype(np.float32);cast_ms+=(time.perf_counter()-b)*1000
                b=time.perf_counter()
                ids[n] = decode_frame_id(img, marker_strips[n])
                if ids[n]!=i:raise RuntimeError(f'fixed 3 ticks stale frame {i}: {ids}')
                id_ms+=(time.perf_counter()-b)*1000;b=time.perf_counter()
                v=np.clip(img[...,:3],0,1);rgb=np.rint(np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055)*255).astype(np.uint8)
                srgb_ms+=(time.perf_counter()-b)*1000;b=time.perf_counter();payloads[n]=rgb.tobytes();pack_ms+=(time.perf_counter()-b)*1000
                if n!='CenterRGB':
                    b=time.perf_counter();gray=np.rint(np.clip(img[...,:3] @ np.array([.2126,.7152,.0722],np.float32),0,1)*255).astype(np.uint8);gray_ms+=(time.perf_counter()-b)*1000
                    b=time.perf_counter()
                    up=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,-1,0],[0,1,0],[0,0,0]],np.float32));up[0]=0
                    right=cv2.filter2D(gray,cv2.CV_16S,np.array([[0,0,0],[0,1,-1],[0,0,0]],np.float32));right[:,-1]=0
                    diff=(np.trunc(np.stack([up, right]) * np.float32(0.5)) + 128).astype(np.int16)
                    sd_ms+=(time.perf_counter()-b)*1000
                    payloads[n+'_gray']=gray.tobytes()
                    payloads[n+'_sd']=diff.tobytes()
                    if save_td:
                        payloads[n + '_td'] = td(gray, previous_gray.get(n)).tobytes()
                        previous_gray[n] = gray
        b=time.perf_counter();hashes={}
        if a.stage=='save':
            if a.async_save:
                if len(pending)>=8:pending.popleft().result()
                pending.append(writer.submit(save,payloads,hashes))
            else:save(payloads,hashes)
        save_ms=(time.perf_counter()-b)*1000
        simulation_after_s = float(world.current_time) - base_simulation_s
        timeline_after_s = float(timeline.get_current_time()) - base_timeline_s
        if a.substep_render and (len(physics_steps) != a.physical_steps_per_sample
                                 or any(abs(dt - physics_dt) > 1e-9 for dt in physics_steps)):
            raise RuntimeError(f'Unexpected PhysX substeps at {i}: {physics_steps}')
        timing_log.write(json.dumps({'index':i,'source_time_s':source_time_s,'simulation_before_s':simulation_before_s,'simulation_after_s':simulation_after_s,'timeline_before_s':timeline_before_s,'timeline_after_s':timeline_after_s,'pixel_ids':ids})+'\n'); timing_log.flush()
        samples.append({'cast_ms':cast_ms,'id_ms':id_ms,'srgb_ms':srgb_ms,'pack_ms':pack_ms,'gpu_process_ms':gpu_process_ms,'index':i,'simulation_ns':i*10000000,'source_time_s':source_time_s,'simulation_before_s':simulation_before_s,'simulation_after_s':simulation_after_s,'timeline_before_s':timeline_before_s,'timeline_after_s':timeline_after_s,'refresh_count':refresh_count,'refresh_sim_deltas':refresh_sim_deltas,'refresh_timeline_deltas':refresh_timeline_deltas,'render_ms':sum(tick_times),'ticks_ms':tick_times,
            'get_ms':(after_get-after_render)*1000,'sync_ms':sync_ms,'copy_ms':copy_ms,'color_ms':color_ms,
            'gray_ms':gray_ms,'sd_ms':sd_ms,'save_ms':save_ms,'total_ms':(time.perf_counter()-begin)*1000,
            'pixel_ids':ids,'hashes':hashes})
        if a.substep_render:
            samples[-1]['physics_steps_s'] = physics_steps.copy()
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
