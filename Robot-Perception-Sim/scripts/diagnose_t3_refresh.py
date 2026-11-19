"""Minimal A/B diagnostic for non-stepping Render Product refreshes."""
import argparse, hashlib, json, sys, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True); p.add_argument('--config', required=True)
    p.add_argument('--center-config', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=['A', 'B'], required=True); p.add_argument('--frames', type=int, default=3)
    a, _ = p.parse_known_args(); a.output.mkdir(parents=True, exist_ok=False)
    import ctypes
    cuda=ctypes.CDLL('libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
    if cuda.cuInit(0): raise RuntimeError('CUDA initialization failed')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'active_gpu':1,'physics_gpu':0,'multi_gpu':False,
        'limit_cpu_threads':8,'extra_args':['--/rtx/hydra/supportMultiTickRate=false','--/app/audio/enabled=false']})
    try: run(a)
    except BaseException:
        (a.output/'failure.txt').write_text(__import__('traceback').format_exc()); raise
    finally: app.close()


def run(a):
    import numpy as np
    import omni.replicator.core as rep
    from omni.replicator.core.scripts.utils import viewport_manager
    from isaacsim.core.api import World
    from isaacsim.core.rendering_manager import RenderingManager
    import omni.timeline
    from omni.hydra.engine.stats import HydraEngineStats
    from pxr import Gf, Sdf, UsdGeom
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'acquisition'))
    from t3_park import load_scene, FrameMarkers
    c=json.loads(Path(a.config).read_text())['stereo']; center=json.loads(Path(a.center_config).read_text())
    stage,_,_,_=load_scene(a.scene,100)
    world=World(physics_dt=.01, rendering_dt=.01); world.reset(); world.play()
    products={}; anns={}; markers={}; centres={}; cameras={}
    for name,offset in [('CenterRGB',0),('Left',.035),('Right',-.035)]:
        w,h=(center['width'],center['height']) if name=='CenterRGB' else (c['width'],c['height'])
        k=[[center['fx'],0,center['cx']],[0,center['fy'],center['cy']],[0,0,1]] if name=='CenterRGB' else c['K']
        d=center['D'] if name=='CenterRGB' else c['D']; path='/World/Cleaner/T3/'+name
        cam=UsdGeom.Camera.Define(stage,path); xf=UsdGeom.Xformable(cam);xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.6,offset,.78),Gf.Vec3d(1.6,offset,.78),Gf.Vec3d(0,0,1)).GetInverse())
        cam.CreateFocalLengthAttr(1.79);cam.CreateHorizontalApertureAttr(1.79*w/k[0][0]);cam.CreateVerticalApertureAttr(1.79*h/k[1][1]);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
        prim=cam.GetPrim();prim.ApplyAPI('OmniLensDistortionOpenCvPinholeAPI');prim.GetAttribute('omni:lensdistortion:model').Set('opencvPinhole')
        vals={'fx':k[0][0],'fy':k[1][1],'cx':float(k[0][2]),'cy':float(k[1][2]),'imageSize':Gf.Vec2i(w,h)}
        vals.update(zip(('k1','k2','p1','p2','k3','k4','k5','k6','s1','s2','s3','s4'),d+[0.]*7))
        for key,val in vals.items():prim.GetAttribute('omni:lensdistortion:opencvPinhole:'+key).Set(val)
        product=viewport_manager.get_render_product(path,(w,h),False,'Diag'+name);products[name]=product
        rp=stage.GetPrimAtPath(product.hydra_texture.get_render_product_path());rp.CreateAttribute('omni:rtx:post:aa:op',Sdf.ValueTypeNames.Token).Set('dlss')
        anns[name]=rep.AnnotatorRegistry.get_annotator('HdrColor',device='cuda');anns[name].attach(product.hydra_texture.get_render_product_path())
        markers[name]=FrameMarkers(stage,[path],(w,h));xy=(np.array(markers[name].centres)-[k[0][2],k[1][2]])/[k[0][0],k[1][1]];rr=(xy*xy).sum(1);rad=1+d[0]*rr+d[1]*rr**2+d[4]*rr**3
        centres[name]=np.rint(xy*rad[:,None]*[k[0][0],k[1][1]]+[k[0][2],k[1][2]]).astype(int).tolist();cameras[name]=path
    for _ in range(100):world.step(render=True)
    base_sim=float(world.current_time);base_timeline=float(omni.timeline.get_timeline_interface().get_current_time());hydra=HydraEngineStats();timeline=omni.timeline.get_timeline_interface()
    rows=[]
    with (a.output/'timing.jsonl').open('w') as log:
        for i in range(a.frames):
            for m in markers.values():m.set(i)
            visibility={name:str(markers[name].tiles[0][4].Get()) for name in markers}
            for attempt in range(8):
                before_sim=float(world.current_time)-base_sim;before_t=float(timeline.get_current_time())-base_timeline;t0=time.perf_counter()
                if a.method=='A':world.step(render=True,step_sim=False)
                else:RenderingManager.render()
                render_ms=(time.perf_counter()-t0)*1000;gpu=hydra.get_gpu_profiler_result()
                decoded={};checksums={}
                for name,ann in anns.items():
                    img=ann.get_data().numpy().astype(np.float32);cells=np.array([img[y-2:y+3,x-2:x+3,:3].mean() for x,y in centres[name]]);bits=cells>(cells[0]+cells[1])/2
                    decoded[name]=int(np.dot(bits[4:20],2**np.arange(15, -1,-1))) if bits.size else -2;checksums[name]=hashlib.sha256(img.tobytes()).hexdigest()
                after_sim=float(world.current_time)-base_sim;after_t=float(timeline.get_current_time())-base_timeline
                row={'requested_id':i,'attempt':attempt,'method':a.method,'visibility':visibility,'decoded_ids':decoded,'checksums':checksums,'simulation_before_s':before_sim,'simulation_after_s':after_sim,'timeline_before_s':before_t,'timeline_after_s':after_t,'render_ms':render_ms,'gpu_profiler':gpu}
                log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
                if all(v==i for v in decoded.values()):break
    (a.output/'summary.json').write_text(json.dumps({'method':a.method,'frames':a.frames,'camera_paths':cameras,'rows':rows},indent=2)+'\n')

if __name__=='__main__':main()
