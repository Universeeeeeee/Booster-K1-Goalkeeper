"""Stream an official virtual ZED X in an isolated Isaac Sim 4.5 scene.

Run with Isaac's python.sh. This sends RGB through the official extension;
renderer depth is saved separately for offline verification, never ingested.
"""

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--camera-usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--fps", type=int, choices=(15, 30, 60), default=30)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--duration", type=float, default=180)
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--right-shift-m", type=float, default=0.0,
                        help="Intentional right-eye extrinsic perturbation for depth-source checks")
    args, _ = parser.parse_known_args()
    if not args.extension_root.is_dir() or not args.camera_usd.is_file():
        parser.error("Official extension directory and camera USD must exist")
    if args.duration <= 0 or not 1024 <= args.port <= 65535:
        parser.error("Positive duration and unprivileged port required")
    args.output.mkdir(parents=True, exist_ok=False)

    driver = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    status = driver.cuInit(0)
    if status:
        raise RuntimeError(f"CUDA initialization failed: {status}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "active_gpu": args.gpu,
                         "physics_gpu": args.gpu, "multi_gpu": False,
                         "fast_shutdown": True,
                         "width": 960, "height": 600, "anti_aliasing": 0})
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    return_code = 0
    try:
        run_scene(app, args, lambda: stop)
    except Exception:
        return_code = 1
        raise
    finally:
        # Scene/OG wrappers must die while their native plugins are still loaded.
        gc.collect()
        print("ZED_POC_PRE_CLOSE_GC_COMPLETE", flush=True)
        app.app.post_quit(return_code)
        app.close()
        print("ZED_POC_APP_CLOSED", flush=True)


def run_scene(app, args, should_stop):
    annotator = world = rgb_left = rgb_right = gt_depth = observation_writer = None
    try:
        import carb.settings
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        import numpy as np
        from PIL import Image
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt
        from isaacsim.core.api import World
        import verify_zed_sdk_stream as frame_codec

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.add_path(str(args.extension_root.resolve()))
        manager.set_extension_enabled_immediate("sl.sensor.camera", True)
        if not manager.is_extension_enabled("sl.sensor.camera"):
            raise RuntimeError("Official ZED camera extension did not enable")
        from sl.sensor.camera.annotators import ZEDAnnotator

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        world = World(physics_dt=1 / 120, rendering_dt=1 / args.fps,
                      stage_units_in_meters=1)
        camera_root = stage.DefinePrim("/World/ZED", "Xform")
        camera_root.GetReferences().AddReference(str(args.camera_usd.resolve()))
        for _ in range(10):
            app.update()
        for prim in Usd.PrimRange(camera_root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                # Keep the rigid body and IMU active, with no free-fall.
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)

        left_path = "/World/ZED/base_link/ZED_X/CameraLeft"
        right_path = "/World/ZED/base_link/ZED_X/CameraRight"
        for path in (left_path, right_path):
            if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
                raise RuntimeError(f"Official ZED X camera prim missing: {path}")
        camera_pose = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(left_path))

        # Unique face colours provide stereo texture, with a known 2 m optical depth.
        plane = UsdGeom.Mesh.Define(stage, "/World/TexturePlane")
        points, colours, indices = [], [], []
        rng = np.random.default_rng(20261024)
        for row in range(40):
            for col in range(64):
                x, y, size = (col - 32) * 0.065, (row - 20) * 0.065, 0.065
                offset = len(points)
                points.extend(Gf.Vec3f(camera_pose.Transform(Gf.Vec3d(a, b, -2.0)))
                              for a, b in ((x, y), (x + size, y),
                                           (x + size, y + size), (x, y + size)))
                indices.extend(range(offset, offset + 4))
                colour = rng.uniform(0.08, 0.9, size=3)
                colours.extend([Gf.Vec3f(*colour)] * 4)
        plane.CreatePointsAttr(Vt.Vec3fArray(points))
        plane.CreateFaceVertexCountsAttr([4] * (40 * 64))
        plane.CreateFaceVertexIndicesAttr(indices)
        plane.CreateDoubleSidedAttr(True)
        plane.CreateSubdivisionSchemeAttr("none")
        plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.faceVarying).Set(colours)
        # Diagnostic source identity: four sentinel bits, uint16 MSB-first,
        # even parity, then two trailing sentinels, visible to both eyes.
        marker = UsdGeom.Mesh.Define(stage, "/World/FrameId")
        marker_points = []
        for cell in range(23):
            x, y, w, h = -0.8 + cell * 0.07, 0.9, 0.07, 0.12
            marker_points.extend(Gf.Vec3f(camera_pose.Transform(Gf.Vec3d(a, b, -1.99)))
                                 for a, b in ((x, y), (x + w, y),
                                              (x + w, y + h), (x, y + h)))
        marker.CreatePointsAttr(Vt.Vec3fArray(marker_points))
        marker.CreateFaceVertexCountsAttr([4] * 23)
        marker.CreateFaceVertexIndicesAttr(list(range(92)))
        marker.CreateDoubleSidedAttr(True)
        marker.CreateSubdivisionSchemeAttr("none")
        marker_colours = marker.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
        marker_colours.Set([Gf.Vec3f(0.01)] * 23)
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(1000)

        if args.right_shift_m:
            right = UsdGeom.Xformable(stage.GetPrimAtPath(right_path))
            right.AddTranslateOp(opSuffix="source_check").Set(Gf.Vec3d(args.right_shift_m, 0, 0))

        world.reset()
        annotator = ZEDAnnotator([Sdf.Path("/World/ZED")], "ZED_X", args.port,
                                 "SVGA", args.fps, not args.network)
        # Stamp the paired render buffers with their own synchronization time,
        # rather than a physics-clock read that can advance before GPU delivery.
        timestamp_node = annotator.graph.create_node(
            annotator._graph_path + "/zed_render_reference_seconds", "omni.graph.nodes.Divide", True)
        annotator.nodes.append(timestamp_node)
        annotator.sync_node.get_attribute("outputs:rationalTimeNumerator").connect(
            timestamp_node.get_attribute("inputs:a"), True)
        annotator.sync_node.get_attribute("outputs:rationalTimeDenominator").connect(
            timestamp_node.get_attribute("inputs:b"), True)
        annotator.sim_time.get_attribute("outputs:simulationTime").disconnect(
            annotator.zed_.get_attribute("inputs:simulationTime"), True)
        timestamp_node.get_attribute("outputs:result").connect(
            annotator.zed_.get_attribute("inputs:simulationTime"), True)
        # v3.2 calls init_camera(left) twice. Explicitly initialize both eyes to
        # the same official SVGA model, without editing the vendor extension.
        for path in (left_path, right_path):
            if not annotator.init_camera(path, [960, 600]):
                raise RuntimeError(f"Could not initialize camera: {path}")
        rgb_left = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_right = rep.AnnotatorRegistry.get_annotator("rgb")
        gt_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb_left.attach(annotator.left_rp)
        rgb_right.attach(annotator.right_rp)
        gt_depth.attach(annotator.left_rp)
        settings = carb.settings.get_settings()
        print("ZED_POC_AA_AFTER_SCENE_INIT", settings.get("/rtx/post/aa/op"), flush=True)
        # Stage/render-product initialization can restore the renderer default
        # (DLSS=3), so apply the instantaneous baseline after those operations.
        # Match Replicator's public set_render_rtx_realtime("Off") prerequisite.
        settings.set("/rtx-transient/post/aa/limitedOps", False)
        settings.set("/rtx-defaults/post/aa/op", 0)
        settings.set("/rtx/post/aa/op", 0)
        world.play()
        timeline = omni.timeline.get_timeline_interface()
        for _ in range(60):
            world.step(render=True)

        calibration = {}
        for side, path in (("left", left_path), ("right", right_path)):
            cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
            focal = cam.GetFocalLengthAttr().Get()
            aperture = cam.GetHorizontalApertureAttr().Get()
            matrix = UsdGeom.XformCache().GetLocalToWorldTransform(cam.GetPrim())
            calibration[side] = {"focal_length_usd": focal, "horizontal_aperture_usd": aperture,
                                 "fx_px": 960 * focal / aperture,
                                 "world_from_usd_camera_row_matrix": np.array(matrix).tolist()}
        metadata = {"source": "official_stereolabs_v3.2.0_rgb_stream", "camera_model": "ZED_X",
                    "sender_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "camera_usd_sha256": hashlib.sha256(args.camera_usd.read_bytes()).hexdigest(),
                    "transport": "network" if args.network else "ipc", "port": args.port,
                    "gpu_requested": args.gpu, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                    "pid": os.getpid(), "resolution": [960, 600], "configured_fps": args.fps,
                    "physics_dt_s": 1 / 120, "plane_optical_z_m": 2.0,
                    "right_shift_m": args.right_shift_m, "cameras": calibration,
                    "sdk_depth_ingested": False, "renderer_depth_source": "renderer_gt_direct",
                    "exposure_model": "instantaneous_render_baseline", "sim2real": False}
        metadata["render_anti_aliasing_op"] = settings.get("/rtx/post/aa/op")
        metadata["render_aa_limited_ops"] = settings.get("/rtx-transient/post/aa/limitedOps")
        metadata["sdk_timestamp_source"] = "paired_render_sync_rational_seconds"
        metadata["sdk_timestamp_clock_domain"] = "SyntheticData NewFrame rational reference; distinct from World and timeline clocks"
        metadata["frame_id_decoder_sha256"] = hashlib.sha256(
            Path(frame_codec.__file__).read_bytes()).hexdigest()
        if metadata["render_anti_aliasing_op"] != 0:
            raise RuntimeError("Renderer replaced the required no-temporal-AA baseline")
        metadata["frame_id_marker"] = {
            "encoding": "1010 + uint16 MSB-first + even parity + 01",
            "cells": 23, "optical_depth_m": 1.99, "pixel_centres": {}}
        for side, path in (("left", left_path), ("right", right_path)):
            inverse = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).GetInverse()
            centres = []
            for cell in range(23):
                point = camera_pose.Transform(Gf.Vec3d(-0.8 + (cell + 0.5) * 0.07, 0.96, -1.99))
                local = inverse.Transform(point)
                centres.append([480 + calibration[side]["fx_px"] * local[0] / -local[2],
                                300 - calibration[side]["fx_px"] * local[1] / -local[2]])
            metadata["frame_id_marker"]["pixel_centres"][side] = centres
        metadata["source_observation_method"] = "replicator_writer_same_packet_reference_and_rgb"
        metadata["render_observations_file"] = "render_observations.jsonl"

        class FrameObservationWriter(rep.Writer):
            data_structure = "renderProduct"
            version = "1.0.0"

            def __init__(self):
                self.annotators = [rep.AnnotatorRegistry.get_annotator("rgb")]
                self.references = {}
                self.failure = None

            def write_metadata(self):
                # The PoC writes its complete configuration beside this log.
                self._is_metadata_written = True

            def write(self, data):
                ref = [int(value) for value in data["reference_time"]]
                observation = {"observation_index": self.num_written,
                               "reference_time": ref, "host_monotonic_ns": time.monotonic_ns(),
                               "observation_valid": False, "reference_conflict": False}
                try:
                    products = {product["camera"]: product for product in data["renderProducts"].values()}
                    for side, path in (("left", left_path), ("right", right_path)):
                        observation[f"observed_{side}_visual_id"] = frame_codec.decode_frame_id(
                            np.asarray(products[path]["rgb"]["data"])[..., :3],
                            metadata["frame_id_marker"]["pixel_centres"][side])
                    left_id = observation["observed_left_visual_id"]
                    right_id = observation["observed_right_visual_id"]
                    observation["observation_valid"] = left_id == right_id
                    key = tuple(ref)
                    if observation["observation_valid"]:
                        if key in self.references and self.references[key] != left_id:
                            observation["reference_conflict"] = True
                            observation["observation_valid"] = False
                            self.failure = "One render packet reference identifies conflicting pixel IDs"
                        self.references[key] = left_id
                except ValueError as error:
                    observation["observation_error"] = str(error)
                except Exception as error:
                    observation["observation_error"] = str(error)
                    self.failure = str(error)
                with (args.output / "render_observations.jsonl").open("a") as observations:
                    observations.write(json.dumps(observation) + "\n")

        observation_writer = FrameObservationWriter()
        observation_writer.attach([annotator.left_rp, annotator.right_rp])
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        stage.GetRootLayer().Export(str(args.output / "scene.usda"))
        (args.output / "ready").write_text(str(os.getpid()))
        print("ZED_POC_READY", json.dumps(metadata), flush=True)
        start = time.monotonic()
        index = 0
        with (args.output / "source_frames.jsonl").open("w") as records:
            while app.is_running() and not should_stop() and time.monotonic() - start < args.duration:
                if (args.output / "stop").exists():
                    break
                before = time.monotonic()
                host_before_step_ns = time.monotonic_ns()
                marker_authored_sim_time_s = world.current_time
                marker_authored_timeline_time_s = timeline.get_current_time()
                value = index % 65536
                payload = [(value >> bit) & 1 for bit in range(15, -1, -1)]
                bits = [1, 0, 1, 0] + payload + [sum(payload) % 2] + [0, 1]
                marker_colours.Set([Gf.Vec3f(1.0 if bit else 0.01) for bit in bits])
                world.step(render=True)
                if observation_writer.failure:
                    raise RuntimeError(observation_writer.failure)
                records.write(json.dumps({"source_frame": index,
                                          "sim_time_s": world.current_time,
                                          "zed_input_sim_time_s": annotator.zed_.get_attribute("inputs:simulationTime").get(),
                                          "og_sim_time_s": annotator.sim_time.get_attribute("outputs:simulationTime").get(),
                                          "render_reference_time_s": timestamp_node.get_attribute("outputs:result").get(),
                                          "render_reference_numerator": annotator.sync_node.get_attribute("outputs:rationalTimeNumerator").get(),
                                          "render_reference_denominator": annotator.sync_node.get_attribute("outputs:rationalTimeDenominator").get(),
                                          "host_before_step_ns": host_before_step_ns,
                                          "visual_frame_id": value,
                                          "marker_authored_sim_time_s": marker_authored_sim_time_s,
                                          "marker_authored_timeline_time_s": marker_authored_timeline_time_s,
                                          "timeline_time_s": timeline.get_current_time(),
                                          "host_monotonic_ns": time.monotonic_ns()}) + "\n")
                records.flush()
                if index in (0, 30, 99):
                    for side, rgb in (("left", rgb_left), ("right", rgb_right)):
                        array = np.array(rgb.get_data()).copy()
                        if array.shape[:2] != (600, 960) or array[..., :3].std() < 5:
                            raise RuntimeError(f"Invalid {side} rendered image: {array.shape}")
                        Image.fromarray(array).save(args.output / f"{side}_{index:06d}.png")
                    np.save(args.output / f"renderer_gt_depth_{index:06d}.npy",
                            np.array(gt_depth.get_data()).copy())
                index += 1
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - before)))
        (args.output / "summary.json").write_text(json.dumps(
            {"rendered_frames": index, "wall_duration_s": time.monotonic() - start,
             "receiver_success": "must be independently verified from SDK output"}, indent=2))
    except Exception:
        # Kit's fast shutdown can exit before Python re-raises the exception.
        traceback.print_exc()
        raise
    finally:
        if observation_writer is not None:
            observation_writer.detach()
        if world is not None:
            world.stop()
        # These CPU observers must detach before the vendor destroys their RPs.
        for observer, product in ((rgb_left, "left_rp"), (rgb_right, "right_rp"),
                                  (gt_depth, "left_rp")):
            if observer is not None and annotator is not None:
                observer.detach(getattr(annotator, product))
        if annotator is not None:
            annotator.destroy()
        if world is not None:
            world.clear_instance()
        print("ZED_POC_SCENE_RELEASED", flush=True)


if __name__ == "__main__":
    main()
