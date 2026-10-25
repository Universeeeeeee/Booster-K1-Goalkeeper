"""Optional kinematic exposure acquisition for the existing official sender.

Motion is authored kinematically. Finite exposures average synchronized rendered
samples after color encoding; they do not reproduce a measured camera ISP.
Separate pixel markers bind output frames and actual subframe render packets.
"""

import hashlib
import json
import math
import time
from pathlib import Path

from acquisition.timing import sample_time, rational_seconds, exposure_window, exposure_sample_times


class AcquisitionState:
    def __init__(self, stage, camera_root, marker, camera_pose, left_path, right_path, args):
        from pxr import Gf, UsdGeom, Vt
        self.stage, self.args, self.camera_pose = stage, args, camera_pose
        self.output_colours = marker.GetDisplayColorPrimvar()
        self.paths = {"left": left_path, "right": right_path}
        self.camera_shift = UsdGeom.Xformable(camera_root).AddTranslateOp(opSuffix="g3_motion")
        self.marker_shift = marker.AddTranslateOp(opSuffix="g3_motion")
        self.ball = UsdGeom.Sphere.Define(stage, "/World/DiagnosticBall")
        self.ball.CreateRadiusAttr(0.09)
        self.ball.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.18, 0.03)])
        self.ball_shift = self.ball.AddTranslateOp()
        self.ball_shift.Set(camera_pose.Transform(Gf.Vec3d(0, -0.15, -1.2)))
        self.subframe_marker = UsdGeom.Mesh.Define(stage, "/World/SubframeId")
        points = []
        for cell in range(23):
            x = -0.8 + cell * 0.07
            points.extend(Gf.Vec3f(camera_pose.Transform(Gf.Vec3d(a, b, -1.99)))
                          for a, b in ((x, 0.70), (x + 0.07, 0.70),
                                       (x + 0.07, 0.80), (x, 0.80)))
        self.subframe_marker.CreatePointsAttr(Vt.Vec3fArray(points))
        self.subframe_marker.CreateFaceVertexCountsAttr([4] * 23)
        self.subframe_marker.CreateFaceVertexIndicesAttr(list(range(92)))
        self.subframe_marker.CreateDoubleSidedAttr(True)
        self.subframe_marker.CreateSubdivisionSchemeAttr("none")
        self.subframe_colours = self.subframe_marker.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
        self.subframe_shift = self.subframe_marker.AddTranslateOp(opSuffix="g3_motion")
        self.set_subframe_id(65535)
        self.subframe_index = 0
        self.first_publication = True

    def metadata(self, calibration, sender_source):
        from acquisition import sdk_exposure, timing
        source_dir = self.args.output / "sources"
        source_dir.mkdir()
        source_files = {}
        for name, path in {"sender": sender_source, "sdk_state": Path(__file__),
                           "sdk_exposure": Path(sdk_exposure.__file__),
                           "timing": Path(timing.__file__)}.items():
            content = path.read_bytes()
            relative = f"sources/{name}.py"
            (self.args.output / relative).write_bytes(content)
            source_files[name] = {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}
        return {"scenario": self.args.acquisition_scenario,
                "source_files": source_files,
                "state_file": "source_frames.jsonl", "state_key": "acquisition",
                "state_clock": "authored kinematic sampling seconds; native World clock recorded separately",
                "motion_clock": "exact sampling seconds relative to origin 1 second",
                "motion_model": "analytic kinematic diagnostic, no robot dynamics",
                "motion_frequency_hz": 2 if self.args.exposure_ms else 0.5,
                "ball_motion_amplitude_m": 0.3,
                "camera_motion_amplitude_m": 0.30 if self.args.exposure_ms else 0.015,
                "realized_exposure_s": self.args.exposure_ms / 1000,
                "exposure_model": "encoded_rgba_box_average" if self.args.exposure_ms else "instantaneous_render_baseline",
                "finite_exposure_supported": True, "rolling_skew_s": 0,
                "integration_approximation": "equal midpoint samples averaged after renderer color encoding; not radiometric or real ISP equivalence",
                "samples_per_exposure": self.args.exposure_samples if self.args.exposure_ms else 1,
                "publication": "immutable CUDA RGBA through official v3.2 bridge; raw buffers retained until teardown",
                "publication_mode": "offline_render_then_replay",
                "replay_wall_period_s": 0.1,
                "prepared_plan_file": "prepared_exposures.jsonl",
                "sdk_timestamp_clock_domain": "authored kinematic exposure midpoint seconds, origin 1; no mapping to native World or host epoch",
                "subframe_marker_pixel_centres": self.marker_centres(calibration),
                "state_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "camera_usd_sha256": hashlib.sha256(self.args.camera_usd.read_bytes()).hexdigest(),
                "calibration_version": hashlib.sha256(json.dumps(
                    {"cameras": calibration, "resolution": [960, 600],
                     "right_shift_m": self.args.right_shift_m}, sort_keys=True).encode()).hexdigest(),
                "pose_convention": "USD camera row-vector world transform; optical forward -Z, up +Y"}

    def marker_centres(self, calibration):
        from pxr import Gf
        import numpy as np
        centres = {}
        for side, camera in calibration.items():
            matrix = np.asarray(camera["world_from_usd_camera_row_matrix"])
            inverse = np.linalg.inv(matrix)
            centres[side] = []
            for cell in range(23):
                world = self.camera_pose.Transform(Gf.Vec3d(-0.8 + (cell + 0.5) * 0.07, 0.75, -1.99))
                local = np.asarray([*world, 1]) @ inverse
                centres[side].append([480 + camera["fx_px"] * local[0] / -local[2],
                                      300 - camera["fx_px"] * local[1] / -local[2]])
        return centres

    def set_subframe_id(self, value):
        from pxr import Gf
        payload = [(value >> bit) & 1 for bit in range(15, -1, -1)]
        bits = [1, 0, 1, 0] + payload + [sum(payload) % 2] + [0, 1]
        self.subframe_colours.Set([Gf.Vec3f(1.0 if bit else 0.01) for bit in bits])

    def author(self, index, world_time, target=None):
        import numpy as np
        from pxr import Gf, UsdGeom
        target = sample_time(index, self.args.fps, 1) if target is None else target
        frequency = 2 if self.args.exposure_ms else 0.5
        phase = math.sin(2 * math.pi * frequency * float(target - 1))
        ball_x = 0.3 * phase if self.args.acquisition_scenario == "ball_motion" else 0.0
        amplitude = 0.30 if self.args.exposure_ms else 0.015
        camera_x = amplitude * phase if self.args.acquisition_scenario == "camera_motion" else 0.0
        displacement = self.camera_pose.TransformDir(Gf.Vec3d(camera_x, 0, 0))
        self.camera_shift.Set(displacement)
        self.marker_shift.Set(displacement)
        self.subframe_shift.Set(displacement)
        ball_world = self.camera_pose.Transform(Gf.Vec3d(ball_x, -0.15, -1.2))
        self.ball_shift.Set(ball_world)
        poses = {side: np.array(UsdGeom.XformCache().GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(path))).tolist() for side, path in self.paths.items()}
        return {"t_sim_state_s": float(target), "native_world_time_s": world_time,
                "t_exposure_start_s": float(target),
                "t_exposure_end_s": float(target), "t_exposure_mid_s": float(target),
                "exposure_model": "instantaneous_render_baseline", "rolling_skew_s": 0,
                "motion_target_s": rational_seconds(target),
                "world_from_usd_camera_row_matrix": poses,
                "ball_world_position_m": list(ball_world), "ball_radius_m": 0.09,
                "ball_optical_x_m": ball_x, "camera_optical_x_m": camera_x}

    def render_exposure(self, index, world, writer, integrator, prepare_only=False):
        from PIL import Image
        from acquisition.sdk_exposure import average_rgba
        integrator.hold()
        start = sample_time(index, self.args.fps, 1)
        window = exposure_window(start, self.args.exposure_ms, self.args.fps)
        targets = exposure_sample_times(start, self.args.exposure_ms, self.args.fps,
                                        self.args.exposure_samples)
        subframes, left, right = [], [], []
        folder = self.args.output / "subframes"
        folder.mkdir(exist_ok=True)
        for target in targets:
            sub_id = self.subframe_index
            self.subframe_index += 1
            if sub_id >= 65536:
                raise RuntimeError("Subframe diagnostic ID space exhausted")
            self.set_subframe_id(sub_id)
            authored_host = time.monotonic_ns()
            state = self.author(index, world.current_time, target)
            writer.subframe_cache.pop(sub_id, None)
            writer.expected_subframe = sub_id
            native_before = world.current_time
            for _ in range(12):
                world.render()
                if sub_id in writer.subframe_cache:
                    break
            else:
                raise RuntimeError(f"No paired Writer pixels for authored subframe {sub_id}")
            packet = writer.subframe_cache.pop(sub_id)
            writer.expected_subframe = None
            if world.current_time != native_before:
                raise RuntimeError("Native World clock advanced while rendering a held kinematic sample")
            state["native_world_time_after_render_s"] = world.current_time
            if packet["visual_id"] != index:
                raise RuntimeError("Writer output frame ID disagrees with held exposure ID")
            paths = {}
            for side, images in (("left", left), ("right", right)):
                array = packet[side]
                path = f"subframes/{side}_{sub_id:06d}.png"
                Image.fromarray(array).save(self.args.output / path)
                paths[f"{side}_path"] = path
                images.append(array)
            subframes.append({"subframe_id": sub_id, "target_s": rational_seconds(target),
                              "state": state, "reference_time": packet["reference_time"],
                              "host_authored_monotonic_ns": authored_host,
                              "host_observed_monotonic_ns": time.monotonic_ns(), **paths})
        composite = (average_rgba(left), average_rgba(right))
        (self.args.output / "composites").mkdir(exist_ok=True)
        paths = {}
        for side, array in zip(("left", "right"), composite):
            path = f"composites/{side}_{index:06d}.png"
            Image.fromarray(array).save(self.args.output / path)
            paths[side] = path
        state = dict(subframes[len(subframes) // 2]["state"])
        state.update({"t_sim_state_s": float(window["mid_s"]),
                      **{f"t_exposure_{key}": float(value) for key, value in window.items()},
                      "exposure_model": "encoded_rgba_box_average" if self.args.exposure_ms else "instantaneous_render_baseline",
                      "window_rational": {key: rational_seconds(value) for key, value in window.items()},
                      "subframes": subframes, "composite_paths": paths,
                      "sdk_composite_timestamp_ns": int(float(window["mid_s"]) * 1e9)})
        if prepare_only:
            state["host_prepared_monotonic_ns"] = time.monotonic_ns()
            return state
        return self.publish_prepared(state, world, integrator, composite)

    def publish_prepared(self, state, world, integrator, composite=None):
        if composite is None:
            from PIL import Image
            import numpy as np
            composite = tuple(np.asarray(Image.open(self.args.output / state["composite_paths"][side])).copy()
                              for side in ("left", "right"))
        host_publication = time.monotonic_ns()
        compute_before = integrator.node.get_compute_count()
        publication = integrator.publish(*composite, state["t_exposure_mid_s"])
        publication["host_armed_monotonic_ns"] = host_publication
        world.render()
        if self.first_publication:
            # The first true execution initializes the official bridge only.
            world.render()
            self.first_publication = False
        publication["native_compute_count_before"] = compute_before
        publication["native_compute_count_after"] = integrator.node.get_compute_count()
        integrator.hold()
        publication["host_after_render_monotonic_ns"] = time.monotonic_ns()
        state["publication"] = publication
        return state

    def set_output_id(self, value):
        from pxr import Gf
        payload = [(value >> bit) & 1 for bit in range(15, -1, -1)]
        bits = [1, 0, 1, 0] + payload + [sum(payload) % 2] + [0, 1]
        self.output_colours.Set([Gf.Vec3f(1.0 if bit else 0.01) for bit in bits])
