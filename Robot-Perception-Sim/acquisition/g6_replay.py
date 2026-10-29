"""Head-camera visual replay of frozen G4 states through the official SDK path."""

import hashlib
import json
import math
from pathlib import Path

from acquisition.sdk_state import AcquisitionState
from acquisition.timing import rational_seconds, sample_time

PREROLL_FRAMES = 30
POSTROLL_FRAMES = 60


def target_midpoints(duration_s, fps):
    """Both exposure conditions share centers; append an endpoint-adjacent window."""
    last = duration_s - .005
    if last < .005:
        raise ValueError("Frozen replay must contain at least a complete 10 ms window")
    count = math.floor((last - .005) * fps + 1e-9) + 1
    result = [.005 + index / fps for index in range(count)]
    if last - result[-1] > 1e-9:
        result.append(last)
    return result


class G6ReplayState(AcquisitionState):
    def __init__(self, stage, camera_root, marker, camera_pose, left_path, right_path, args, scene):
        super().__init__(stage, camera_root, marker, camera_pose, left_path, right_path, args)
        stage.RemovePrim("/World/DiagnosticBall")
        self.scene = scene
        marker.GetVisibilityAttr().Set("invisible")
        self.subframe_marker.GetVisibilityAttr().Set("invisible")
        self._build_eye_markers()
        self.target_exposure_ms = args.exposure_ms
        self.midpoints = target_midpoints(scene.duration_s, args.fps)
        self.target_ids = list(range(PREROLL_FRAMES, PREROLL_FRAMES + len(self.midpoints)))
        args.acquisition_frames = PREROLL_FRAMES + len(self.midpoints) + POSTROLL_FRAMES
        self.sequence_id = f"{scene.trajectory_name}:{scene.head_pose}:{args.exposure_ms:g}ms"
        self.receiver_ready_evidence = None

    def _build_eye_markers(self):
        """Rendered per-eye diagnostic strips, outside the moving sphere's central ROI."""
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        import numpy as np

        self.eye_marker_centers = {"output": {}, "subframe": {}}
        colors = {"output": {}, "subframe": {}}
        self.marker_depths_m = {}
        materials = {"output": [], "subframe": []}
        self.marker_emission_inputs = {"output": [], "subframe": []}
        for kind in materials:
            for cell in range(23):
                path = f"/World/G6Diagnostics/Materials/{kind}_{cell}"
                material = UsdShade.Material.Define(self.stage, path)
                shader = UsdShade.Shader.Define(self.stage, path + "/Surface")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
                shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
                shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
                emission = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
                emission.Set(Gf.Vec3f(.01))
                shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                materials[kind].append(material)
                self.marker_emission_inputs[kind].append(emission)
        for side,path in self.paths.items():
            camera = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            pose = UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())
            focal = camera.GetFocalLengthAttr().Get()
            fx = 960 * focal / camera.GetHorizontalApertureAttr().Get()
            fy = 600 * focal / camera.GetVerticalApertureAttr().Get()
            depth = float(camera.GetClippingRangeAttr().Get()[0]) * 1.02
            self.marker_depths_m[side] = depth
            for kind,y0 in (("output",20), ("subframe",45)):
                mesh = UsdGeom.Mesh.Define(self.stage, f"/World/G6Diagnostics/{side}/{kind}")
                points, centers = [], []
                for cell in range(23):
                    x0 = 20 + cell * 12
                    for x,y in ((x0,y0), (x0+12,y0), (x0+12,y0+12), (x0,y0+12)):
                        points.append(Gf.Vec3f(pose.Transform(Gf.Vec3d((x-480)*depth/fx,
                                                                 -(y-300)*depth/fy,-depth))))
                    centers.append(np.asarray(pose.Transform(Gf.Vec3d((x0+6-480)*depth/fx,
                                                                       -(y0+6-300)*depth/fy,-depth))).tolist())
                mesh.CreatePointsAttr(Vt.Vec3fArray(points))
                mesh.CreateFaceVertexCountsAttr([4]*23)
                # Pixel Y grows downward, so reverse the corner winding to emit toward the eye.
                mesh.CreateFaceVertexIndicesAttr([i+corner for i in range(0,92,4) for corner in (0,3,2,1)])
                mesh.CreateDoubleSidedAttr(True)
                mesh.CreateSubdivisionSchemeAttr("none")
                binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
                for cell,material in enumerate(materials[kind]):
                    subset = binding.CreateMaterialBindSubset(f"cell_{cell}", [cell], UsdGeom.Tokens.face)
                    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
                binding.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.nonOverlapping)
                colors[kind][side] = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
                self.eye_marker_centers[kind][side] = centers
        self.output_colours = colors["output"]["left"]
        self.right_output_colours = colors["output"]["right"]
        self.subframe_colours = colors["subframe"]["left"]
        self.right_subframe_colours = colors["subframe"]["right"]
        self.set_output_id(65535)
        self.set_subframe_id(65535)

    def _project_marker_centers(self, kind, calibration):
        import numpy as np

        result = {}
        for side,camera in calibration.items():
            inverse = np.linalg.inv(np.asarray(camera["world_from_usd_camera_row_matrix"]))
            result[side] = []
            for point in self.eye_marker_centers[kind][side]:
                local = np.r_[point,1] @ inverse
                result[side].append([camera["cx_px"]+camera["fx_px"]*local[0]/-local[2],
                                     camera["cy_px"]-camera["fy_px"]*local[1]/-local[2]])
        return result

    def marker_centres(self, calibration):
        return self._project_marker_centers("subframe", calibration)

    def frame_marker_centres(self, calibration):
        return self._project_marker_centers("output", calibration)

    def set_output_id(self, value):
        super().set_output_id(value)
        if hasattr(self, "right_output_colours"):
            self.right_output_colours.Set(self.output_colours.Get())
            for emission,color in zip(self.marker_emission_inputs["output"],self.output_colours.Get()):
                emission.Set(color)

    def set_subframe_id(self, value):
        super().set_subframe_id(value)
        if hasattr(self, "right_subframe_colours"):
            self.right_subframe_colours.Set(self.subframe_colours.Get())
            for emission,color in zip(self.marker_emission_inputs["subframe"],self.subframe_colours.Get()):
                emission.Set(color)

    def phase(self, index):
        if index < PREROLL_FRAMES:
            return "preroll", None
        target = index - PREROLL_FRAMES
        return ("target", target) if target < len(self.midpoints) else ("postroll", None)

    def metadata(self, calibration, sender_source):
        from acquisition import g6_scene

        metadata = super().metadata(calibration, sender_source)
        for key in ("motion_frequency_hz", "ball_motion_amplitude_m", "camera_motion_amplitude_m"):
            metadata.pop(key)
        for name, path in (("g6_replay", Path(__file__)), ("g6_scene", Path(g6_scene.__file__)),
                           ("frozen_g4_trajectory", self.scene.trajectory_path),
                           ("mount_config", self.scene.mount_config_path)):
            content = path.read_bytes()
            relative = f"sources/{name}{path.suffix}"
            (self.args.output / relative).write_bytes(content)
            metadata["source_files"][name] = {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}
        metadata.update({"scenario": "g6_replay", "sequence_id": self.sequence_id,
                         "motion_model": "nearest frozen G4 1ms state; no interpolation or new collision response",
                         "state_clock": "frozen G4 physical seconds from shot origin; SDK and native World clocks separately recorded",
                         "motion_clock": "requested physical midpoint list plus subframe offsets; nearest frozen sample realized time recorded",
                         "sdk_timestamp_clock_domain": "immutable publication-window seconds from SDK origin1; physical shot clock mapping is explicit per state",
                         "source_trajectory_sha256": hashlib.sha256(self.scene.trajectory_path.read_bytes()).hexdigest(),
                         "source_trajectory_duration_s": self.scene.duration_s,
                         "head_pose": self.scene.head_pose, "ball_appearance": "frozen_g1_plain_white_no_checker_texture",
                         "preroll_frames": PREROLL_FRAMES, "postroll_frames": POSTROLL_FRAMES,
                         "target_visual_ids": self.target_ids, "target_physical_midpoints_s": self.midpoints,
                         "target_physical_intervals_s": [b-a for a,b in zip(self.midpoints,self.midpoints[1:])],
                         "endpoint_policy": "final midpoint duration-5ms appended if distinct; final interval may be shorter",
                         "source_observations_file": "source_observations.jsonl",
                         "warmup_policy": "30 instantaneous fixed-start frames; require receiver metadata and 3 grabs before targets",
                         "postroll_policy": "60 instantaneous fixed-end frames; excluded from target denominator",
                         "diagnostic_markers": {"geometry": "independent per-eye rendered near-clip strips",
                                                "material": "literal per-cell emissive PreviewSurface; no scene-light dependence",
                                                "depths_m": self.marker_depths_m,
                                                "pixel_bounds": {"output": [20,20,296,32], "subframe": [20,45,296,57]},
                                                "cross_eye_scope": "other-eye near strip lies outside the pinhole frustum; verify actual per-eye pixels"},
                         "replay_wall_period_s": .2,
                         "motion_clock_mapping": "target midpoint list plus subframe offset from exposure midpoint; SDK publication clock recorded separately",
                         "state_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         "scene_provenance": self.scene.provenance()})
        return metadata

    def author(self, index, world_time, target=None):
        from pxr import UsdGeom
        import numpy as np

        target = sample_time(index, self.args.fps, 1) if target is None else target
        phase, target_index = self.phase(index)
        if phase == "target":
            midpoint = self.midpoints[target_index]
            offset = float(target - sample_time(index, self.args.fps, 1))
            requested = midpoint + offset - self.target_exposure_ms / 2000
            start, end = midpoint - self.target_exposure_ms / 2000, midpoint + self.target_exposure_ms / 2000
            next_time = (self.midpoints[target_index + 1] if target_index + 1 < len(self.midpoints)
                         else self.scene.duration_s)
            interval_end = min(self.scene.samples, key=lambda sample: abs(sample["time"] - next_time))["time"]
        else:
            midpoint = 0. if phase == "preroll" else self.scene.duration_s
            requested = -.001 if phase == "preroll" else self.scene.duration_s + .001
            start = end = midpoint
            interval_end = midpoint
        state = self.scene.author_state(requested)
        poses = {side: np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(path))).tolist() for side,path in self.paths.items()}
        state.update({"phase": phase, "target_index": target_index, "source_frame": index,
                      "sequence_id": self.sequence_id, "desired_physical_midpoint_s": midpoint,
                      "trajectory_time_s": state["realized_source_time_s"],
                      "trajectory_requested_time_s": state["requested_source_time_s"],
                      "trajectory_realized_time_s": state["realized_source_time_s"],
                      "source_trajectory_sha256": state["trajectory_sha256"],
                      "exposure_ms": self.args.exposure_ms, "trial_exposure_ms": self.target_exposure_ms,
                      "time_start_s": start, "time_end_s": end,
                      "observation_interval_end_s": interval_end,
                      "observation_interval_basis": "next realized midpoint; terminal interval ends at frozen record endpoint",
                      "t_sim_state_s": float(target), "native_world_time_s": world_time,
                      "t_exposure_start_s": float(target), "t_exposure_mid_s": float(target),
                      "t_exposure_end_s": float(target), "rolling_skew_s": 0,
                      "motion_target_s": rational_seconds(target),
                      "world_from_usd_camera_row_matrix": poses,
                      "visibility_basis": "independent optical frustum and 9 USD mesh surface rays; finite sampling/non-Mesh omission recorded"})
        return state

    def render_exposure(self, index, world, writer, integrator, prepare_only=False):
        # Pre/postroll initialize/flush transport; only target images model10ms exposure.
        phase, _ = self.phase(index)
        exposure = self.args.exposure_ms
        if phase != "target":
            self.args.exposure_ms = 0
        try:
            state = super().render_exposure(index, world, writer, integrator, prepare_only)
        finally:
            self.args.exposure_ms = exposure
        sub_states = [sample["state"] for sample in state["subframes"]]
        state.update({"subframe_times_s": [s["requested_source_time_s"] for s in sub_states],
                      "subframe_realized_times_s": [s["realized_source_time_s"] for s in sub_states],
                      "subframe_quantization_errors_s": [s["quantization_error_s"] for s in sub_states],
                      "any_visible": any(s["expected_visible"] for s in sub_states),
                      "all_visible": all(s["expected_visible"] for s in sub_states),
                      "any_partial_occlusion": any(s["visibility_reason"] == "partially_occluded" for s in sub_states),
                      "maximum_subframe_time_error_s": max(abs(s["quantization_error_s"]) for s in sub_states)})
        if prepare_only and phase == "target":
            row = {key: state[key] for key in (
                "source_frame", "sequence_id", "trajectory_name", "source_time_s", "trajectory_time_s",
                "head_pose",
                "desired_physical_midpoint_s", "trajectory_requested_time_s", "trajectory_realized_time_s",
                "exposure_ms", "time_start_s", "time_end_s", "subframe_times_s", "subframe_realized_times_s",
                "observation_interval_end_s", "observation_interval_basis",
                "subframe_quantization_errors_s", "maximum_subframe_time_error_s", "phase", "target_index",
                "ball_world_position_m", "ball_center_camera_m", "ball_radius_m", "T_world_camera",
                "expected_visible", "visibility_reason", "visibility_basis", "visibility_evidence",
                "any_visible", "all_visible", "any_partial_occlusion", "source_trajectory_sha256")}
            row["subframe_visibility"] = [{key: s[key] for key in (
                "requested_source_time_s", "realized_source_time_s", "expected_visible",
                "visibility_reason", "visibility_evidence")} for s in sub_states]
            with (self.args.output / "source_observations.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return state

    def publish_prepared(self, state, world, integrator, composite=None):
        if state["phase"] == "target" and state["target_index"] == 0:
            ready = self.args.g6_receiver_ready
            frames = ready.parent / "frames.jsonl"
            if not ready.is_file() or not frames.is_file():
                raise RuntimeError("G6 receiver did not open before the target sequence")
            received = [json.loads(line) for line in frames.read_text().splitlines()]
            if len(received) < 3:
                raise RuntimeError("G6 receiver did not grab at least3 pre-roll frames before targets")
            sdk = json.loads(ready.read_text())
            if sdk["depth_mode"] != "NEURAL":
                raise RuntimeError("G6 requires actual SDK NEURAL mode")
            self.receiver_ready_evidence = {"receiver_metadata_sha256": hashlib.sha256(ready.read_bytes()).hexdigest(),
                                            "preroll_grabs_before_target": len(received)}
        if self.receiver_ready_evidence is not None:
            state["receiver_ready_evidence"] = self.receiver_ready_evidence
        return super().publish_prepared(state, world, integrator, composite)
