"""Independent static-ball states; only rendered RGB reaches the ZED bridge."""

import math

from acquisition.sdk_state import AcquisitionState


DEPTHS_M = (1.0, 1.5, 2.0, 3.0, 4.0)
CASES = [(depth, angle) for depth in DEPTHS_M for angle in ("left", "center", "right")]
CASES += [(2.0, "out_of_fov"), (2.0, "occluded")]


class StaticBallState(AcquisitionState):
    def __init__(self, *args):
        import numpy as np
        from pxr import Gf, UsdGeom, Vt

        super().__init__(*args)
        self.stage.RemovePrim("/World/DiagnosticBall")
        self.ball = UsdGeom.Mesh.Define(self.stage, "/World/DiagnosticBall")
        points, indices, colours = [], [], []
        rng = np.random.default_rng(20261029)
        coarse = np.random.default_rng(20261030).random((8, 16)) < 0.3
        for row in range(32):
            for col in range(64):
                offset = len(points)
                for rr, cc in ((row, col), (row + 1, col), (row + 1, col + 1), (row, col + 1)):
                    phi, theta = math.pi * rr / 32, 2 * math.pi * cc / 64
                    points.append(Gf.Vec3f(0.11 * math.sin(phi) * math.cos(theta),
                                          0.11 * math.cos(phi), 0.11 * math.sin(phi) * math.sin(theta)))
                indices.extend(range(offset, offset + 4))
                brightness = rng.uniform(0.45, 1)
                if self.args.g5_texture == "coarse_contrast":
                    colours.append(Gf.Vec3f(0.002) if coarse[row // 4, col // 4]
                                   else Gf.Vec3f(1, 0.34, 0.025))
                else:
                    colours.append(Gf.Vec3f(brightness, brightness * 0.34, brightness * 0.025))
        self.ball.CreatePointsAttr(Vt.Vec3fArray(points))
        self.ball.CreateFaceVertexCountsAttr([4] * len(colours))
        self.ball.CreateFaceVertexIndicesAttr(indices)
        self.ball.CreateDoubleSidedAttr(True)
        self.ball.CreateSubdivisionSchemeAttr("none")
        self.ball.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(colours)
        self.ball_shift = self.ball.AddTranslateOp()
        self.occluder = UsdGeom.Cube.Define(self.stage, "/World/Occluder")
        self.occluder.CreateSizeAttr(1)
        self.occluder.CreateDisplayColorAttr([Gf.Vec3f(0.12)])
        transform = self.occluder.AddTransformOp()
        local = Gf.Matrix4d(1)
        local.SetScale(Gf.Vec3d(0.6, 0.6, 0.08))
        local.SetTranslateOnly(Gf.Vec3d(0, 0, -1.5))
        transform.Set(local * self.camera_pose)

    def metadata(self, calibration, sender_source):
        import hashlib
        from pathlib import Path

        metadata = super().metadata(calibration, sender_source)
        content = Path(__file__).read_bytes()
        (self.args.output / "sources/g5_static.py").write_bytes(content)
        metadata["source_files"]["g5_static"] = {
            "path": "sources/g5_static.py", "sha256": hashlib.sha256(content).hexdigest()}
        metadata.update({"scenario": "g5_static", "ball_radius_m": 0.11,
                         "depths_m": DEPTHS_M, "angles": ["left", "center", "right"],
                         "frames_per_case": 6, "case_cycles": 2,
                         "sphere_texture": self.args.g5_texture,
                         "background_optical_depth_m": 6.0,
                         "motion_model": "static sphere held for six samples per case",
                         "replay_wall_period_s": 0.2})
        return metadata

    def author(self, index, world_time, target=None):
        import numpy as np
        from pxr import Gf, UsdGeom

        state = super().author(index, world_time, target)
        depth, angle = CASES[(index // 6) % len(CASES)]
        x = {"left": -0.4 * depth, "center": 0, "right": 0.4 * depth,
             "out_of_fov": 2.5 * depth, "occluded": 0}[angle]
        world = self.camera_pose.Transform(Gf.Vec3d(x, 0, -depth))
        self.ball_shift.Set(world)
        self.occluder.GetVisibilityAttr().Set("inherited" if angle == "occluded" else "invisible")
        matrix = np.asarray(self.camera_pose)
        # USD uses row vectors, forward -Z/up +Y; runtime uses column optical +Z/down +Y.
        optical = matrix.T @ np.diag([1, -1, -1, 1])
        state.update({"depth_label_m": depth, "angle_label": angle,
                      "ball_center_camera_m": [x, 0, depth],
                      "ball_world_position_m": list(world), "ball_radius_m": 0.11,
                      "T_world_camera": optical.tolist(),
                      "expected_visible": angle not in ("out_of_fov", "occluded")})
        return state
