"""Replace the existing park cleaner's stereo rig with one T3 centre camera."""
import hashlib
from pathlib import Path


def load_scene(scene_path, fps, *, preserve_viewport=False, move_cleaner=True):
    import omni.usd
    from pxr import Gf, UsdGeom

    context = omni.usd.get_context()
    if not context.open_stage(str(scene_path)):
        raise RuntimeError('Could not open existing park scene')
    stage = context.get_stage()
    # Headless capture owns its render product. Live preview needs the UI product.
    if not preserve_viewport:
        stage.RemovePrim('/Render')
    body = UsdGeom.Xformable(stage.GetPrimAtPath('/World/Cleaner'))
    old_rig = stage.GetPrimAtPath('/World/Cleaner/MeasuredRig')
    if not body or not old_rig:
        raise RuntimeError('Existing cleaner and ZED rig are required')
    # Match the previously delivered moving-cleaner scene: traffic frozen at first sample.
    frozen = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            times = attr.GetTimeSamples()
            if times:
                value = attr.Get(times[0])
                attr.Clear()
                attr.Set(value)
                frozen.append(str(attr.GetPath()))
    stage.RemovePrim('/World/Cleaner/MeasuredRig')
    stage.RemovePrim('/World/Cleaner/SensorRig')
    body.ClearXformOpOrder()
    body_op = body.AddTransformOp()
    camera_path = '/World/Cleaner/T3/CenterRGB'
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(.6, 0, .78), Gf.Vec3d(1.6, 0, .78), Gf.Vec3d(0, 0, 1)).GetInverse())
    initial = Gf.Matrix4d().SetTranslate(Gf.Vec3d(-48, -8.375, 0))
    body_op.Set(initial)

    def set_state(index):
        # Repeat the existing 12-second straight route for a sustained load.
        t = (index % (12 * fps)) / fps if move_cleaner else 0.0
        speed_mps = float(getattr(set_state, 'speed_mps', 1.5))
        route_s = 12.0 / max(speed_mps / 1.5, 1e-6)
        t_route = ((index % max(int(route_s * fps), 1)) / fps)
        xyz = (-48 + speed_mps * t_route, -8.375, 0)
        matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*xyz))
        body_op.Set(matrix)
        return {'source_time_s': t, 'cleaner_position_m': xyz,
                'trajectory': 'existing straight route; speed adjustable; 12m route', 'speed_mps': speed_mps, 'yaw_rad': 0.}

    set_state.speed_mps = 1.5

    provenance = {'scene': str(scene_path),
                  'scene_sha256': hashlib.sha256(Path(scene_path).read_bytes()).hexdigest(),
                  'mount': 'existing cleaner stereo midpoint [.6,0,.78] m, forward +X',
                  'frozen_attributes': frozen,
                  'trajectory': 'existing straight route, repeated every 12 seconds'}
    return stage, camera_path, set_state, provenance


class FrameMarkers:
    """Static emissive tiles; only geometric visibility changes each frame."""
    def __init__(self, stage, camera_paths, resolution=(1280,720), *, row=32):
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        width_px, height_px = resolution
        self.tiles = []
        self.centres = [[32 + i * 24, row] for i in range(23)]
        materials = []
        for name, value in (('Black', .001), ('White', 1.0)):
            root = '/World/FrameDiagnostics/' + name
            material = UsdShade.Material.Define(stage, root)
            shader = UsdShade.Shader.Define(stage, root + '/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
            shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
            shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(value))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            materials.append(material)
        for side, path in zip(('left', 'right'), camera_paths):
            camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
            pose = UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())
            near = camera.GetClippingRangeAttr().Get()[0]
            focal = camera.GetFocalLengthAttr().Get()
            fx = width_px * focal / camera.GetHorizontalApertureAttr().Get()
            fy = height_px * focal / camera.GetVerticalApertureAttr().Get()
            eye_tiles = []
            for i in range(-1, 23):
                depth = near * (1.1 if i == -1 else 1.02)
                x, width = (20, 552) if i == -1 else (20+i*24, 24)
                points = [Gf.Vec3f((u-width_px/2)*depth/fx, -(v-height_px/2)*depth/fy, -depth)
                          for u,v in ((x,row-12),(x+width,row-12),(x+width,row+12),(x,row+12))]
                group = 'FrameDiagnostics' if row == 32 else f'FrameDiagnostics{row}'
                mesh = UsdGeom.Mesh.Define(stage, f'{path}/{group}/Tile{i+1}')
                mesh.CreatePointsAttr(points)
                mesh.CreateFaceVertexCountsAttr([4])
                mesh.CreateFaceVertexIndicesAttr([0,3,2,1])
                mesh.CreateDoubleSidedAttr(True)
                mesh.CreateSubdivisionSchemeAttr('none')
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[0 if i == -1 else 1])
                if i >= 0:
                    eye_tiles.append(mesh.CreateVisibilityAttr())
            self.tiles.append(eye_tiles)
        self.set(65535)

    def set(self, value):
        payload = [(value >> bit) & 1 for bit in range(15, -1, -1)]
        bits = [1,0,1,0] + payload + [sum(payload) % 2] + [0,1]
        for tiles in self.tiles:
            for tile, bit in zip(tiles, bits):
                tile.Set('inherited' if bit else 'invisible')


