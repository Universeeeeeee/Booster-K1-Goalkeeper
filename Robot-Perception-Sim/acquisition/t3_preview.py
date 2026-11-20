"""T3 preview with stereo SD/TD and center HDR processing on the GPU."""
import json
from pathlib import Path


def configure_cameras(stage, stereo, center):
    from pxr import Gf, UsdGeom

    optics = {}
    for name, offset in (('Left', stereo['baseline_m'] / 2), ('CenterRGB', 0),
                         ('Right', -stereo['baseline_m'] / 2)):
        config = center if name == 'CenterRGB' else stereo
        width, height = config['width'], config['height']
        intrinsic = ([[center['fx'], 0, center['cx']], [0, center['fy'], center['cy']], [0, 0, 1]]
                     if name == 'CenterRGB' else stereo['K'])
        path = '/World/Cleaner/T3/' + name
        camera = UsdGeom.Camera.Define(stage, path)
        transform = UsdGeom.Xformable(camera)
        transform.ClearXformOpOrder()
        transform.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(.6, offset, .78), Gf.Vec3d(1.6, offset, .78),
            Gf.Vec3d(0, 0, 1)).GetInverse())
        focal = config['focal_length_mm']
        camera.CreateFocalLengthAttr(focal)
        camera.CreateHorizontalApertureAttr(focal * width / intrinsic[0][0])
        camera.CreateVerticalApertureAttr(focal * height / intrinsic[1][1])
        camera.CreateHorizontalApertureOffsetAttr(0.)
        camera.CreateVerticalApertureOffsetAttr(0.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
        prim = camera.GetPrim()
        prim.ApplyAPI('OmniLensDistortionOpenCvPinholeAPI')
        prim.GetAttribute('omni:lensdistortion:model').Set('opencvPinhole')
        values = {'fx': intrinsic[0][0], 'fy': intrinsic[1][1],
                  'cx': float(intrinsic[0][2]), 'cy': float(intrinsic[1][2]),
                  'imageSize': Gf.Vec2i(width, height)}
        keys = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6', 's1', 's2', 's3', 's4')
        values.update(zip(keys, config['D'] + [0.] * 7))
        for key, value in values.items():
            if not prim.GetAttribute('omni:lensdistortion:opencvPinhole:' + key).Set(value):
                raise RuntimeError(f'Cannot set {name} lens attribute {key}')
        optics[name] = {'path': path, 'resolution': [width, height], 'K': intrinsic,
                        'D': config['D'], 'focal_length_mm': focal,
                        'mount_m': [.6, offset, .78]}
    return optics


class SensorPreview:
    def __init__(self, app, scene_path, stereo_config, center_config, output):
        import carb.settings
        import omni.replicator.core as rep
        import warp as wp
        from omni.replicator.core.scripts.utils import viewport_manager
        from isaacsim.core.api import World
        from pxr import Sdf
        from t3_park import load_scene
        from t3_gpu_sd import TemporalDifference

        self.app = app
        self.stereo = json.loads(Path(stereo_config).read_text())['stereo']
        self.center = json.loads(Path(center_config).read_text())
        self.fps = self.stereo['fps']
        self.output = Path(output)
        self.stage, self.path, self.set_state, _ = load_scene(
            scene_path, self.fps, preserve_viewport=True)
        self.set_state.speed_mps = 1.5
        self.optics = configure_cameras(self.stage, self.stereo, self.center)
        settings = carb.settings.get_settings()
        settings.set('/rtx/post/motionblur/enabled', False)
        settings.set('/rtx/post/aa/op', 3)
        settings.set('/rtx-transient/post/aa/limitedOps', False)
        settings.set('/rtx/dlss/frameGeneration', False)
        self.products = {}
        self.annotators = {}
        self.buffers = {}
        self.temporal = {}
        for name, optics in self.optics.items():
            width, height = optics['resolution']
            product = viewport_manager.get_render_product(
                optics['path'], (width, height), False, 'T3Preview' + name)
            product_path = product.hydra_texture.get_render_product_path()
            prim = self.stage.GetPrimAtPath(product_path)
            prim.CreateAttribute('omni:rtx:post:aa:op', Sdf.ValueTypeNames.Token).Set('dlss')
            annotator = rep.AnnotatorRegistry.get_annotator('HdrColor', device='cuda')
            annotator.attach(product_path)
            self.products[name] = product
            self.annotators[name] = annotator
        self.world = World(physics_dt=1 / self.fps, rendering_dt=1 / self.fps)
        self.world.reset()
        self.world.play()
        for _ in range(60):
            self.world.step(render=True)
        for name, annotator in self.annotators.items():
            source = annotator.get_data()
            width, height = self.optics[name]['resolution']
            if source.shape != (height, width, 4) or not source.device.is_cuda:
                raise RuntimeError(f'Unexpected {name} HDR buffer {source.shape}')
            self.buffers[name] = (
                wp.empty((height, width, 3), dtype=wp.uint8, device=source.device),
                wp.empty((height, width), dtype=wp.uint8, device=source.device),
                wp.empty((2, height, width), dtype=wp.int16, device=source.device))
            if name != 'CenterRGB':
                self.temporal[name] = TemporalDifference(self.buffers[name][1])
        device = self.annotators['CenterRGB'].get_data().device
        self.meter = wp.zeros(1, dtype=float, device=device)
        self.exposure = self.center['exposure']['initial_s']
        self.window = self.create_window()
        self.speed_model = None
        self.verification_gray = {}

    def create_window(self):
        import omni.ui as ui

        self.providers = {}
        window = ui.Window('T3 Sensors', width=1350, height=850,
                           position_x=10, position_y=35)
        with window.frame:
            with ui.VStack(spacing=6):
                self.status = ui.Label('Preparing three camera outputs', height=24)
                with ui.HStack(height=32, spacing=8):
                    ui.Label('Cleaner speed', width=120)
                    ui.Button('- 0.1', width=64, clicked_fn=lambda: self._adjust_speed(-.1))
                    self.speed_model = ui.FloatSlider(min=0.0, max=3.0, step=0.1,
                                                      width=520).model
                    self.speed_model.set_value(1.5)
                    ui.Button('+ 0.1', width=64, clicked_fn=lambda: self._adjust_speed(.1))
                    self.speed_label = ui.Label('1.5 m/s', width=80)
                    self.speed_model.add_value_changed_fn(self._speed_changed)
                with ui.HStack(spacing=10):
                    for eye in ('Left', 'CenterRGB', 'Right'):
                        with ui.VStack(spacing=5):
                            if eye == 'CenterRGB':
                                panels = [('CenterRGB', 'Center RGB  800 x 600  GPU AE')]
                            else:
                                panels = [(eye, f'{eye} grayscale  1280 x 720'),
                                          (eye + '_up', 'SD up  1 ... 255'),
                                          (eye + '_right', 'SD right  1 ... 255'),
                                          (eye + '_td', 'TD  -127 ... +127')]
                            for key, label in panels:
                                ui.Label(label, height=20)
                                provider = ui.ByteImageProvider()
                                self.providers[key] = provider
                                ui.ImageWithProvider(
                                    provider, height=300 if eye == 'CenterRGB' else 156,
                                    fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                            if eye == 'CenterRGB':
                                ui.Label(f'70 mm stereo baseline\n1.79 mm focal length\nD = {self.stereo["D"]}\nSD neutral gray = 128',
                                         height=100, word_wrap=True)
                                self.ae_status = ui.Label('', height=40)
        return window

    def _speed_changed(self, model):
        speed = float(model.as_float)
        self.set_state.speed_mps = speed
        self.speed_label.text = f'{speed:.1f} m/s'

    def _adjust_speed(self, delta):
        value = min(3.0, max(0.0, float(self.speed_model.as_float) + delta))
        self.speed_model.set_value(round(value, 1))

    def sample(self, index):
        import numpy as np
        import warp as wp
        from t3_gpu_sd import images, differences
        from t3_gpu_color import convert

        self.set_state(index)
        for _ in range(3):
            self.world.step(render=True)
        sources = {name: annotator.get_data() for name, annotator in self.annotators.items()}
        exposure = self.exposure
        self.meter.zero_()
        for name, source in sources.items():
            rgb, gray, difference = self.buffers[name]
            height, width = source.shape[:2]
            if name == 'CenterRGB':
                scale = exposure / self.center['exposure']['reference_exposure_s']
                wp.launch(convert, (height, width), inputs=[source, rgb, self.meter, scale],
                          device=source.device)
            else:
                wp.launch(images, (height, width), inputs=[source, rgb, gray], device=source.device)
                wp.launch(differences, (height, width), inputs=[gray, difference], device=source.device)
                self.temporal[name].sample(gray, index)
        data = {}
        for name, (rgb, gray, difference) in self.buffers.items():
            if name == 'CenterRGB':
                data[name] = rgb.numpy()
            else:
                data[name] = gray.numpy()
                data[name + '_sd'] = difference.numpy()
                data[name + '_td'] = self.temporal[name].output.numpy()
        brightness = float(self.meter.numpy()[0]) / ((self.center['height'] - 120) * self.center['width'])
        settings = self.center['exposure']
        ratio = np.clip(settings['target'] / max(brightness, 1e-12),
                        1 / settings['max_step_ratio'], settings['max_step_ratio'])
        self.exposure = float(np.clip(exposure * ratio, settings['min_s'], settings['max_s']))
        if index in (0, 1):
            self.verify_sample(sources, data, exposure, index)
            self.verification_gray = {eye: data[eye] for eye in self.temporal} if index == 0 else {}
        self.update_images(data)
        self.ae_status.text = f'Exposure {exposure * 1000:.3f} ms\nLinear mean {brightness:.3f}'
        return data

    def update_images(self, data):
        import cv2
        import numpy as np

        for key, provider in self.providers.items():
            if key.endswith('_up') or key.endswith('_right'):
                eye, channel = key.rsplit('_', 1)
                plane = data[eye + '_sd'][0 if channel == 'up' else 1]
                image = plane.astype(np.uint8)
            elif key.endswith('_td'):
                image = (data[key].astype(np.int16) + 128).astype(np.uint8)
            else:
                image = data[key]
            size = (400, 300) if key == 'CenterRGB' else (400, 225)
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA if image.ndim == 3 else cv2.COLOR_GRAY2RGBA)
            provider.set_bytes_data(rgba.ravel().tolist(), size)

    def verify_sample(self, sources, data, exposure, index):
        import numpy as np
        from PIL import Image
        from robot_perception_sim.t3_sdtd import TD_ENCODING, sd, td

        audit = {'optics': self.optics, 'sampling_hz': self.fps, 'render_ticks_per_sample': 3,
                 'stereo_gray': 'round(clip(linear luminance, 0, 1) * 255), fixed gain 1',
                 'center_processing': 'GPU HDR exposure and sRGB, causal AE',
                 'sd_encoding': 'half_trunc_plus_128',
                 'td_encoding': TD_ENCODING, 'sample_index': index,
                 'sd_channels': ['gray[y,x] - gray[y-1,x]', 'gray[y,x] - gray[y,x+1]'],
                 'exposure_s': exposure, 'checks': {}}
        for eye in ('Left', 'Right'):
            source = sources[eye].numpy().astype(np.float32)
            expected_gray = np.rint(np.clip(source[..., :3] @ np.array([.2126, .7152, .0722], np.float32), 0, 1) * 255).astype(np.uint8)
            if not np.array_equal(expected_gray, data[eye]):
                raise RuntimeError(f'{eye} GPU grayscale differs from CPU reference')
            expected_sd = np.stack(sd(data[eye])).astype(np.int16)
            if not np.array_equal(expected_sd, data[eye + '_sd']):
                raise RuntimeError(f'{eye} GPU SD differs from NumPy reference')
            previous = self.verification_gray.get(eye) if index == 1 else None
            actual_td = data[eye + '_td']
            if actual_td.dtype != np.int8 or not np.array_equal(actual_td, td(data[eye], previous)):
                raise RuntimeError(f'{eye} GPU TD differs from NumPy reference')
            audit['checks'][eye] = {'gray_exact': True, 'sd_exact': True, 'td_exact': True,
                                   'nonzero_sd': int(np.count_nonzero(data[eye + '_sd'] != 128))}
            Image.fromarray(data[eye]).save(self.output / (eye + '_gray.png'))
            np.save(self.output / (eye + '_sd.npy'), data[eye + '_sd'])
            np.save(self.output / (eye + '_td.npy'), actual_td)
        source = sources['CenterRGB'].numpy().astype(np.float32)
        linear = np.clip(source[..., :3] * (exposure / self.center['exposure']['reference_exposure_s']), 0, 1)
        expected = np.rint(np.where(linear <= .0031308, linear * 12.92,
                                   1.055 * linear ** (1 / 2.4) - .055) * 255).astype(np.uint8)
        maximum_error = int(np.abs(expected.astype(np.int16) - data['CenterRGB']).max())
        if maximum_error > 1 or float(data['CenterRGB'].std()) < 2:
            raise RuntimeError('Center GPU RGB output failed image verification')
        audit['checks']['CenterRGB'] = {'max_error_dn': maximum_error, 'std': float(data['CenterRGB'].std())}
        self.status.text = f'T3  |  sample clock {self.fps} Hz  |  three cameras ready'
        Image.fromarray(data['CenterRGB']).save(self.output / 'CenterRGB.png')
        (self.output / 'sensors.json').write_text(json.dumps(audit, indent=2))
        print(f'THREE_CAMERA_OUTPUT_VERIFIED {self.output}', flush=True)

    def close(self):
        self.window.visible = False
        for annotator in self.annotators.values():
            annotator.detach()
        for product in self.products.values():
            product.hydra_texture.set_updates_enabled(False)
