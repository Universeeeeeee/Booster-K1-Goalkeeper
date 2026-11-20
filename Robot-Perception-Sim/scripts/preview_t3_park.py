"""Live T3 three-camera park preview with GPU grayscale, SD/TD and center RGB AE."""
import argparse
import ctypes
import os
import socket
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'acquisition'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stereo-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--public-ip', required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seconds', type=int, default=0)
    parser.add_argument('--signal-port', type=int, default=int(os.environ.get('T3_SIGNAL_PORT', '49100')))
    parser.add_argument('--stream-port', type=int, default=int(os.environ.get('T3_STREAM_PORT', '1024')))
    args, _ = parser.parse_known_args()
    if args.seconds < 0:
        parser.error('seconds must be nonnegative; 0 means until stopped')
    args.output.mkdir(parents=True, exist_ok=True)
    for kind, port in ((socket.SOCK_STREAM, args.signal_port), (socket.SOCK_DGRAM, args.stream_port)):
        with socket.socket(socket.AF_INET, kind) as probe:
            if kind == socket.SOCK_STREAM:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('0.0.0.0', port))
                if kind == socket.SOCK_STREAM:
                    probe.listen()
            except OSError as exc:
                protocol = 'TCP' if kind == socket.SOCK_STREAM else 'UDP'
                raise RuntimeError(f'{protocol} {port} is occupied; previous preview may still be stopping') from exc
    cuda = ctypes.CDLL('libcuda.so.1', mode=ctypes.RTLD_GLOBAL)
    if cuda.cuInit(0):
        raise RuntimeError('CUDA initialization failed')
    from isaacsim import SimulationApp
    stream = '--/exts/omni.kit.livestream.app/primaryStream/'
    print(f'STARTING Isaac 6 WebRTC; TCP {args.signal_port}, UDP {args.stream_port}', flush=True)
    app = SimulationApp({
        'headless': True, 'hide_ui': False, 'active_gpu': args.gpu,
        'physics_gpu': 0, 'multi_gpu': False, 'limit_cpu_threads': 8,
        'width': 1920, 'height': 1080, 'window_width': 1920, 'window_height': 1080,
        'renderer': 'RaytracedLighting', 'fast_shutdown': True,
        'extra_args': ['--enable', 'omni.kit.livestream.app',
                       stream + 'publicIp=' + args.public_ip,
                       stream + f'signalPort={args.signal_port}', stream + f'streamPort={args.stream_port}',
                       stream + 'targetFps=30',
                       '--/rtx/hydra/supportMultiTickRate=false',
                       '--/rtx/ecoMode/enabled=false',
                       '--/app/renderer/skipWhileMinimized=false']})
    code = 0
    try:
        from t3_preview import SensorPreview
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        from isaacsim.core.api import World
        from PIL import Image, ImageStat
        import omni.usd
        context = omni.usd.get_context()
        sensors = None

        def restore_preview():
            nonlocal sensors
            if sensors is not None:
                sensors.close()
                World.clear_instance()
            print('LOADING T3 three-camera park scene', flush=True)
            sensors = SensorPreview(app, args.scene, args.stereo_config, args.config, args.output)
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError('No active viewport for WebRTC')
            viewport.camera_path = sensors.path
            viewport.set_texture_resolution(tuple(sensors.optics['CenterRGB']['resolution']))
            viewport.updates_enabled = True
            sensors.sample(0)
            for _ in range(5):
                app.update()
            return sensors.stage

        stage = restore_preview()
        snapshot = args.output / 'viewport.png'
        capture = capture_viewport_to_file(get_active_viewport(), str(snapshot))
        deadline = time.monotonic() + 15
        while not snapshot.exists() and time.monotonic() < deadline:
            app.update()
        if not snapshot.exists():
            raise RuntimeError('Viewport did not deliver a rendered frame')
        if max(ImageStat.Stat(Image.open(snapshot).convert('RGB')).stddev) < 2:
            raise RuntimeError(f'Viewport is blank; diagnostic image {snapshot}')
        print(f'VIEWPORT_IMAGE_VERIFIED {snapshot}', flush=True)
        import omni.kit.renderer_capture
        screen_capture = omni.kit.renderer_capture.acquire_renderer_capture_interface()
        screen_capture.capture_next_frame_swapchain(str(args.output / 'sensors_window.png'))
        for _ in range(5):
            app.update()
        with socket.create_connection(('127.0.0.1', args.signal_port), timeout=2):
            pass
        duration = f'{args.seconds}s' if args.seconds else 'until stopped'
        print(f'PREVIEW_READY camera={sensors.path}; three cameras and GPU SD/TD; duration={duration}', flush=True)
        start = time.monotonic()
        next_report = 0
        index = 1
        processed = 0
        while app.is_running() and (args.seconds == 0 or time.monotonic() - start < args.seconds):
            tick = time.monotonic()
            elapsed = tick - start
            if context.get_stage() != stage:
                if context.get_stage_loading_status()[2] > 0:
                    app.update()
                    time.sleep(.05)
                    continue
                print('STAGE_REPLACED restoring T3 three-camera preview', flush=True)
                stage = restore_preview()
                index = 1
            sensors.sample(index)
            index += 1
            processed += 1
            rate = processed / max(time.monotonic() - start, .001)
            sensors.status.text = (f'T3  |  sample clock {sensors.fps} Hz  |  actual {rate:.1f} samples/s  |  '
                                   f'source {index / sensors.fps:.2f}s  |  3 render ticks/sample')
            if elapsed >= next_report:
                print(f'PREVIEW running {elapsed:.0f}s; samples={index}; source={index / sensors.fps:.2f}s; '
                      f'actual={rate:.2f} samples/s; duration={duration}', flush=True)
                next_report += 10
            time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
        sensors.close()
        print('PREVIEW_FINISHED', flush=True)
    except BaseException:
        code = 1
        traceback.print_exc()
        raise
    finally:
        app.app.post_quit(code)
        app.close()


if __name__ == '__main__':
    main()
