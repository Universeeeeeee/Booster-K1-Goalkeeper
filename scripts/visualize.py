"""Render a static preview, or open the native MuJoCo viewer (Space pauses)."""
import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
from k1_goalkeeper.scene import build_scene, load_config
from k1_goalkeeper.scene_simulation import SceneSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-mode", choices=["free", "fixed", "absent"], default="free")
    parser.add_argument("--control-mode", choices=["zero_torque", "k1_walk"], default="zero_torque")
    parser.add_argument("--velocity", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=[0, 0, 0],
                        help="Body velocity command after 2s standing; k1_walk only")
    parser.add_argument("--command-duration", type=float, default=2, help="Motion seconds before automatic stop")
    parser.add_argument("--camera", choices=["overview", "goal_front", "robot_side"], default="overview")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, help="PNG static preview, without advancing physics")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--paused", action="store_true", help="Start viewer paused; Space toggles")
    args = parser.parse_args()
    if not math.isfinite(args.command_duration) or args.command_duration <= 0:
        parser.error("--command-duration must be positive and finite")
    if any(args.velocity) and args.control_mode != "k1_walk":
        parser.error("--velocity requires --control-mode k1_walk")
    if args.control_mode == "k1_walk" and args.robot_mode != "free":
        parser.error("k1_walk requires --robot-mode free")
    model = mujoco.MjModel.from_xml_path(str(build_scene(robot_mode=args.robot_mode, config=load_config(args.config))))
    simulation = SceneSimulation(model, control_mode=args.control_mode)
    data = simulation.data
    if simulation.controller is not None:
        simulation.set_velocity(args.velocity)  # Validate before opening the viewer.
        simulation.set_velocity((0, 0, 0))
    print(f"Robot mode: {args.robot_mode}; control mode: {args.control_mode}. "
          "Space pauses; R resets. Static preview is not balance validation.")
    if args.output:
        from PIL import Image
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            renderer.update_scene(data, camera=args.camera)
            Image.fromarray(renderer.render()).save(args.output)
        print(args.output.resolve())
        return
    from mujoco import viewer as mjviewer
    paused = [args.paused]
    reset_requested = [False]

    def key_callback(keycode):
        if keycode == 32:
            paused[0] = not paused[0]
        elif keycode in (82, 114):
            reset_requested[0] = True

    with mjviewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera(args.camera).id
        while viewer.is_running():
            start = time.monotonic()
            if reset_requested[0]:
                simulation.reset()
                reset_requested[0] = False
            if not paused[0]:
                for _ in range(10):
                    if simulation.controller is not None:
                        command = args.velocity if 2 <= data.time < 2 + args.command_duration else (0, 0, 0)
                        simulation.set_velocity(command)
                    simulation.step()
            viewer.sync()
            time.sleep(max(0, .01 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()
