#!/usr/bin/env python3
"""Run the frozen G1 cases against a generated USD in native Isaac Sim 4.5.

Run with Isaac's python.sh; source GoalDetector is imported unchanged from
--source-root. All samples come from initialized PhysX tensor views.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import sys


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--robot-scene", type=Path)
    parser.add_argument("--source-export", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    sys.path.insert(0, str(args.source_root))
    from k1_goalkeeper.detection import GoalDetector

    # Native Isaac startup requires CUDA driver initialization on this host.
    cuda_result = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL).cuInit(0)
    if cuda_result != 0:
        raise RuntimeError(f"cuInit returned {cuda_result}")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "active_gpu": args.gpu,
                         "physics_gpu": args.gpu, "multi_gpu": False,
                         "fast_shutdown": True})
    import numpy as np
    import omni.usd
    from omni.physx import get_physx_simulation_interface, get_physx_interface
    from omni.physx.bindings._physx import ContactEventType
    from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics
    from isaacsim.core.api import SimulationContext
    from isaacsim.core.prims import SingleRigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import open_stage

    dt = protocol["timestep_s"]
    records_summary = []
    engine_readback = None
    runtime_ball = None
    completed = False

    def open_sim(path):
        open_stage(str(path.resolve()))
        app.update()
        stage = omni.usd.get_context().get_stage()
        ball_prim = next(p for p in stage.Traverse() if p.GetName().endswith("_ball")
                         and p.HasAPI(UsdPhysics.RigidBodyAPI))
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0)
        sim = SimulationContext(physics_dt=dt, rendering_dt=dt,
                                physics_prim_path="/World/PhysicsScene", backend="numpy", device="cpu")
        sim.initialize_physics()
        sim.play()
        ball = SingleRigidPrim(str(ball_prim.GetPath()))
        ball.initialize()
        # Native initialization may restore 60Hz defaults; apply after handles exist.
        sim.set_simulation_dt(physics_dt=dt, rendering_dt=dt)

        return stage, sim, ball

    def close_sim(sim):
        sim.stop()
        SimulationContext.clear_instance()

    try:
        for shot in ([] if args.smoke_only else protocol["cases"]):
            hashes = []
            for repeat in range(max(2, protocol["repeats"])):
                stage, sim, ball = open_sim(args.scene)
                ball.set_world_pose(np.array(shot["position"], dtype=float), np.array([1., 0., 0., 0.]))
                ball.set_linear_velocity(np.array(shot["velocity"], dtype=float))
                ball.set_angular_velocity(np.array(shot["angular_velocity"], dtype=float))
                # No warmup step: preserves the frozen shot's initial conditions.
                runtime_ball = {"mass_kg": np.asarray(ball._rigid_prim_view.get_masses()).tolist(),
                                "inertia_kg_m2": np.asarray(ball._rigid_prim_view.get_inertias()).tolist(),
                                "source": "initialized PhysX tensor rigid body view",
                                "context_dt_getter_s": sim.get_physics_dt(),
                                "native_simulate_dt_argument_s": dt,
                                "gravity_readback": sim.get_physics_context().get_gravity()}
                ball_path = str(ball.prim_path)
                pending = []

                def contact_callback(headers, data):
                    for header in headers:
                        actors = [str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                                  str(PhysicsSchemaTools.intToSdfPath(header.actor1))]
                        if ball_path not in actors:
                            continue
                        colliders = [str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                                     str(PhysicsSchemaTools.intToSdfPath(header.collider1))]
                        other = colliders[1] if actors[0] == ball_path else colliders[0]
                        points = []
                        for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                            c = data[index]
                            points.append({"position": list(c.position), "normal": list(c.normal),
                                           "impulse": list(c.impulse), "separation": float(c.separation)})
                        event_type = ("CONTACT_FOUND" if header.type == ContactEventType.CONTACT_FOUND else
                                      "CONTACT_LOST" if header.type == ContactEventType.CONTACT_LOST else "CONTACT_PERSISTS")
                        pending.append({"type": event_type, "other_collider": other, "points": points})

                subscription = get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
                delivered_steps = []
                step_subscription = get_physx_interface().subscribe_physics_step_events(lambda step_dt: delivered_steps.append(float(step_dt)))

                def sample(t):
                    pos, quat = ball.get_world_pose()
                    velocity = ball.get_linear_velocity()
                    angular = np.asarray(ball.get_angular_velocity(), dtype=float)
                    # Rotate world angular velocity into body's coordinates.
                    w, x, y, z = np.asarray(quat, dtype=float)
                    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
                    result = {"time": t, "position": np.asarray(pos).tolist(),
                              "quaternion_wxyz": np.asarray(quat).tolist(),
                              "velocity": np.asarray(velocity).tolist(),
                              "angular_velocity_body": (rotation.T @ angular).tolist(),
                              "angular_velocity_world": angular.tolist()}
                    if not all(np.isfinite(result[key]).all() for key in ("position", "velocity", "angular_velocity_world")):
                        raise RuntimeError("Non-finite PhysX state")
                    return result

                clock_origin = sim.current_time
                delivered_time = 0.
                initial = sample(0.)
                trajectory = [initial]
                contacts, lifetimes, raw_events = [], [], []
                active = {}
                detector = GoalDetector(radius=protocol["ball_radius_m"])
                event = detector.update(0., initial["position"])
                for step in range(math.ceil(shot["duration"] / dt)):
                    if event:
                        break
                    before = trajectory[-1]
                    pending.clear()
                    delivered_steps.clear()
                    get_physx_simulation_interface().simulate(dt, delivered_time)
                    get_physx_simulation_interface().fetch_results()
                    if len(delivered_steps) != 1:
                        raise RuntimeError(f"Expected one native physics step event, got {delivered_steps}")
                    delivered_time += delivered_steps[0]
                    after = sample(delivered_time)
                    after["simulation_context_clock_s"] = sim.current_time - clock_origin
                    actual_step = after["time"] - before["time"]
                    if not math.isclose(actual_step, dt, rel_tol=0, abs_tol=1e-9):
                        raise RuntimeError(f"Actual simulation clock step {actual_step} != {dt}")
                    if shot["name"] == "free_fall" and step == 0:
                        acceleration = (after["velocity"][2] - before["velocity"][2]) / actual_step
                        if not math.isclose(acceleration, -9.81, rel_tol=0, abs_tol=.01):
                            raise RuntimeError(f"Actual free-fall acceleration {acceleration} inconsistent with gravity")
                    for contact in pending:
                        raw_events.append({"time_before_step": before["time"], "time_after_step": after["time"], **contact})
                        other = contact["other_collider"]
                        name = other.rsplit("/", 1)[-1]
                        if name.startswith("g") and "_" in name:
                            name = name.split("_", 1)[1]
                        kind = "ground" if name == "ground" else "goal"
                        if "FOUND" in contact["type"] or ("PERSISTS" in contact["type"] and other not in active):
                            active[other] = after["time"]
                            contacts.append({"time": before["time"], "kind": kind, "geom": name,
                                             "position": contact["points"][0]["position"] if contact["points"] else None,
                                             "onset_observation": contact["type"],
                                             "ball_velocity_before": before["velocity"],
                                             "ball_velocity_after": after["velocity"]})
                        elif "LOST" in contact["type"]:
                            onset = active.pop(other, None)
                            lifetimes.append({"geom": name, "found_sample_time_s": onset,
                                              "lost_sample_time_s": after["time"],
                                              "duration_s": after["time"] - onset if onset is not None else None})
                    trajectory.append(after)
                    event = detector.update(after["time"], after["position"])
                for other, onset in active.items():
                    lifetimes.append({"collider": other, "found_sample_time_s": onset,
                                      "lost_sample_time_s": None, "duration_s": None,
                                      "right_censored_at_s": trajectory[-1]["time"]})
                record = {"shot": shot, "robot_mode": "absent", "control_status": "absent",
                          "outcome": event.outcome if event else "Timeout",
                          "termination_reason": event.reason if event else "duration_reached",
                          "goal": bool(event and event.outcome == "Goal"), "robot_contact": False,
                          "goal_contact": any(c["kind"] == "goal" for c in contacts),
                          "exit_event": event.to_dict() if event else None,
                          "initial_state": initial, "final_state": trajectory[-1], "trajectory": trajectory,
                          "contacts": contacts, "contact_lifetimes": lifetimes, "raw_contact_events": raw_events,
                          "timestep": dt}
                write_json(args.output_dir / f"{shot['name']}_repeat_{repeat}.json", record)
                hashes.append(digest(record))
                if engine_readback is None:
                    engine_readback = {str(p.GetPath()): {a.GetName(): str(a.Get()) for a in p.GetAttributes()
                                        if a.GetName().startswith(("physics:", "physx"))}
                                       for p in stage.Traverse() if any(a.GetName().startswith(("physics:", "physx")) for a in p.GetAttributes())}
                subscription = None
                step_subscription = None
                close_sim(sim)
                print(f"G1_CASE {shot['name']} repeat={repeat} outcome={record['outcome']} contacts={len(contacts)} final={trajectory[-1]['position']}", flush=True)
            records_summary.append({"case": shot["name"], "repeat_full_record_sha256": hashes,
                                    "repeat_identical": len(set(hashes)) == 1})
        if args.robot_scene:
            stage, sim, ball = open_sim(args.robot_scene)
            roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            robot = SingleArticulation(str(roots[0].GetPath()))
            robot.initialize()
            source = json.loads(args.source_export.read_text())["contract"] if args.source_export else None
            if source is None:
                raise ValueError("--source-export required for source-pose robot smoke")
            trunk = next(body for body in source["bodies"] if body["name"] == "trunk")
            robot.set_world_pose(np.asarray(trunk["position_world_initial_m"]),
                                 np.asarray(trunk["quaternion_world_initial_wxyz"]))
            robot.set_joint_positions(np.zeros(robot.num_dof))
            robot.set_joint_velocities(np.zeros(robot.num_dof))
            robot.set_linear_velocity(np.zeros(3))
            robot.set_angular_velocity(np.zeros(3))
            def robot_state():
                pos, quat = robot.get_world_pose()
                joints = robot.get_joints_state()
                return {"base_position": np.asarray(pos).tolist(), "base_quaternion_wxyz": np.asarray(quat).tolist(),
                        "base_linear_velocity": np.asarray(robot.get_linear_velocity()).tolist(),
                        "base_angular_velocity": np.asarray(robot.get_angular_velocity()).tolist(),
                        "joint_positions": np.asarray(joints.positions).tolist(),
                        "joint_velocities": np.asarray(joints.velocities).tolist(),
                        "dof_names": list(robot.dof_names), "num_dof": robot.num_dof}
            first = robot_state()
            first_step_contacts = []
            def smoke_contact_callback(headers, data):
                for header in headers:
                    points = [data[i] for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data)]
                    first_step_contacts.append({"type": str(header.type),
                        "colliders": [str(PhysicsSchemaTools.intToSdfPath(header.collider0)), str(PhysicsSchemaTools.intToSdfPath(header.collider1))],
                        "separations_m": [float(point.separation) for point in points]})
            smoke_subscription = get_physx_simulation_interface().subscribe_contact_report_events(smoke_contact_callback)
            if first["num_dof"] != 22:
                raise RuntimeError(f"Expected 22 runtime DOFs, got {first['num_dof']}")
            if args.source_export:
                source = json.loads(args.source_export.read_text())["contract"]
                expected_names = [j["name"] for j in source["robot"]["joints"]]
                if set(first["dof_names"]) != set(expected_names):
                    raise RuntimeError("Runtime joint names do not match frozen 22 source joints")
            runtime_robot = {"body_names": list(robot._articulation_view.body_names),
                             "body_masses_kg": np.asarray(robot._articulation_view.get_body_masses()).tolist(),
                             "body_inertias_kg_m2": np.asarray(robot._articulation_view.get_body_inertias()).tolist(),
                             "dof_limits_rad": np.asarray(robot._articulation_view.get_dof_limits()).tolist(),
                             "source": "initialized PhysX tensor articulation view"}
            for step in range(100):
                get_physx_simulation_interface().simulate(dt, step * dt)
                get_physx_simulation_interface().fetch_results()
                if step == 0:
                    smoke_subscription = None
            last = robot_state()
            for state in (first, last):
                for key in ("base_position", "base_quaternion_wxyz", "joint_positions", "joint_velocities"):
                    if not np.isfinite(state[key]).all():
                        raise RuntimeError("Non-finite robot smoke state")
            write_json(args.output_dir / "robot_runtime_smoke.json", {"initial": first, "after_100_steps": last,
                       "runtime_parameters": runtime_robot, "first_step_contacts": first_step_contacts,
                       "initial_contacts_semantics": "reports from first1ms step after exact source state reset, not pre-step contact query", "timestep_s": dt, "free_base_displacement_m": (np.array(last["base_position"])-first["base_position"]).tolist()})
            close_sim(sim)
        write_json(args.output_dir / "protocol.json", {**protocol, "engine": "PhysX / Isaac Sim 4.5.0",
                   "engine_parameters": {"usd_authored_attributes": engine_readback, "runtime_ball_tensor": runtime_ball}, "contact_time_semantics": "PhysX report after each 1ms step; onset labeled pre-step like source; raw contains both clocks"})
        write_json(args.output_dir / "manifest.json", {"engine": "PhysX / Isaac Sim 4.5.0", "cases": records_summary,
                   "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
                   "source_detector_sha256": hashlib.sha256((args.source_root / "k1_goalkeeper/detection.py").read_bytes()).hexdigest(),
                   "source_protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
                   "stepping": "public PhysX simulate(.001,t), fetch_results; validated physics-step event dt and independent free-fall acceleration",
                   "cuda_cuInit_result": cuda_result, "active_gpu": args.gpu, "physics_gpu": args.gpu,
                   "multi_gpu": False, "renderer_used": False, "physx_tensor_device": "cpu"})
        completed = True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        write_json(args.output_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        if not completed:
            import omni.kit.app
            omni.kit.app.get_app().post_quit(1)
        app.close()


if __name__ == "__main__":
    main()
