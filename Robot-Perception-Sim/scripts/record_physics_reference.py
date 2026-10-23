#!/usr/bin/env python3
"""Record repeatable MuJoCo-only ball references for later PhysX comparison."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from k1_goalkeeper.scene import build_scene, K1_XML, GOAL_URDF
from k1_goalkeeper.simulation import Shot, preset_shots, run_shot


def ground_rebounds(record, upward_threshold=0.05):
    """One downward-to-upward transition per actual ground-contact onset.

    run_shot records onset time at the pre-step clock. Soft contact may take
    several steps to reverse velocity; search until the next ground onset.
    A persistent contact or positive launch velocity alone is not a rebound.
    """
    onsets = [event for event in record["contacts"] if event["geom"] == "ground"]
    trajectory = record["trajectory"]
    events = []
    for index, contact in enumerate(onsets):
        if contact["ball_velocity_before"][2] >= 0:
            continue
        end = onsets[index + 1]["time"] if index + 1 < len(onsets) else float("inf")
        samples = [sample for sample in trajectory if contact["time"] < sample["time"] < end]
        upward = next((sample for sample in samples if sample["velocity"][2] > upward_threshold), None)
        if upward:
            events.append({"ground_contact_onset_s": contact["time"],
                           "upward_sample_time_s": upward["time"],
                           "velocity_before_contact_m_s": contact["ball_velocity_before"],
                           "upward_velocity_m_s": upward["velocity"],
                           "sample_time_resolution_s": record["timestep"]})
    return events


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results/g1" /
                        datetime.now(timezone.utc).strftime("physics_reference_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(PROJECT / "results"):
        parser.error("output-dir must be a new directory under Robot-Perception-Sim/results")
    output.mkdir(parents=True, exist_ok=False)
    # All artifacts, including any native MuJoCo warning log, stay in this new directory.
    import os
    os.chdir(output)
    scene = build_scene(output / "source_scene_absent.xml", robot_mode="absent")
    model = mujoco.MjModel.from_xml_path(str(scene))
    presets = preset_shots(seed=0)
    shots = [Shot("free_fall", (0, 0, 1.5), (0, 0, 0), duration=0.1),
             Shot("ground_bounce", (0, 0, 1.2), (0, 0, 0), duration=2.0),
             presets["center_low"], presets["post"], presets["crossbar"]]
    protocol = {"robot_mode": "absent", "seed": 0, "repeats": 2,
                "cases": [asdict(shot) for shot in shots],
                "case_sources": {"free_fall": "tests/test_ball_physics.py:test_ball_free_fall_before_ground_contact",
                                 "ground_bounce": "tests/test_ball_physics.py:test_ball_bounces_and_does_not_tunnel_through_ground",
                                 "center_low/post/crossbar": "k1_goalkeeper.simulation.preset_shots(seed=0)"},
                "rebound_definition": "actual ground onset with negative pre-contact Vz followed by Vz > 0.05 m/s before next ground onset",
                "contact_time_semantics": "run_shot contact onset uses pre-step time and post-step velocity; not exact impact time",
                "termination": "unchanged run_shot: first whole-ball exit event or duration",
                "ball_radius_m": float(model.geom_size[model.geom('ball_collision').id, 0]),
                "ball_mass_kg": float(model.body_mass[model.body('ball').id]),
                "ball_inertia_kg_m2": model.body_inertia[model.body('ball').id].tolist(),
                "timestep_s": float(model.opt.timestep), "gravity_m_s2": model.opt.gravity.tolist()}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries = []
    for shot in shots:
        records = [run_shot(model, shot, robot_mode="absent") for _ in range(2)]
        hashes = [digest(record) for record in records]
        if hashes[0] != hashes[1]:
            raise RuntimeError(f"Repeated full state/event/trajectory mismatch: {shot.name}")
        for index, record in enumerate(records):
            (output / f"{shot.name}_repeat_{index}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        record = records[0]
        position = np.asarray([sample["position"] for sample in record["trajectory"]])
        velocity = np.asarray([sample["velocity"] for sample in record["trajectory"]])
        summary = {"case": shot.name, "actual_outcome": record["outcome"],
                   "exit_event": record["exit_event"], "contact_onsets": record["contacts"],
                   "ground_rebounds": ground_rebounds(record),
                   "minimum_ball_center_z_m": float(position[:, 2].min()),
                   "actual_duration_s": record["final_state"]["time"],
                   "finite_states": bool(np.isfinite(position).all() and np.isfinite(velocity).all()),
                   "repeat_full_record_sha256": hashes, "repeat_identical": True}
        if shot.name == "free_fall":
            time = record["final_state"]["time"]
            summary["analytic_free_fall_errors"] = {
                "height_m": float(position[-1, 2] - (shot.position[2] - 0.5 * 9.81 * time**2)),
                "vertical_velocity_m_s": float(velocity[-1, 2] + 9.81 * time)}
        summaries.append(summary)
    sources = {scene, output / "protocol.json", Path(__file__).resolve(), K1_XML, GOAL_URDF,
               ROOT / "k1_goalkeeper/simulation.py", ROOT / "k1_goalkeeper/detection.py",
               ROOT / "k1_goalkeeper/scene.py", ROOT / "config/simulation.json",
               ROOT / "tests/test_ball_physics.py", ROOT / "tests/test_goal_collision.py"}
    for mesh in ET.parse(K1_XML).getroot().findall("asset/mesh"):
        sources.add(K1_XML.parent / "meshes" / mesh.get("file"))
    manifest = {"stage": "G1 MuJoCo-only physics reference", "mujoco_version": mujoco.__version__,
                "physx_comparison_complete": False, "usd_migration_complete": False,
                "renderer_used": False, "cases": summaries,
                "source_hashes": [{"path": str(path.relative_to(ROOT)),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                  for path in sorted(sources)]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(output)
    for item in summaries:
        print(f"{item['case']}: actual={item['actual_outcome']}; contacts={len(item['contact_onsets'])}; rebounds={len(item['ground_rebounds'])}; repeat identical")


if __name__ == "__main__":
    main()
