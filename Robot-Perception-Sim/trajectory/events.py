"""Adapter over the original whole-ball GoalDetector and native contact evidence."""


def contact_kind(geom, source_kind=None):
    if geom == "ground":
        return "ground_contact"
    if geom.startswith("goal_") and geom.endswith("_collision"):
        return "frame_contact"
    if source_kind == "robot" or geom.startswith("k1_geom_"):
        return "robot_contact"
    return "other_contact"


def classify_record(record, detector_class):
    detector = detector_class(radius=record.get("ball_radius_m", .11))
    event = None
    for sample in record["trajectory"]:
        event = detector.update(sample["time"], sample["position"])
    events = []
    for contact in record.get("contacts", []):
        events.append({"type": "contact_begin", "category": contact_kind(contact["geom"], contact["kind"]),
                       "time_s": contact["time"], "object": contact["geom"],
                       "contact_position_world_m": contact.get("position"),
                       "ball_velocity_before_m_s": contact["ball_velocity_before"],
                       "ball_velocity_after_m_s": contact["ball_velocity_after"],
                       "onset_observation": contact.get("onset_observation"),
                       "time_semantics": "pre-step lower bound; raw reports retain both bounds"})
    states = record["trajectory"]
    by_time = {s["time"]: i for i, s in enumerate(states)}
    for raw in record.get("raw_contact_events", []):
        if raw["type"] != "CONTACT_LOST":
            continue
        index = by_time[raw["time_after_step"]]
        events.append({"type": "contact_end", "time_s": raw["time_after_step"],
                       "object": raw["other_collider"],
                       "contact_position_world_m": raw["points"][0]["position"] if raw["points"] else None,
                       "ball_velocity_before_m_s": states[index-1]["velocity"],
                       "ball_velocity_after_m_s": states[index]["velocity"]})
    # A native contact onset need not be a bounce (e.g. persistent resting contact).
    # Count measured vertical reversal near the floor, independently of onset count.
    ground = [c for c in record.get("contacts", []) if contact_kind(c["geom"], c["kind"]) == "ground_contact"]
    ground_report_times = {r["time_after_step"] for r in record.get("raw_contact_events", [])
                           if r["other_collider"].endswith("_ground") and r["type"] != "CONTACT_LOST"}
    for before, after in zip(states, states[1:]):
        if (ground and after["time"] in ground_report_times and before["position"][2] <= .14 and before["velocity"][2] < -.05
                and after["velocity"][2] > .05):
            events.append({"type": "ground_bounce", "time_s": after["time"], "object": "ground",
                           "ball_position_world_m": after["position"],
                           "ball_velocity_before_m_s": before["velocity"],
                           "ball_velocity_after_m_s": after["velocity"]})
    if event:
        value = event.to_dict()
        events.append({"type": "whole_ball_exit", "time_s": value.pop("time"), **value})
    else:
        events.append({"type": "timeout", "time_s": states[-1]["time"], "reason": "duration_reached"})
    events.sort(key=lambda e: e["time_s"])
    ground_count = sum(e.get("category") == "ground_contact" for e in events)
    frame = any(e.get("category") == "frame_contact" for e in events)
    robot = any(e.get("category") == "robot_contact" for e in events)
    outcome = event.outcome if event else "Timeout"
    return {"source": record.get("source", "physx_physical_shot"),
            "outcome": outcome, "termination_reason": event.reason if event else "duration_reached",
            "robot_contact": robot, "robot_contact_is_save": None,
            "ground_contact_count": ground_count,
            "ground_bounce_count": sum(e["type"] == "ground_bounce" for e in events), "frame_contact": frame,
            "classification": ("robot_then_goal" if robot else "frame_then_goal" if frame else "ground_then_goal" if ground_count else "direct_goal")
                              if outcome == "Goal" else "robot_then_non_goal" if robot else "frame_then_non_goal" if frame else outcome.lower(),
            "visibility": "unobserved_no_camera_evidence",
            "events": events, "label_basis": "original GoalDetector + native contact identities + measured floor velocity reversal"}
