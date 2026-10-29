"""G6 endpoint coverage and receiver-ready guards without an SDK/Isaac process."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from acquisition.g6_replay import G6ReplayState, target_midpoints
from acquisition.sdk_state import AcquisitionState


@pytest.mark.parametrize("duration,count", [(.264, 17), (.240, 15)])
def test_shared_midpoints_cover_endpoint_without_exposure_overhang(duration, count):
    centers = target_midpoints(duration, 60)
    assert len(centers) == count
    assert centers[0] == .005
    assert centers[-1] == pytest.approx(duration - .005)
    assert all(a < b for a,b in zip(centers, centers[1:]))
    assert centers[-1] - centers[-2] < 1/60
    for midpoint in centers:
        assert midpoint - .005 >= -1e-12
        assert midpoint + .005 <= duration + 1e-12
        assert ((midpoint - .005) + (midpoint + .005))/2 == pytest.approx(midpoint)


def test_too_short_frozen_trace_rejected():
    with pytest.raises(ValueError, match="complete 10 ms"):
        target_midpoints(.009, 60)


@pytest.fixture
def replay(tmp_path, monkeypatch):
    replay = G6ReplayState.__new__(G6ReplayState)
    replay.args = SimpleNamespace(g6_receiver_ready=tmp_path / "metadata.json")
    replay.receiver_ready_evidence = None
    monkeypatch.setattr(AcquisitionState, "publish_prepared", lambda self,state,*_: state)
    return replay


def test_target_refuses_missing_receiver(replay):
    with pytest.raises(RuntimeError, match="did not open"):
        replay.publish_prepared({"phase": "target", "target_index": 0}, None, None)


@pytest.mark.parametrize("grabs,mode,error", [(2,"NEURAL","at least3"), (3,"PERFORMANCE","actual SDK NEURAL")])
def test_target_refuses_incomplete_or_wrong_mode_receiver(replay, grabs, mode, error):
    ready = replay.args.g6_receiver_ready
    ready.write_text(json.dumps({"depth_mode": mode}))
    (ready.parent / "frames.jsonl").write_text("".join(json.dumps({"frame_index": n})+"\n" for n in range(grabs)))
    with pytest.raises(RuntimeError, match=error):
        replay.publish_prepared({"phase": "target", "target_index": 0}, None, None)


def test_target_records_receiver_readback_and_preroll_grabs(replay):
    ready = replay.args.g6_receiver_ready
    ready.write_text(json.dumps({"depth_mode": "NEURAL"}))
    (ready.parent / "frames.jsonl").write_text("".join(json.dumps({"frame_index": n})+"\n" for n in range(3)))
    state = replay.publish_prepared({"phase": "target", "target_index": 0}, None, None)
    assert state["receiver_ready_evidence"]["preroll_grabs_before_target"] == 3
    assert len(state["receiver_ready_evidence"]["receiver_metadata_sha256"]) == 64
