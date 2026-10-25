from fractions import Fraction
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from acquisition.sdk_exposure import ExposureIntegrator, average_rgba
from acquisition.timing import exposure_sample_times


def test_encoded_rgba_average_does_not_wrap_and_is_not_radiometry():
    frames = np.array([[[[0, 250, 10, 255]]], [[[255, 254, 11, 255]]]], dtype=np.uint8)
    before = frames.copy()
    np.testing.assert_array_equal(average_rgba(frames), [[[128, 252, 10, 255]]])
    np.testing.assert_array_equal(frames, before)
    assert average_rgba(frames).dtype == np.uint8


@pytest.mark.parametrize("frames", [np.zeros((0, 2, 2, 4), dtype=np.uint8),
    np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((2, 2, 2, 3), dtype=np.uint8),
    np.zeros((2, 2, 2, 4), dtype=np.float32)])
def test_average_rejects_invalid_shape_or_type(frames):
    with pytest.raises(ValueError, match="subframes"):
        average_rgba(frames)


def test_exact_exposure_subframe_schedule():
    start = Fraction(1000000001, 120)
    times = exposure_sample_times(start, 5, 120, 3)
    assert times == tuple(start + Fraction(n, 1200) for n in (1, 3, 5))
    assert sum(times) / len(times) == start + Fraction(1, 400)
    assert exposure_sample_times(start, 0, 120, 3) == (start,)


@pytest.mark.parametrize("samples", [0, -1, 1.5, True])
def test_invalid_subframe_count(samples):
    with pytest.raises(ValueError):
        exposure_sample_times(1, 5, 120, samples)


@pytest.fixture
def bridge(monkeypatch):
    class Attribute:
        def __init__(self, value=None):
            self.value = value
            self.disconnected = []

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

        def disconnect(self, destination, modify_usd):
            assert modify_usd is True
            self.disconnected.append(destination)

    class Node:
        def __init__(self):
            self.attributes = {}

        def get_attribute(self, name):
            return self.attributes.setdefault(name, Attribute())

    allocations = []
    synchronized = []

    def array(values, dtype, device, copy):
        assert dtype == "uint8" and device == "cuda:0" and copy is True
        allocation = SimpleNamespace(data=values.copy(), ptr=1000 + len(allocations))
        allocations.append(allocation)
        return allocation

    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(array=array, uint8="uint8", synchronize_device=synchronized.append))
    node = Node()
    node.get_attribute("inputs:width").set(2)
    node.get_attribute("inputs:height").set(1)
    left, right, timestamp = Node(), Node(), Node()
    annotator = SimpleNamespace(zed_=node, left_rgb_annot=SimpleNamespace(get_node=lambda: left),
                               right_rgb_annot=SimpleNamespace(get_node=lambda: right))
    integrator = ExposureIntegrator(annotator, timestamp)
    return integrator, node, left, right, timestamp, allocations, synchronized


def test_publish_retains_private_buffers_and_gates_off_without_closing(bridge):
    integrator, node, left, right, timestamp, allocations, synchronized = bridge
    assert node.get_attribute("inputs:stream").get() is False
    for side, source in (("Left", left), ("Right", right)):
        for field in ("dataPtr", "bufferSize"):
            assert source.get_attribute(f"outputs:{field}").disconnected == [node.get_attribute(f"inputs:{field}{side}")]
    assert timestamp.get_attribute("outputs:result").disconnected == [node.get_attribute("inputs:simulationTime")]
    image = np.arange(8, dtype=np.uint8).reshape(1, 2, 4)
    metadata = integrator.publish(image, image, Fraction(3, 2))
    image[:] = 255
    np.testing.assert_array_equal(allocations[0].data, np.arange(8, dtype=np.uint8))
    integrator.hold()
    integrator.publish(image, image, 2)
    assert len(integrator.retained_buffers) == 2
    assert integrator.retained_buffers[0] == tuple(allocations[:2])
    assert node.get_attribute("inputs:dataPtrLeft").get() == allocations[2].ptr
    assert node.get_attribute("inputs:bufferSizeRight").get() == 8
    assert synchronized == ["cuda:0", "cuda:0"]
    assert metadata["bytes_per_eye"] == 8
    assert len(metadata["source_sha256_left_right"][0]) == 64
    assert node.get_attribute("inputs:stream").get() is True
    integrator.hold()
    assert node.get_attribute("inputs:stream").get() is False


@pytest.mark.parametrize("image", [np.zeros((1, 2, 3), dtype=np.uint8), np.zeros((2, 2, 4), dtype=np.uint8),
                                  np.zeros((1, 2, 4), dtype=np.float32)])
def test_publish_rejects_bad_rgba_before_allocating(bridge, image):
    integrator, _, _, _, _, allocations, _ = bridge
    with pytest.raises(ValueError, match="uint8 RGBA"):
        integrator.publish(image, image, 1)
    assert not allocations


@pytest.mark.parametrize("timestamp", [0, float("nan"), float("inf"), True])
def test_publish_rejects_invalid_timestamp(bridge, timestamp):
    image = np.zeros((1, 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="stamp_s"):
        bridge[0].publish(image, image, timestamp)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:-1", None])
def test_host_device_is_not_accepted_for_bridge(bridge, device):
    with pytest.raises(ValueError, match="explicit CUDA"):
        ExposureIntegrator(None, None, device=device)
