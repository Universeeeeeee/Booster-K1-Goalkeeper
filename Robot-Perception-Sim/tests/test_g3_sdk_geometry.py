import numpy as np
import pytest

from scripts.verify_g3_sdk_geometry import orange_component


def test_sphere_centroid_uses_connected_pixels_not_projected_coordinate():
    rgb = np.zeros((100, 120, 3), dtype=np.uint8)
    y, x = np.indices(rgb.shape[:2])
    rgb[(x - 63) ** 2 + (y - 51) ** 2 <= 12 ** 2] = [225, 142, 66]
    rgb[32:35, 35:38] = [225, 142, 66]  # Small unrelated orange patch.
    measured, pixels = orange_component(rgb, [60, 50], 12)
    np.testing.assert_allclose(measured, [63, 51])
    assert len(pixels) > 400


def test_missing_sphere_cannot_pass_from_prediction_alone():
    with pytest.raises(ValueError, match="insufficient"):
        orange_component(np.zeros((100, 120, 3), dtype=np.uint8), [60, 50], 12)
