import numpy as np
import pytest

from panoptic_perception.dataset.augmentations import apply_augmentations
from panoptic_perception.dataset.bdd100k_dataset import (
    FoggyBDD100KDataset,
    FoggyBDDPreprocessor,
)
from panoptic_perception.dataset.types import DatasetMode, FrameData


def _preprocessor(**adverse_params):
    params = {
        "fog_mode": "homogeneous",
        "homogeneous_depth": 0.723,
        "max_depth_meters": 80.0,
        "atmospheric_light": [0.85, 0.85, 0.85],
        "apply_darkness": False,
        **adverse_params,
    }
    return FoggyBDDPreprocessor({}, adverse_params=params)


def test_homogeneous_fog_uses_constant_transmission():
    preprocessor = _preprocessor()
    image = np.array(
        [
            [[20, 40, 60], [80, 100, 120]],
            [[140, 160, 180], [200, 220, 240]],
        ],
        dtype=np.uint8,
    )
    depth = np.full(image.shape[:2], 0.723, dtype=np.float32)
    beta = 0.01

    foggy = preprocessor._apply_degradation(image, depth, beta=beta, gamma=None)

    transmission = np.exp(-beta * 0.723 * 80.0)
    expected = (
        (image.astype(np.float32) / 255.0) * transmission
        + 0.85 * (1.0 - transmission)
    )
    expected = np.clip(expected * 255.0, 0.0, 255.0).astype(np.uint8)

    np.testing.assert_array_equal(foggy, expected)


def test_homogeneous_fog_preserves_clean_image():
    preprocessor = _preprocessor()
    dataset = FoggyBDD100KDataset.__new__(FoggyBDD100KDataset)
    dataset.mode = DatasetMode.EVAL
    dataset.preprocessor = preprocessor
    original = np.full((8, 12, 3), 40, dtype=np.uint8)
    frame = FrameData(image=original.copy(), image_path="sample.jpg")

    foggy_frame, depth, fog_applied = dataset._maybe_apply_homogeneous_fog(frame)

    assert fog_applied
    np.testing.assert_array_equal(foggy_frame.clean_image, original)
    assert not np.array_equal(foggy_frame.image, original)
    np.testing.assert_array_equal(depth, np.full((8, 12), 0.723, dtype=np.float32))


def test_pair_and_depth_share_geometric_augmentation():
    clean = np.zeros((40, 80, 3), dtype=np.uint8)
    foggy = np.zeros_like(clean)
    depth = np.zeros((40, 80), dtype=np.float32)
    clean[10:20, 15:25] = 250
    foggy[10:20, 15:25] = 200
    depth[10:20, 15:25] = 1.0
    frame = FrameData(
        image=foggy,
        clean_image=clean,
        depth_map=depth,
        image_path="sample.jpg",
    )

    augmented = apply_augmentations(
        frame,
        {
            "degrees": 0,
            "translate": 0,
            "scale": 0,
            "shear": 0,
            "hsv_h": 0,
            "hsv_s": 0,
            "hsv_v": 0,
            "salt_prob": 0,
            "pepper_prob": 0,
            "flip_prob": 1,
        },
        img_size=(64, 64),
    )

    clean_peak = np.unravel_index(np.argmax(augmented.clean_image[..., 0]), (64, 64))
    foggy_peak = np.unravel_index(np.argmax(augmented.image[..., 0]), (64, 64))
    depth_peak = np.unravel_index(np.argmax(augmented.depth_map), (64, 64))
    assert clean_peak == foggy_peak == depth_peak


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"fog_mode": "unknown"}, "Unsupported fog_mode"),
        ({"homogeneous_depth": 1.1}, "homogeneous_depth must be in"),
    ],
)
def test_homogeneous_fog_rejects_invalid_config(params, message):
    with pytest.raises(ValueError, match=message):
        _preprocessor(**params)
