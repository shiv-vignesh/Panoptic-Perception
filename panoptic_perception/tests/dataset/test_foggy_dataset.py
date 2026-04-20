"""
Tests for FoggyBDD100KDataset.

Uses the 50-image dev_set at BDD100k/100k/dev_set/train/ with
synthetic depth maps generated in a temporary directory.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest
import random
import cv2
from collections import Counter
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.bdd100k_dataset import (
    FoggyBDD100KDataset, BDD100KDataset, BDDPreprocessor,
    DatasetMode, visualize_batch,
)
from dataset.adverse_weather import (
    SyntheticFogGenerator, SyntheticLowLightGenerator,
    HeuristicDepthEstimator, FogParameters,
)

# ---------------------------------------------------------------------------
# Paths — adjust if your layout differs
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
BDD_ROOT = PROJECT_ROOT / "BDD100k"

DEV_IMAGES_DIR = str(BDD_ROOT / "100k" / "dev_set")
DETECTION_DIR = str(BDD_ROOT / "bdd100k_labels" / "100k")
DRIVABLE_DIR = str(BDD_ROOT / "bdd100k_drivable_maps" / "labels")

PREPROCESSOR_KWARGS = {
    "image_resize": (768, 1280),
    "original_image_size": (720, 1280),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dev_image_ids():
    """Return sorted list of image IDs in the dev set."""
    train_dir = os.path.join(DEV_IMAGES_DIR, "train")
    assert os.path.isdir(train_dir), f"Dev set not found: {train_dir}"
    ids = sorted([f.split(".")[0] for f in os.listdir(train_dir) if f.endswith(".jpg")])
    assert len(ids) == 50, f"Expected 50 dev images, got {len(ids)}"
    return ids


@pytest.fixture(scope="session")
def depth_cache_dir(dev_image_ids):
    """Create a temp dir with synthetic depth maps (vertical gradient) for every dev image."""
    tmp = tempfile.mkdtemp(prefix="foggy_depth_test_")
    train_dir = os.path.join(tmp, "train")
    os.makedirs(train_dir)

    # Create a plausible depth map for each image: vertical gradient 0 (near/bottom) → 1 (far/top)
    h, w = 720, 1280
    depth = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]
    depth = np.broadcast_to(depth, (h, w)).copy()

    for img_id in dev_image_ids:
        np.save(os.path.join(train_dir, f"{img_id}.npy"), depth)

    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def base_dataset_kwargs(depth_cache_dir):
    return {
        "images_dir": DEV_IMAGES_DIR,
        "detection_annotations_dir": DETECTION_DIR,
        "segmentation_annotations_dir": "",
        "drivable_annotations_dir": DRIVABLE_DIR,
        "preprocessor_kwargs": PREPROCESSOR_KWARGS,
        "depth_map_dir": depth_cache_dir,
        "adverse_params": {
            "fog_betas": [0.005, 0.010, 0.020],
            "darkness_gammas": [1.3, 1.6, 2.0],
            "enable_fog_only": True,
            "enable_darkness_only": True,
            "enable_compound": True,
        },
    }


@pytest.fixture
def foggy_dataset(base_dataset_kwargs):
    """FoggyBDD100KDataset with fog always applied (prob=1.0), no augmentation."""
    return FoggyBDD100KDataset(
        dataset_kwargs=base_dataset_kwargs,
        dataset_type="train",
        perform_augmentation=False,
        mode=DatasetMode.TRAIN,
        strict_map=True,
        apply_fog_prob=1.0,
    )


@pytest.fixture
def foggy_dataset_with_aug(base_dataset_kwargs):
    """FoggyBDD100KDataset with fog always applied, augmentation enabled."""
    return FoggyBDD100KDataset(
        dataset_kwargs=base_dataset_kwargs,
        dataset_type="train",
        perform_augmentation=True,
        mode=DatasetMode.TRAIN,
        strict_map=True,
        apply_fog_prob=1.0,
    )


@pytest.fixture
def clean_only_dataset(base_dataset_kwargs):
    """FoggyBDD100KDataset with fog never applied (prob=0.0)."""
    return FoggyBDD100KDataset(
        dataset_kwargs=base_dataset_kwargs,
        dataset_type="train",
        perform_augmentation=False,
        mode=DatasetMode.TRAIN,
        strict_map=True,
        apply_fog_prob=0.0,
    )


# ===========================================================================
# 1. Construction & validation
# ===========================================================================
class TestConstruction:
    def test_creates_successfully(self, foggy_dataset):
        assert len(foggy_dataset) == 50

    def test_inherits_bdd100k(self, foggy_dataset):
        assert isinstance(foggy_dataset, BDD100KDataset)

    def test_image_depth_id_match(self, foggy_dataset):
        assert set(foggy_dataset.image_ids) == set(foggy_dataset.depth_map_ids)

    def test_strict_map_mismatch_raises(self, base_dataset_kwargs, depth_cache_dir):
        """Remove one depth map and verify strict_map=True raises."""
        train_dir = os.path.join(depth_cache_dir, "train")
        files = sorted(os.listdir(train_dir))
        removed = files[0]
        removed_path = os.path.join(train_dir, removed)
        depth_backup = np.load(removed_path)
        os.remove(removed_path)

        try:
            with pytest.raises(AssertionError, match="Image-depth ID mismatch"):
                FoggyBDD100KDataset(
                    dataset_kwargs=base_dataset_kwargs,
                    dataset_type="train",
                    strict_map=True,
                )
        finally:
            np.save(removed_path, depth_backup)

    def test_non_strict_filters_to_intersection(self, base_dataset_kwargs, depth_cache_dir):
        """With one depth map removed, non-strict should filter to 49 images."""
        train_dir = os.path.join(depth_cache_dir, "train")
        files = sorted(os.listdir(train_dir))
        removed_path = os.path.join(train_dir, files[0])
        depth_backup = np.load(removed_path)
        os.remove(removed_path)

        try:
            ds = FoggyBDD100KDataset(
                dataset_kwargs=base_dataset_kwargs,
                dataset_type="train",
                strict_map=False,
            )
            assert len(ds) == 49
        finally:
            np.save(removed_path, depth_backup)


# ===========================================================================
# 2. Variant builder
# ===========================================================================
class TestBuildVariants:
    def test_all_enabled(self, base_dataset_kwargs):
        ds = FoggyBDD100KDataset(
            dataset_kwargs=base_dataset_kwargs, dataset_type="train",
            strict_map=True, apply_fog_prob=1.0,
        )
        fog_only = [(b, None) for b in ds.fog_betas]
        dark_only = [(None, g) for g in ds.darkness_gammas]
        compound = [(b, g) for b in ds.fog_betas for g in ds.darkness_gammas]
        expected = fog_only + dark_only + compound
        assert ds.variants == expected
        # 3 fog-only + 3 dark-only + 9 compound = 15
        assert len(ds.variants) == 15

    def test_fog_only(self, base_dataset_kwargs):
        kwargs = {**base_dataset_kwargs, "adverse_params": {
            **base_dataset_kwargs["adverse_params"],
            "enable_fog_only": True,
            "enable_darkness_only": False,
            "enable_compound": False,
        }}
        ds = FoggyBDD100KDataset(
            dataset_kwargs=kwargs, dataset_type="train", strict_map=True,
        )
        assert all(g is None for _, g in ds.variants)
        assert len(ds.variants) == 3

    def test_darkness_only(self, base_dataset_kwargs):
        kwargs = {**base_dataset_kwargs, "adverse_params": {
            **base_dataset_kwargs["adverse_params"],
            "enable_fog_only": False,
            "enable_darkness_only": True,
            "enable_compound": False,
        }}
        ds = FoggyBDD100KDataset(
            dataset_kwargs=kwargs, dataset_type="train", strict_map=True,
        )
        assert all(b is None for b, _ in ds.variants)
        assert len(ds.variants) == 3

    def test_compound_only(self, base_dataset_kwargs):
        kwargs = {**base_dataset_kwargs, "adverse_params": {
            **base_dataset_kwargs["adverse_params"],
            "enable_fog_only": False,
            "enable_darkness_only": False,
            "enable_compound": True,
        }}
        ds = FoggyBDD100KDataset(
            dataset_kwargs=kwargs, dataset_type="train", strict_map=True,
        )
        assert all(b is not None and g is not None for b, g in ds.variants)
        assert len(ds.variants) == 9

    def test_none_enabled_raises(self, base_dataset_kwargs):
        kwargs = {**base_dataset_kwargs, "adverse_params": {
            **base_dataset_kwargs["adverse_params"],
            "enable_fog_only": False,
            "enable_darkness_only": False,
            "enable_compound": False,
        }}
        with pytest.raises(AssertionError, match="At least one variant type"):
            FoggyBDD100KDataset(
                dataset_kwargs=kwargs, dataset_type="train", strict_map=True,
            )


# ===========================================================================
# 3. Variant rotation (_next_variant)
# ===========================================================================
class TestVariantRotation:
    def test_exhausts_all_before_repeat(self, foggy_dataset):
        """Each image should see every variant exactly once before any reset."""
        image_id = foggy_dataset.image_ids[0]
        n = len(foggy_dataset.variants)
        seen = set()
        for _ in range(n):
            v = foggy_dataset._next_variant(image_id)
            assert v not in seen, f"Variant {v} repeated before full cycle"
            seen.add(v)
        assert seen == set(foggy_dataset.variants)

    def test_resets_after_full_cycle(self, foggy_dataset):
        image_id = foggy_dataset.image_ids[1]
        n = len(foggy_dataset.variants)
        # Exhaust all
        for _ in range(n):
            foggy_dataset._next_variant(image_id)
        # Next call should reset and still return a valid variant
        v = foggy_dataset._next_variant(image_id)
        assert v in foggy_dataset.variants

    def test_independent_per_image(self, foggy_dataset):
        """Different images track variants independently."""
        id_a = foggy_dataset.image_ids[0]
        id_b = foggy_dataset.image_ids[1]
        foggy_dataset._served.clear()

        v_a = foggy_dataset._next_variant(id_a)
        v_b = foggy_dataset._next_variant(id_b)
        # Both should have served exactly 1
        assert len(foggy_dataset._served[id_a]) == 1
        assert len(foggy_dataset._served[id_b]) == 1


# ===========================================================================
# 4. Scene-aware degradation (_select_degradation)
# ===========================================================================
class TestSelectDegradation:
    def _make_attrs(self, weather, tod):
        return {"weather": weather, "timeofday": tod}

    def test_clear_daytime_passes_through(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("clear", "daytime"), 0.020, 2.0
        )
        assert beta == 0.020
        assert gamma == 2.0

    def test_foggy_nulls_beta(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("foggy", "daytime"), 0.020, 2.0
        )
        assert beta is None
        assert gamma == 2.0

    def test_night_nulls_gamma(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("clear", "night"), 0.020, 2.0
        )
        assert beta == 0.020
        assert gamma is None

    def test_foggy_night_nulls_both(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("foggy", "night"), 0.020, 2.0
        )
        assert beta is None
        assert gamma is None

    def test_dawn_dusk_caps_gamma(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("clear", "dawn/dusk"), 0.020, 2.0
        )
        assert beta == 0.020
        assert gamma == 1.5  # capped

    def test_dawn_dusk_low_gamma_unchanged(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("clear", "dawn/dusk"), 0.020, 1.3
        )
        assert gamma == 1.3  # already below cap

    def test_rainy_caps_beta(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("rainy", "daytime"), 0.020, 2.0
        )
        assert beta == 0.010  # capped
        assert gamma == 2.0

    def test_snowy_caps_beta(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("snowy", "daytime"), 0.020, 2.0
        )
        assert beta == 0.010

    def test_snowy_night(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(
            self._make_attrs("snowy", "night"), 0.020, 2.0
        )
        assert beta == 0.010  # capped
        assert gamma is None  # night

    def test_none_attributes_uses_defaults(self, foggy_dataset):
        beta, gamma = foggy_dataset._select_degradation(None, 0.020, 2.0)
        assert beta == 0.020
        assert gamma == 2.0


# ===========================================================================
# 5. _apply_degradation
# ===========================================================================
class TestApplyDegradation:
    @pytest.fixture
    def sample_image_and_depth(self):
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        depth = np.linspace(1.0, 0.0, 720, dtype=np.float32)[:, None]
        depth = np.broadcast_to(depth, (720, 1280)).copy()
        return img, depth

    def test_fog_only(self, foggy_dataset, sample_image_and_depth):
        img, depth = sample_image_and_depth
        result = foggy_dataset._apply_degradation(img.copy(), depth, beta=0.010, gamma=None)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(result, img)

    def test_darkness_only(self, foggy_dataset, sample_image_and_depth):
        img, depth = sample_image_and_depth
        result = foggy_dataset._apply_degradation(img.copy(), depth, beta=None, gamma=1.6)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        # Darker image should have lower mean
        assert result.mean() < img.mean()

    def test_compound(self, foggy_dataset, sample_image_and_depth):
        img, depth = sample_image_and_depth
        result = foggy_dataset._apply_degradation(img.copy(), depth, beta=0.010, gamma=1.6)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_both_none_returns_unchanged(self, foggy_dataset, sample_image_and_depth):
        img, depth = sample_image_and_depth
        original = img.copy()
        result = foggy_dataset._apply_degradation(img, depth, beta=None, gamma=None)
        np.testing.assert_array_equal(result, original)


# ===========================================================================
# 6. __getitem__ / prepare_training_sample — output format
# ===========================================================================
class TestGetItem:
    def test_output_keys(self, foggy_dataset):
        sample = foggy_dataset[0]
        expected_keys = {"image", "segmentation_mask", "drivable_mask",
                         "detection_targets", "image_path", "scene_attributes"}
        assert set(sample.keys()) == expected_keys

    def test_image_tensor_shape(self, foggy_dataset):
        sample = foggy_dataset[0]
        assert sample["image"].ndim == 3
        assert sample["image"].shape[0] == 3  # C, H, W

    def test_image_normalized_range(self, foggy_dataset):
        sample = foggy_dataset[0]
        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_image_is_float(self, foggy_dataset):
        sample = foggy_dataset[0]
        assert sample["image"].dtype == torch.float32

    def test_detection_targets_shape(self, foggy_dataset):
        sample = foggy_dataset[0]
        targets = sample["detection_targets"]
        assert targets.ndim == 2
        assert targets.shape[1] == 5  # (class, cx, cy, w, h)

    def test_foggy_image_differs_from_clean(self, foggy_dataset, clean_only_dataset):
        """With fog prob=1.0 vs 0.0, the same index should produce different images."""
        random.seed(42)
        np.random.seed(42)
        foggy_sample = foggy_dataset[0]

        random.seed(42)
        np.random.seed(42)
        clean_sample = clean_only_dataset[0]

        # Images should differ (fog was applied to one, not the other)
        assert not torch.allclose(foggy_sample["image"], clean_sample["image"], atol=1e-3)

    def test_targets_identical_foggy_vs_clean(self, foggy_dataset, clean_only_dataset):
        """Fog doesn't change bounding boxes — same targets regardless."""
        foggy_targets = foggy_dataset[0]["detection_targets"]
        clean_targets = clean_only_dataset[0]["detection_targets"]
        torch.testing.assert_close(foggy_targets, clean_targets)


# ===========================================================================
# 7. Augmentation path
# ===========================================================================
class TestAugmentation:
    def test_augmented_output_shape(self, foggy_dataset_with_aug):
        sample = foggy_dataset_with_aug[0]
        H, W = PREPROCESSOR_KWARGS["image_resize"]
        assert sample["image"].shape == (3, H, W)

    def test_non_augmented_output_shape(self, foggy_dataset):
        sample = foggy_dataset[0]
        # Without augmentation, still resized to target
        assert sample["image"].shape[0] == 3

    def test_multiple_samples_no_crash(self, foggy_dataset_with_aug):
        """Iterate 10 samples with augmentation — no errors."""
        for i in range(min(10, len(foggy_dataset_with_aug))):
            sample = foggy_dataset_with_aug[i]
            assert sample["image"].shape[0] == 3


# ===========================================================================
# 8. Collate compatibility
# ===========================================================================
class TestCollateCompat:
    def test_collate_with_foggy_samples(self, foggy_dataset):
        batch = [foggy_dataset[i] for i in range(4)]
        collated = BDDPreprocessor.collate_fn(batch)

        assert collated["images"].shape[0] == 4
        assert collated["images"].ndim == 4  # (B, C, H, W)
        assert collated["detections"] is None or collated["detections"].ndim == 2

    def test_dataloader_iteration(self, foggy_dataset):
        loader = DataLoader(
            foggy_dataset, batch_size=4, shuffle=False,
            num_workers=0, collate_fn=BDDPreprocessor.collate_fn,
        )
        batch = next(iter(loader))
        assert batch["images"].shape[0] == 4
        assert "scene_attributes" in batch


# ===========================================================================
# 9. Fog probability ratio
# ===========================================================================
class TestFogProbability:
    def test_always_foggy(self, base_dataset_kwargs):
        ds = FoggyBDD100KDataset(
            dataset_kwargs=base_dataset_kwargs, dataset_type="train",
            strict_map=True, apply_fog_prob=1.0,
        )
        # All samples should go through the fog branch — verify no crash
        for i in range(min(5, len(ds))):
            ds[i]

    def test_never_foggy(self, base_dataset_kwargs):
        ds = FoggyBDD100KDataset(
            dataset_kwargs=base_dataset_kwargs, dataset_type="train",
            strict_map=True, apply_fog_prob=0.0,
        )
        for i in range(min(5, len(ds))):
            ds[i]


# ===========================================================================
# 10. Visual sanity check (optional, writes to disk)
# ===========================================================================
class TestVisualSanity:
    SAVE_DIR = str(BDD_ROOT / "foggy_dataset_test_outputs")

    def test_save_foggy_vs_clean_comparison(self, foggy_dataset, clean_only_dataset):
        """Save side-by-side foggy and clean samples for manual inspection."""
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        for i in range(min(5, len(foggy_dataset))):
            foggy_sample = foggy_dataset[i]
            clean_sample = clean_only_dataset[i]

            foggy_img = (foggy_sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            clean_img = (clean_sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Stack side by side
            combined = np.concatenate([clean_img, foggy_img], axis=1)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.SAVE_DIR, f"compare_{i}.png"), combined_bgr)

        assert os.path.exists(os.path.join(self.SAVE_DIR, "compare_0.png"))
