from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module import (
    FogParameters,
    SyntheticFogGenerator,
    SyntheticLowLightGenerator,
    apply_nighttime_fog,
    build_paired_dataset_grid,
    compute_distribution_features,
    default_darkness_gammas,
    default_fog_betas,
    summarize_feature_distributions,
    visualize_random_triplets,
)
from panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module._io import (
    write_rgb,
)
from panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.config import (
    load_config,
)
from panoptic_perception.scripts.experiments.synthetic_adverse_augmentation_module.depth_estimators import (
    HeuristicDepthEstimator,
)


# ─── Fixtures / helpers ──────────────────────────────────────────────


class DummyDepthEstimator:
    """Deterministic linear-gradient depth for tests."""

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        y = np.linspace(0.0, 1.0, num=h, dtype=np.float32)[:, None]
        return np.tile(y, (1, w))


def _sample_image(h: int = 64, w: int = 96) -> np.ndarray:
    """Synthetic gradient image (uint8 RGB)."""
    x = np.linspace(0, 255, num=w, dtype=np.uint8)
    y = np.linspace(0, 255, num=h, dtype=np.uint8)[:, None]
    r = np.tile(x, (h, 1))
    g = np.tile(y, (1, w))
    b = ((r.astype(np.int16) + g.astype(np.int16)) // 2).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


# ─── HeuristicDepthEstimator ─────────────────────────────────────────


def _heuristic_estimator() -> HeuristicDepthEstimator:
    cfg = load_config()
    h, io_cfg = cfg["heuristic_depth"], cfg["io"]
    return HeuristicDepthEstimator(
        vertical_weight=h["vertical_weight"],
        intensity_weight=h["intensity_weight"],
        edge_weight=h["edge_weight"],
        edge_epsilon=h["edge_epsilon"],
        uint8_max=float(io_cfg["uint8_max"]),
    )


def _low_light_generator() -> SyntheticLowLightGenerator:
    cfg = load_config()
    ll = cfg["low_light"]
    return SyntheticLowLightGenerator(
        gamma_min=ll["gamma_min"],
        gamma_max=ll["gamma_max"],
        gamma_min_threshold=ll["gamma_min_threshold"],
        config=cfg,
    )


def _fog_params(beta: float, max_depth_meters: float | None = None) -> FogParameters:
    cfg = load_config()
    fog = cfg["fog"]
    return FogParameters(
        beta=beta,
        max_depth_meters=max_depth_meters or fog["max_depth_meters"],
        atmospheric_light_quantile=fog["atmospheric_light_quantile"],
        atmospheric_light_min_pixels=fog["atmospheric_light_min_pixels"],
    )


class TestHeuristicDepthEstimator:
    def test_output_shape_and_range(self):
        estimator = _heuristic_estimator()
        image = _sample_image(128, 192)
        depth = estimator.estimate(image)

        assert depth.shape == (128, 192)
        assert depth.dtype == np.float32
        assert 0.0 <= float(depth.min())
        assert float(depth.max()) <= 1.0

    def test_top_row_farther_than_bottom(self):
        estimator = _heuristic_estimator()
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        depth = estimator.estimate(image)

        assert depth[0, :].mean() > depth[-1, :].mean()

    def test_all_black_image(self):
        estimator = _heuristic_estimator()
        depth = estimator.estimate(np.zeros((32, 32, 3), dtype=np.uint8))

        assert depth.shape == (32, 32)
        assert np.isfinite(depth).all()

    def test_all_white_image(self):
        estimator = _heuristic_estimator()
        depth = estimator.estimate(np.full((32, 32, 3), 255, dtype=np.uint8))

        assert depth.shape == (32, 32)
        assert np.isfinite(depth).all()


# ─── SyntheticLowLightGenerator ──────────────────────────────────────


class TestLowLight:
    def test_gamma_darkens_image(self):
        image = _sample_image()
        gen = _low_light_generator()
        out = gen.apply(image, gamma=2.2)

        assert out.shape == image.shape
        assert out.dtype == np.uint8
        assert out.mean() < image.mean()

    def test_higher_gamma_darker(self):
        image = _sample_image()
        gen = _low_light_generator()
        out_low = gen.apply(image, gamma=1.5)
        out_high = gen.apply(image, gamma=3.0)

        assert out_high.mean() < out_low.mean()

    def test_boundary_gammas_accepted(self):
        image = _sample_image()
        gen = _low_light_generator()

        gen.apply(image, gamma=1.5)
        gen.apply(image, gamma=3.0)

    def test_gamma_outside_range_raises(self):
        gen = _low_light_generator()
        image = _sample_image()

        with pytest.raises(ValueError, match="outside"):
            gen.apply(image, gamma=1.0)

        with pytest.raises(ValueError, match="outside"):
            gen.apply(image, gamma=4.0)

    def test_invalid_constructor_ranges(self):
        cfg = load_config()
        ll = cfg["low_light"]
        with pytest.raises(ValueError, match="gamma_min must be >"):
            SyntheticLowLightGenerator(
                gamma_min=0.5,
                gamma_max=3.0,
                gamma_min_threshold=ll["gamma_min_threshold"],
                config=cfg,
            )
        with pytest.raises(ValueError, match="gamma_max must be >= gamma_min"):
            SyntheticLowLightGenerator(
                gamma_min=2.0,
                gamma_max=1.5,
                gamma_min_threshold=ll["gamma_min_threshold"],
                config=cfg,
            )


# ─── SyntheticFogGenerator ───────────────────────────────────────────


class TestFog:
    def test_output_shapes_and_dtypes(self):
        image = _sample_image()
        fog = SyntheticFogGenerator(depth_estimator=DummyDepthEstimator())
        out, depth, transmission = fog.generate(image, _fog_params(0.03))

        assert out.shape == image.shape
        assert out.dtype == np.uint8
        assert depth.shape == image.shape[:2]
        assert depth.dtype == np.float32
        assert transmission.shape == image.shape[:2]
        assert 0.0 <= float(depth.min()) <= float(depth.max()) <= 1.0
        assert 0.0 <= float(transmission.min()) <= float(transmission.max()) <= 1.0

    def test_heavier_fog_reduces_contrast(self):
        image = _sample_image()
        fog = SyntheticFogGenerator(depth_estimator=DummyDepthEstimator())
        out_light, _, _ = fog.generate(image, _fog_params(0.008))
        out_heavy, _, _ = fog.generate(image, _fog_params(0.060))

        assert float(out_heavy.std()) < float(out_light.std())

    def test_custom_atmospheric_light(self):
        image = _sample_image()
        fog = SyntheticFogGenerator(depth_estimator=DummyDepthEstimator())
        A = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        params = _fog_params(0.04)
        params = FogParameters(
            beta=params.beta,
            max_depth_meters=params.max_depth_meters,
            atmospheric_light_quantile=params.atmospheric_light_quantile,
            atmospheric_light_min_pixels=params.atmospheric_light_min_pixels,
            atmospheric_light=A,
        )
        out, _, _ = fog.generate(image, params)

        assert out.shape == image.shape
        assert out.dtype == np.uint8

    def test_precomputed_depth_reused(self):
        image = _sample_image()
        fog = SyntheticFogGenerator(depth_estimator=DummyDepthEstimator())
        _, depth, _ = fog.generate(image, _fog_params(0.03))

        out2, depth2, _ = fog.generate(
            image, _fog_params(0.05), precomputed_depth=depth
        )
        assert out2.dtype == np.uint8
        np.testing.assert_array_equal(depth, depth2)


# ─── Compound nighttime fog ──────────────────────────────────────────


class TestCompound:
    def test_both_orders_valid(self):
        image = _sample_image()
        fog = SyntheticFogGenerator(depth_estimator=DummyDepthEstimator())
        low = _low_light_generator()

        out_a, _, _ = apply_nighttime_fog(
            image, fog, _fog_params(0.04), low, gamma=2.0,
            apply_order="dark_then_fog",
        )
        out_b, _, _ = apply_nighttime_fog(
            image, fog, _fog_params(0.04), low, gamma=2.0,
            apply_order="fog_then_dark",
        )
        assert out_a.shape == image.shape
        assert out_b.shape == image.shape
        assert out_a.dtype == np.uint8
        assert out_b.dtype == np.uint8

    def test_invalid_order_raises(self):
        image = _sample_image()
        fog = SyntheticFogGenerator(depth_estimator=DummyDepthEstimator())
        low = _low_light_generator()

        with pytest.raises(ValueError, match="apply_order"):
            apply_nighttime_fog(
                image, fog, _fog_params(0.04), low, gamma=2.0,
                apply_order="invalid",
            )


# ─── Paired dataset grid ─────────────────────────────────────────────


class TestDatasetGrid:
    def test_5x5_grid_output(self, tmp_path: Path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        for idx in range(3):
            write_rgb(input_dir / f"img_{idx}.png", _sample_image())

        fog = SyntheticFogGenerator(depth_estimator=DummyDepthEstimator())
        low = _low_light_generator()
        summary = build_paired_dataset_grid(
            input_images_dir=input_dir,
            output_dir=output_dir,
            fog_generator=fog,
            low_light_generator=low,
            fog_betas=default_fog_betas(),
            darkness_gammas=default_darkness_gammas(),
        )

        assert summary["num_input_images"] == 3
        assert summary["num_fog_levels"] == 5
        assert summary["num_darkness_levels"] == 5
        assert summary["num_degraded_images"] == 75

        clean_images = list((output_dir / "clean").glob("*.png"))
        assert len(clean_images) == 3

        manifest_csv = output_dir / "manifests" / "paired_manifest.csv"
        manifest_json = output_dir / "manifests" / "paired_manifest.json"
        summary_json = output_dir / "manifests" / "summary.json"
        assert manifest_csv.exists()
        assert manifest_json.exists()
        assert summary_json.exists()

        data = json.loads(manifest_json.read_text(encoding="utf-8"))
        assert len(data) == 75

        record = data[0]
        assert "image_id" in record
        assert "fog_beta" in record
        assert "gamma" in record


# ─── Analysis hooks ──────────────────────────────────────────────────


class TestAnalysis:
    def test_distribution_features(self, tmp_path: Path):
        synthetic_dir = tmp_path / "synthetic"
        real_dir = tmp_path / "real"
        synthetic_dir.mkdir()
        real_dir.mkdir()

        base = _sample_image()
        darker = np.clip(
            (base.astype(np.float32) / 255.0) ** 2.2 * 255.0, 0, 255
        ).astype(np.uint8)

        for i in range(4):
            write_rgb(synthetic_dir / f"s_{i}.png", darker)
            write_rgb(real_dir / f"r_{i}.png", base)

        syn_feat = compute_distribution_features(synthetic_dir)
        real_feat = compute_distribution_features(real_dir)
        report = summarize_feature_distributions(syn_feat, real_feat)

        assert "mean_luma" in report
        assert "mean_saturation" in report
        assert report["mean_luma"]["synthetic"] < report["mean_luma"]["real"]

    def test_visualize_creates_panel(self, tmp_path: Path):
        clean_dir = tmp_path / "clean"
        degraded_dir = tmp_path / "degraded"
        output_path = tmp_path / "panels" / "triplets.png"
        clean_dir.mkdir()
        degraded_dir.mkdir()

        for i in range(3):
            clean = _sample_image()
            degraded = np.clip(
                clean.astype(np.float32) * 0.6 + 20, 0, 255
            ).astype(np.uint8)
            write_rgb(clean_dir / f"pair_{i}.png", clean)
            write_rgb(degraded_dir / f"pair_{i}.png", degraded)

        visualize_random_triplets(
            clean_dir, degraded_dir, output_path, num_samples=2, seed=1
        )
        assert output_path.exists()
