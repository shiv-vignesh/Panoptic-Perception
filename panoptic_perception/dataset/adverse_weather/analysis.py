from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ._io import iter_images, read_rgb
from .config import load_config


@dataclass
class DistributionFeatures:
    mean_luma: float
    std_luma: float
    mean_saturation: float
    mean_contrast_proxy: float


def compute_distribution_features(
    images_dir: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, float]:
    """Compute compact distribution stats for domain-level comparison."""
    cfg = load_config(config_path)
    images_dir = Path(images_dir)
    image_paths = list(iter_images(images_dir, config=cfg))
    if not image_paths:
        raise ValueError(f"No supported images found in {images_dir}")

    ana, io_cfg = cfg["analysis"], cfg["io"]
    uint8_max = float(io_cfg["uint8_max"])
    luma_coeffs = ana["luma_coefficients"]
    sat_max = float(ana["saturation_max"])

    luma_means: list[float] = []
    luma_stds: list[float] = []
    sat_means: list[float] = []
    contrast_proxies: list[float] = []

    for path in image_paths:
        rgb_u8 = read_rgb(path)
        rgb = rgb_u8.astype(np.float32) / uint8_max

        rw, gw, bw = luma_coeffs[0], luma_coeffs[1], luma_coeffs[2]
        luma = rw * rgb[..., 0] + gw * rgb[..., 1] + bw * rgb[..., 2]
        luma_means.append(float(luma.mean()))
        luma_stds.append(float(luma.std()))

        # OpenCV uint8 HSV: S ∈ [0, saturation_max]. Compute from uint8 directly.
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        sat_means.append(float(hsv[..., 1].mean()) / sat_max)

        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        contrast_proxies.append(float(np.sqrt(gx * gx + gy * gy).mean()))

    return asdict(DistributionFeatures(
        mean_luma=float(np.mean(luma_means)),
        std_luma=float(np.mean(luma_stds)),
        mean_saturation=float(np.mean(sat_means)),
        mean_contrast_proxy=float(np.mean(contrast_proxies)),
    ))


def summarize_feature_distributions(
    synthetic_features: dict[str, float],
    real_features: dict[str, float],
    config_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Absolute and relative deltas between synthetic and real stats."""
    cfg = load_config(config_path)
    epsilon = float(cfg["analysis"]["epsilon"])
    out: dict[str, dict[str, float]] = {}
    for k, syn_v in synthetic_features.items():
        real_v = real_features[k]
        delta = syn_v - real_v
        out[k] = {
            "synthetic": syn_v,
            "real": real_v,
            "abs_delta": delta,
            "rel_delta": delta / (abs(real_v) + epsilon),
        }
    return out


def visualize_random_triplets(
    clean_dir: str | Path,
    degraded_dir: str | Path,
    output_path: str | Path,
    num_samples: int | None = None,
    seed: int | None = None,
    config_path: str | Path | None = None,
) -> None:
    """Save clean-vs-degraded comparison grid for visual inspection."""
    cfg = load_config(config_path)
    ana = cfg["analysis"]
    num_samples = num_samples if num_samples is not None else ana["num_samples"]
    seed = seed if seed is not None else ana["seed"]

    clean_dir = Path(clean_dir)
    degraded_dir = Path(degraded_dir)
    output_path = Path(output_path)

    clean_map = {p.name: p for p in iter_images(clean_dir, config=cfg)}
    degraded_map = {p.name: p for p in iter_images(degraded_dir, config=cfg)}
    common = sorted(clean_map.keys() & degraded_map.keys())
    if not common:
        raise ValueError("No common filenames between clean and degraded directories.")

    rng = np.random.default_rng(seed)
    picks = rng.choice(common, size=min(num_samples, len(common)), replace=False)

    fig_w = ana["figsize_width"]
    fig_h = ana["figsize_height_per_sample"] * len(picks)
    fig, axes = plt.subplots(len(picks), 2, figsize=(fig_w, fig_h))
    if len(picks) == 1:
        axes = np.array([axes])

    for row, name in enumerate(picks):
        axes[row, 0].imshow(read_rgb(clean_map[name]))
        axes[row, 0].set_title(f"Clean: {name}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(read_rgb(degraded_map[name]))
        axes[row, 1].set_title(f"Degraded: {name}")
        axes[row, 1].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=ana["dpi"])
    plt.close(fig)
