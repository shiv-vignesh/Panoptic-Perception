from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ._io import iter_images, read_rgb, write_rgb, write_npy
from .augmentors import (
    ApplyOrder,
    FogParameters,
    SyntheticFogGenerator,
    SyntheticLowLightGenerator,
    apply_nighttime_fog,
)

from panoptic_perception.dataset.adverse_weather.depth_estimators import (
    DepthAnythingEstimator,
    DepthEstimator,
    HeuristicDepthEstimator,
)

from .config import DEFAULT_CONFIG_PATH, load_config

logger = logging.getLogger(__name__)


class ImagePathDataset(Dataset):
    """Simple dataset that loads images by path for DataLoader prefetching."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image_rgb = read_rgb(path)  # uint8 (H, W, 3)
        return path.stem, image_rgb


def default_fog_betas(config_path: str | Path | None = None) -> list[float]:
    """Fog beta levels from config."""
    return load_config(config_path)["dataset"]["fog_betas"]


def default_darkness_gammas(config_path: str | Path | None = None) -> list[float]:
    """Darkness gamma levels from config."""
    return load_config(config_path)["dataset"]["darkness_gammas"]


@dataclass
class ManifestRecord:
    image_id: str
    clean_path: str
    degraded_path: str
    fog_beta: float
    gamma: float
    condition: str

def build_depth_dataset(
    input_images_dir: str | Path,
    output_dir: str | Path,
    depth_estimator : DepthEstimator,
    config_path: str | Path | None = None,
):

    cfg = load_config(config_path)

    input_dir = Path(input_images_dir)
    out = Path(output_dir)
    
    image_paths = list(iter_images(input_dir, config=cfg))
    if not image_paths:
        raise ValueError(f"No supported images found in {input_dir}")    
    
    _iter = tqdm(image_paths, desc=f'Creating Depth maps')
    for img_idx, image_path in enumerate(_iter, 1):
        image_id = image_path.stem
        image_rgb = read_rgb(image_path)

        depth = depth_estimator.estimate(image_rgb)
        
        depth_arr_path = out / f'{image_path.stem}.npy'
        write_npy(depth_arr_path, depth)


def build_depth_dataset_batch(
    input_images_dir: str | Path,
    output_dir: str | Path,
    depth_estimator: DepthAnythingEstimator,
    config_path: str | Path | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    write_threads: int = 4,
):
    """Batched depth dataset builder with DataLoader prefetching and async writes.

    Skips images whose .npy files already exist in output_dir.
    """
    cfg = load_config(config_path)

    input_dir = Path(input_images_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_images(input_dir, config=cfg))
    if not image_paths:
        raise ValueError(f"No supported images found in {input_dir}")

    # Skip images that already have depth maps
    existing = {p.stem for p in out.glob("*.npy")}
    remaining = [p for p in image_paths if p.stem not in existing]

    if not remaining:
        logger.info("All %d depth maps already exist, nothing to do.", len(image_paths))
        return

    logger.info(
        "Found %d existing depth maps, %d remaining to generate.",
        len(existing), len(remaining),
    )

    dataset = ImagePathDataset(remaining)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    write_pool = ThreadPoolExecutor(max_workers=write_threads)
    futures = []

    for stems, images in tqdm(loader, desc="Creating Depth maps (batched)"):
        # images: (B, H, W, 3) uint8 tensor from default collate
        images_np = [images[i].numpy() for i in range(images.shape[0])]
        depths = depth_estimator.estimate_batch(images_np)

        for stem, depth in zip(stems, depths):
            f = write_pool.submit(write_npy, out / f"{stem}.npy", depth)
            futures.append(f)

    # Wait for all writes to finish
    for f in futures:
        f.result()
    write_pool.shutdown()

    logger.info("Done. Generated %d depth maps in %s", len(remaining), out)


def build_paired_dataset_grid(
    input_images_dir: str | Path,
    output_dir: str | Path,
    fog_generator: SyntheticFogGenerator,
    low_light_generator: SyntheticLowLightGenerator,
    fog_betas: Sequence[float] | None = None,
    darkness_gammas: Sequence[float] | None = None,
    compound_order: ApplyOrder | None = None,
    config_path: str | Path | None = None,
) -> dict:
    """Build paired clean/degraded set with 5x5 grid defaults.

    Depth is estimated once per source image and reused across all
    (beta, gamma) combinations. Parameters not provided are read from config.
    """
    cfg = load_config(config_path)
    ds_cfg = cfg["dataset"]
    fog_cfg = cfg["fog"]
    atm_quantile = fog_cfg["atmospheric_light_quantile"]
    atm_min_pixels = fog_cfg["atmospheric_light_min_pixels"]

    if fog_betas is None:
        fog_betas = ds_cfg["fog_betas"]
    if darkness_gammas is None:
        darkness_gammas = ds_cfg["darkness_gammas"]
    if compound_order is None:
        compound_order = ds_cfg["compound_order"]

    input_dir = Path(input_images_dir)
    out = Path(output_dir)
    clean_dir = out / "clean"
    degraded_root = out / "degraded"
    manifest_dir = out / "manifests"

    image_paths = list(iter_images(input_dir, config=cfg))
    if not image_paths:
        raise ValueError(f"No supported images found in {input_dir}")

    records: list[ManifestRecord] = []
    num_images = len(image_paths)
    max_depth_m = fog_cfg["max_depth_meters"]
    
    _iter = tqdm(image_paths, desc=f'Creating Paired Dataset Grid.')
    for img_idx, image_path in enumerate(_iter, 1):
        image_id = image_path.stem
        image_rgb = read_rgb(image_path)
        clean_path = clean_dir / image_path.name
        write_rgb(clean_path, image_rgb)

        depth = fog_generator.depth_estimator.estimate(image_rgb)

        for i, beta in enumerate(fog_betas):
            for j, gamma in enumerate(darkness_gammas):
                params = FogParameters(
                    beta=beta,
                    max_depth_meters=max_depth_m,
                    atmospheric_light_quantile=atm_quantile,
                    atmospheric_light_min_pixels=atm_min_pixels,
                )
                degraded, _, _ = apply_nighttime_fog(
                    image_rgb=image_rgb,
                    fog_generator=fog_generator,
                    fog_params=params,
                    gamma_generator=low_light_generator,
                    gamma=gamma,
                    apply_order=compound_order,
                    precomputed_depth=depth,
                )
                condition = f"fog_{i + 1}_dark_{j + 1}"
                degraded_path = degraded_root / condition / image_path.name
                write_rgb(degraded_path, degraded)
                rel_clean = str(clean_path.relative_to(out))
                rel_degraded = str(degraded_path.relative_to(out))
                records.append(ManifestRecord(
                    image_id=image_id,
                    clean_path=rel_clean,
                    degraded_path=rel_degraded,
                    fog_beta=beta,
                    gamma=gamma,
                    condition=condition,
                ))

        logger.info("Processed %d/%d: %s", img_idx, num_images, image_id)

    summary = {
        "num_input_images": num_images,
        "num_fog_levels": len(fog_betas),
        "num_darkness_levels": len(darkness_gammas),
        "num_degraded_images": len(records),
        "output_dir": str(out),
        "config_path": str(Path(config_path) if config_path else DEFAULT_CONFIG_PATH),
    }

    manifest_dir.mkdir(parents=True, exist_ok=True)
    dicts = [asdict(r) for r in records]

    with (manifest_dir / "paired_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dicts[0].keys()))
        writer.writeheader()
        writer.writerows(dicts)

    json_indent = cfg["io"]["json_indent"]
    with (manifest_dir / "paired_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=json_indent)

    with (manifest_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=json_indent)

    return summary
