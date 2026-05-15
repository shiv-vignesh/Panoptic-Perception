"""Precompute Depth Anything maps for a BDD100K-style image tree.

Usage:
    python -m panoptic_perception.scripts.utils.precompute_depth \
        --images-dir /path/to/bdd100k/images \
        --depth-dir  /path/to/bdd100k/depth \
        --splits train val \
        --model-name LiheYoung/depth-anything-small-hf \
        --batch-size 8 \
        --device cuda

Layout produced (mirrors images dir):
    <depth-dir>/<split>/<image_id>.npy   # float32, shape (H, W), values in [0, 1]

Values follow the existing DepthAnythingEstimator convention:
    1.0 = closest, 0.0 = farthest   (inverted normalized depth)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from panoptic_perception.dataset.adverse_weather import DepthAnythingEstimator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, required=True,
                   help="Root with <split>/*.jpg")
    p.add_argument("--depth-dir", type=Path, required=True,
                   help="Output root; will create <split>/*.npy")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--model-name", default="LiheYoung/depth-anything-small-hf",
                   help="HF model id. Use depth-anything-base/large for higher quality.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--normalization-epsilon", type=float, default=1e-6)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--overwrite", dest="skip_existing", action="store_false")
    return p.parse_args()


def iter_batches(items, batch_size):
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_split(estimator: DepthAnythingEstimator,
                  images_dir: Path,
                  depth_dir: Path,
                  split: str,
                  batch_size: int,
                  skip_existing: bool) -> None:
    src = images_dir / split
    dst = depth_dir / split
    if not src.is_dir():
        print(f"[skip] {src} not found")
        return
    dst.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in src.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if skip_existing:
        image_paths = [p for p in image_paths
                       if not (dst / f"{p.stem}.npy").exists()]

    print(f"[{split}] {len(image_paths)} images to process -> {dst}")
    pbar = tqdm(total=len(image_paths), desc=split)

    for batch_paths in iter_batches(image_paths, batch_size):
        imgs = []
        valid_paths = []
        for p in batch_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                pbar.update(1)
                continue
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            valid_paths.append(p)

        if not imgs:
            continue

        depths = estimator.estimate_batch(imgs)
        for p, d in zip(valid_paths, depths):
            out = dst / f"{p.stem}.npy"
            np.save(out, d.astype(np.float32))
        pbar.update(len(batch_paths))
    pbar.close()


def main() -> None:
    args = parse_args()
    estimator = DepthAnythingEstimator(
        model_name=args.model_name,
        device=args.device,
        normalization_epsilon=args.normalization_epsilon,
    )
    for split in args.splits:
        process_split(estimator, args.images_dir, args.depth_dir,
                      split, args.batch_size, args.skip_existing)


if __name__ == "__main__":
    main()
