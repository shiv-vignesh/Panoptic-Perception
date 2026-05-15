"""Mac-friendly smoke test for FoggyBDD100KDataset.

Loads a handful of samples and dumps (foggy, clean, transmission) triplets
to disk so you can eyeball that:
  - fog has a depth-correlated spatial gradient
  - transmission_target is in [0, 1] and dark where pixels are far
  - shapes line up with the augmented image

Run:
    python -m panoptic_perception.scripts.utils.sanity_check_fog_dataset \
        --images-dir /Users/ashu1069/Projects/BDD_100k/100k \
        --det-dir    /Users/ashu1069/Projects/BDD_100k/labels \
        --seg-dir    /Users/ashu1069/Projects/BDD_100k/bdd100k_seg_maps/labels \
        --drivable-dir /Users/ashu1069/Projects/BDD_100k/bdd100k_drivable_maps/labels \
        --out-dir /tmp/fog_sanity \
        --num-samples 8
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

from panoptic_perception.dataset.bdd100k_dataset import (
    FoggyBDD100KDataset, DatasetMode,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True)
    p.add_argument("--det-dir", required=True)
    p.add_argument("--seg-dir", default="")
    p.add_argument("--drivable-dir", default="")
    p.add_argument("--out-dir", default="/tmp/fog_sanity")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--split", default="val")
    return p.parse_args()


def to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.permute(1, 2, 0).cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset_kwargs = {
        "images_dir": args.images_dir,
        "detection_annotations_dir": args.det_dir,
        "segmentation_annotations_dir": args.seg_dir,
        "drivable_annotations_dir": args.drivable_dir,
        "depth_map_dir": "",
        "preprocessor_kwargs": {
            "image_resize": [384, 640],
            "original_image_size": [720, 1280],
        },
        "adverse_params": {
            "fog_betas": [0.015, 0.030],
            "darkness_gammas": [1.3, 1.5],
            "atmospheric_light": [0.9, 0.9, 0.9],
            "enable_fog_only": True,
            "enable_darkness_only": False,
            "enable_compound": False,
            "max_depth_meters": 150.0,
        },
    }

    ds = FoggyBDD100KDataset(
        dataset_kwargs=dataset_kwargs,
        dataset_type=args.split,
        perform_augmentation=False,
        mode=DatasetMode.TRAIN,
        strict_map=False,
        apply_fog_prob=1.0,
    )

    print(f"dataset size: {len(ds)}")

    for i in range(min(args.num_samples, len(ds))):
        sample = ds[i]
        foggy = to_uint8_rgb(sample["image"])
        clean = to_uint8_rgb(sample["clean_image"])
        t_tensor = sample["transmission_target"].squeeze(0).cpu().numpy()
        t_vis = (t_tensor * 255).astype(np.uint8)
        t_vis = cv2.applyColorMap(t_vis, cv2.COLORMAP_VIRIDIS)
        t_vis = cv2.cvtColor(t_vis, cv2.COLOR_BGR2RGB)

        side_by_side = np.concatenate([clean, foggy, t_vis], axis=1)
        out_path = os.path.join(args.out_dir, f"sample_{i:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

        print(
            f"[{i}] fog_applied={sample['fog_applied'].item():.0f} "
            f"beta={sample['beta'].item():.4f} "
            f"t range=({t_tensor.min():.3f}, {t_tensor.max():.3f}) "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
