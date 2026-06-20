import os

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import cProfile
import pstats

from panoptic_perception.dataset.bdd100k_dataset import (
    BDD100KDataset,
    FoggyBDD100KDataset,
    BDDPreprocessor
)

from panoptic_perception.dataset.adverse_weather.depth_estimators import DepthAnythingEstimator

@pytest.fixture
def dataset_type():
    return "train"

@pytest.fixture
def batch_size():
    return 32

@pytest.fixture
def dataset_kwargs():
    return {
        "images_dir": "../data/100k/100k",
        "detection_annotations_dir": "../data/bdd100k_labels/100k",
        "segmentation_annotations_dir": "../data/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "../data/drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        },
        "depth_map_dir":None,
        "adverse_params": {
            "fog_betas": [0.005, 0.010, 0.020],
            "darkness_gammas": [1.3, 1.6, 2.0],
            "enable_fog_only": True,
            "enable_darkness_only": True,
            "enable_compound": True
        }
    }

@pytest.fixture
def depth_anything_backend():

    depth_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return DepthAnythingEstimator(
        model_name="LiheYoung/depth-anything-small-hf",
        device=depth_device, normalization_epsilon=1e-8,
    )    



@pytest.fixture
def foggy_dataset_kwargs():
    return {
        "images_dir": "data/100k/100k",
        "detection_annotations_dir": "data/bdd100k_labels/100k",
        "segmentation_annotations_dir": "data/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "data/drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }

@pytest.fixture
def dataloader(dataset_kwargs, dataset_type, batch_size):
    try:
        dataset = BDD100KDataset(dataset_kwargs, dataset_type=dataset_type)
        print(f"✓ Dataset created successfully")
        print(f"  Dataset type: {dataset.dataset_type}")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Images directory: {dataset.images_dir}")
        print(f"  First few image IDs: {dataset.image_ids[:5]}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging, increase for training
            collate_fn=BDDPreprocessor.collate_fn
        )

        print(f"✓ DataLoader created successfully")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Number of batches: {len(dataloader)}")
        print(f"  Shuffle: {dataloader.shuffle if hasattr(dataloader, 'shuffle') else 'N/A'}")

        return dataloader
        
    except Exception as e:
        print(f"⚠ Dataset creation failed (expected if data not present)")
        print(f"  Error: {e}")
        print(f"  Note: Update paths in test_dataset_creation() to point to actual BDD100K data")
        
        exit(1)

@pytest.fixture
def foggy_dataloader(dataset_kwargs, depth_anything_backend, dataset_type, batch_size):
    try:
        dataset = FoggyBDD100KDataset(dataset_kwargs, 
                                    dataset_type=dataset_type,
                                    perform_augmentation=False,
                                    depth_estimator=depth_anything_backend, 
                                    apply_fog_prob=1.0)

        print(f"✓ Dataset created successfully")
        print(f"  Dataset type: {dataset.dataset_type}")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Images directory: {dataset.images_dir}")
        print(f"  First few image IDs: {dataset.image_ids[:5]}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging, increase for training
            collate_fn=dataset.preprocessor.gpu_collate_fn
        )

        print(f"✓ DataLoader created successfully")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Number of batches: {len(dataloader)}")
        print(f"  Shuffle: {dataloader.shuffle if hasattr(dataloader, 'shuffle') else 'N/A'}")

        return dataloader
    except Exception as e:
        print(f"⚠ Dataset creation failed (expected if data not present)")
        print(f"  Error: {e}")
        print(f"  Note: Update paths in test_dataset_creation() to point to actual BDD100K data")
        exit(1)

@pytest.fixture
def warmup_iters():
    return 5

@pytest.fixture
def c_prof() -> cProfile.Profile:
    return cProfile.Profile()

def test_profile_dataset_dataloader(dataloader, warmup_iters, c_prof):

    def run_dataloader():
        try:
            for _ in dataloader:
                break
        except Exception as e:
            raise RuntimeError(
                f'Runtime Error: {e}'
            )        

    print(f'Running Warmup Iterations')
    
    for _ in range(warmup_iters):
        run_dataloader()
        
    print(f'Warmup Iterations Complete, running profiler')
    
    c_prof.enable()
    run_dataloader()
    c_prof.disable()

    stats = pstats.Stats(c_prof).sort_stats('cumulative')
    stats.print_stats(r'__getitem__|prepare_training_sample|_apply_degradation')
    print()
    stats.print_stats(30)

    c_prof.dump_stats("dataloader.prof")


def test_profile_foggy_dataset_dataloader(foggy_dataloader, warmup_iters, c_prof):

    def run_dataloader():
        try:
            for _ in foggy_dataloader:
                break
        except Exception as e:
            raise RuntimeError(
                f'Runtime Error: {e}'
            )

    print(f'Running Warmup Iterations')

    for _ in range(warmup_iters):
        run_dataloader()

    print(f'Warmup Iterations Complete, running profiler')

    c_prof.enable()
    run_dataloader()
    c_prof.disable()

    stats = pstats.Stats(c_prof).sort_stats('cumulative')
    stats.print_stats(r'__getitem__|prepare_training_sample|_apply_degradation')
    print()
    stats.print_stats(30)

    c_prof.dump_stats("dataloader.prof")


def test_visualize_foggy_batch(foggy_dataloader):
    """
    Pull one batch and write per-sample composites (clean | degraded | depth)
    to viz_foggy_batch/. Useful for eyeballing fog + depth quality.
    """
    out_dir = "viz_foggy_batch"
    os.makedirs(out_dir, exist_ok=True)

    batch = next(iter(foggy_dataloader))
    images   = batch["images"].detach().cpu().numpy()        # (B, 3, H, W) float [0,1]
    clean    = batch["clean_images"].detach().cpu().numpy()  # (B, 3, H, W) float [0,1]
    depths   = batch["depth_maps"].detach().cpu().numpy()    # (B, H, W)    float
    fog_mask = batch["fog_mask"].detach().cpu().numpy()      # (B,) bool

    def _chw_to_bgr(img_chw_float):
        rgb = (img_chw_float.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _depth_to_bgr(depth_hw):
        # Per-sample min-max normalize for visualization contrast.
        # Non-fogged samples are all-zero → render as flat black.
        d_min, d_max = float(depth_hw.min()), float(depth_hw.max())
        if d_max <= d_min:
            d_u8 = np.zeros_like(depth_hw, dtype=np.uint8)
        else:
            d_u8 = (((depth_hw - d_min) / (d_max - d_min)) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)

    B = images.shape[0]
    for i in range(B):
        clean_bgr    = _chw_to_bgr(clean[i])
        degraded_bgr = _chw_to_bgr(images[i])
        depth_bgr    = _depth_to_bgr(depths[i])

        # Align depth resolution to the image if estimator returned a different size
        if depth_bgr.shape[:2] != clean_bgr.shape[:2]:
            depth_bgr = cv2.resize(
                depth_bgr,
                (clean_bgr.shape[1], clean_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        composite = np.concatenate([clean_bgr, degraded_bgr, depth_bgr], axis=1)
        tag = "fogged" if bool(fog_mask[i]) else "clean"
        fname = f"sample{i:02d}_{tag}.png"
        cv2.imwrite(os.path.join(out_dir, fname), composite)

    print(f"Wrote {B} composites (clean | degraded | depth) to {out_dir}/")
