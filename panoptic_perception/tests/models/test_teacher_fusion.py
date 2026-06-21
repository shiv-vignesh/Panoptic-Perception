import os

import pytest
import torch

from panoptic_perception.scripts.train.train_v2 import create_model
from panoptic_perception.models.teacher_model import TeacherFusion
from panoptic_perception.dataset import DataLoaderBuilder


@pytest.fixture
def batch_size():
    return 1


@pytest.fixture
def image_size():
    # (H, W) — matches the cfg used in base_model_kwargs (yolo-768-1280)
    return (640, 640)


@pytest.fixture
def num_classes():
    # BDD100K reduced classes (matches BDD100KClassesReduced)
    return 6


@pytest.fixture
def dummy_image(batch_size, image_size):
    """(B, 3, H, W) float32 in [0, 1] — matches BDDPreprocessor.collate_fn output."""
    h, w = image_size
    return torch.rand(batch_size, 3, h, w, dtype=torch.float32)


@pytest.fixture
def dummy_depth(batch_size, image_size):
    """(B, 1, H, W) float32 in [0, 1] — single-channel depth as fed to depth_backbone."""
    h, w = image_size
    return torch.rand(batch_size, 3, h, w, dtype=torch.float32)


@pytest.fixture
def dummy_detection_targets(batch_size, num_classes):
    """(N_total, 6) float32: [batch_idx, cls, cx, cy, w, h] — post-collate format.

    Three boxes per sample, with batch_idx in column 0. cx/cy in [0,1], w/h in [0.001,1]
    to match collate_fn's clamp range.
    """
    rows = []
    per_sample = 3
    for batch_idx in range(batch_size):
        for _ in range(per_sample):
            cls = float(torch.randint(0, num_classes, (1,)).item())
            cx, cy = torch.rand(2).tolist()
            w, h = (torch.rand(2) * 0.5 + 0.05).tolist()   # bias toward visible boxes
            rows.append([float(batch_idx), cls, cx, cy, w, h])
    return torch.tensor(rows, dtype=torch.float32)


@pytest.fixture
def dummy_drivable_mask(batch_size, image_size):
    """(B, H, W) long with values in {0, 1, 2} — matches drivable_area_seg post-collate."""
    h, w = image_size
    return torch.randint(0, 1, (batch_size, h, w), dtype=torch.long)


@pytest.fixture
def dummy_targets(dummy_detection_targets, dummy_drivable_mask):
    """Mirrors the slice of trainer_refactor's `targets` dict that detection + drivable need."""
    return {
        "detections":         dummy_detection_targets,    # (N, 6)
        "drivable_area_seg":  dummy_drivable_mask,        # (B, H, W) long
        "lane_seg":           None,
        "lanes_detections":   None,
        "lane_seg_masks":     None,
        "clean_images":       None,
    }


@pytest.fixture
def base_model_kwargs():
    return {
        "model_type":"yolop",
        "cfg_path":"panoptic_perception/configs/models/yolo-detection-drivable.cfg",
        "device":"cuda:0"
    }

@pytest.fixture
def loss_function_kwargs():
    return {
        "detection":{
            "_type":"detection-loss-ATSS",
            "kwargs":{}
        },
        "drivable_segmentation":{
            "_type":"segmentation-loss",
            "kwargs":{}
        },
        "lane_detection":{
            "_type":"lane-detection-loss",
            "kwargs":{}
        },
        "loss_weights": {
            "_description": "Multi-task loss weights",
            "detection": 1.0,
            "drivable_segmentation": 1.0,
            "lane_detection": 1.0,
            "lane_segmentation": 0.0
        }        
    }

@pytest.fixture
def teacher_fusion_kwargs(base_model_kwargs):
    return {
        "model_kwargs":base_model_kwargs,
        "fusion_kwargs": {
            "backbone_intercepts": {
                "detection" : [(17, 128), (20, 256), (23, 512)]
            },
            "fusion_type":"attention",
            "weighted_fusion":False,
            "num_fusion_blocks":3
        }
    }

def test_teacher_fusion(teacher_fusion_kwargs, loss_function_kwargs, dummy_image, dummy_depth, dummy_targets):

    image_model, device = create_model(teacher_fusion_kwargs["model_kwargs"], loss_function_kwargs)
    depth_model, device = create_model(teacher_fusion_kwargs["model_kwargs"], loss_function_kwargs)

    assert image_model is not depth_model
    assert id(image_model) != id(depth_model)

    teacher_model = TeacherFusion(
        image_model=image_model,
        depth_model=depth_model,
        fusion_kwargs=teacher_fusion_kwargs["fusion_kwargs"]
    )

    teacher_model.eval()
    with torch.no_grad():
        model_outputs = teacher_model.forward(
            image=dummy_image,
            depth=dummy_depth,
            targets=dummy_targets
        )

    print(model_outputs)


# --------------------------------------------------------------------------- #
# Real-data integration test                                                  #
# --------------------------------------------------------------------------- #

@pytest.fixture
def real_dataset_kwargs():
    """
    Production-shaped dataset config for the privileged-info teacher:
      - dataset_class=foggy → uses FoggyBDDPreprocessor (provides depth_maps)
      - apply_fog=False     → skip fog blend; emit clean image + depth
      - depth_backend=heuristic → fast CPU depth; switch to depth_anything for prod
      - collate_path=cpu    → matches heuristic backend (numpy outputs)
    """
    return {
        "dataset_class": "foggy",
        "apply_fog": False,
        "collate_path": "cpu",
        "adverse_params": {
            "depth_backend": "heuristic",
            "depth_device":  "cpu",
        },
        "images_dir": "../data/100k/100k",
        "detection_annotations_dir": "../data/bdd100k_labels/100k",
        "segmentation_annotations_dir": "../data/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "../data/drivable_maps/labels",
        "train_batch_size": 2,             # smoke test — full prod is 6
        "train_shuffle":    True,
        "train_num_workers": 0,            # foggy path forces this anyway
        "train_preprocessor_kwargs": {
            "image_resize":         [768, 1280],
            "original_image_size":  [720, 1280],
            "perform_augmentation": True,
            "augment_params": {
                "degrees": 5, "translate": 0.05, "scale": 0.15, "shear": 2,
                "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
                "salt_prob": 0.005, "pepper_prob": 0.005,
                "flip_prob": 0.5,
                "img_size": [768, 1280],
            },
            "advanced_aug": {"mosaic_prob": 0.0, "mixup_prob": 0.0, "copy_paste_prob": 0.0},
        },
        "val_batch_size":   2,
        "val_shuffle":      False,
        "val_num_workers":  0,
        "val_preprocessor_kwargs": {
            "image_resize":        [768, 1280],
            "original_image_size": [720, 1280],
        },
    }


@pytest.fixture
def real_teacher_fusion_kwargs(base_model_kwargs):
    """
    Mirrors teacher_fusion_kwargs but with the production cfg the dataloader
    feeds (768x1280). Drops backbone_intercepts to taps known to exist in
    yolo-detection-drivable.cfg with the channel counts listed.
    """
    return {
        "model_kwargs": base_model_kwargs,
        "fusion_kwargs": {
            "backbone_intercepts": {
                "detection": [(17, 128), (20, 256), (23, 512)],
            },
            "fusion_type":       "simple",   # non-square at 768x1280 → AttentionBasedFusion not supported
            "weighted_fusion":   False,
            "num_fusion_blocks": 1,           # smoke test — keep small
        },
    }


def _skip_if_no_data(dataset_kwargs):
    """Skip the test cleanly if the workspace data paths aren't present."""
    for k in ("images_dir", "detection_annotations_dir", "drivable_annotations_dir"):
        path = dataset_kwargs.get(k, "")
        if path and not os.path.isdir(path):
            pytest.skip(f"Required dataset path missing: {k}={path!r}")


def test_teacher_fusion_real_data(real_dataset_kwargs, real_teacher_fusion_kwargs, loss_function_kwargs):
    """
    End-to-end smoke: build a real DataLoader (apply_fog=False), pull one batch,
    feed clean image + depth into TeacherFusion, assert PanopticModelOutputs lands.
    """
    _skip_if_no_data(real_dataset_kwargs)

    builder = DataLoaderBuilder(real_dataset_kwargs, logger=None)
    train_loader = builder.build_train()

    batch = next(iter(train_loader))

    # Sanity: depth-only contract from FoggyBDDPreprocessor
    assert batch["images"]      is not None, "expected clean images in batch"
    assert batch["depth_maps"]  is not None, "expected depth_maps in batch"
    assert batch["fog_mask"].sum().item() == 0, "apply_fog=False should produce all-False fog_mask"
    assert batch["depth_maps"].shape[0] == batch["images"].shape[0], "B mismatch image vs depth"

    # depth_maps shape: (B, H, W). TeacherFusion's depth backbone is a YOLOP copy
    # built from the same cfg (in_channels=3), so replicate single-channel depth
    # across 3 channels. Drop this once the cfg's depth_in_channels=1 is honored
    # at backbone-construction time.
    depth_3ch = batch["depth_maps"].unsqueeze(1).repeat(1, 3, 1, 1).float()

    # Build models
    image_model, _ = create_model(real_teacher_fusion_kwargs["model_kwargs"], loss_function_kwargs)
    depth_model, _ = create_model(real_teacher_fusion_kwargs["model_kwargs"], loss_function_kwargs)

    teacher_model = TeacherFusion(
        image_model=image_model,
        depth_model=depth_model,
        fusion_kwargs=real_teacher_fusion_kwargs["fusion_kwargs"],
    )

    targets = {
        "detections":         batch.get("detections"),
        "drivable_area_seg":  batch.get("drivable_area_seg"),
        "lane_seg":           batch.get("segmentation_masks"),
        "lanes_detections":   batch.get("lanes_detections"),
        "lane_seg_masks":     batch.get("lane_seg_masks"),
        "clean_images":       None,
    }

    teacher_model.eval()
    with torch.no_grad():
        model_outputs = teacher_model.forward(
            image=batch["images"],
            depth=depth_3ch,
            targets=targets,
        )

    assert model_outputs is not None, "teacher_model.forward returned None"
    assert model_outputs.detection_logits is not None, "missing detection_logits in output"
    print(f"detection_logits shapes: {[tuple(t.shape) for t in model_outputs.detection_logits]}")
    print(f"drivable_segmentation_logits: "
          f"{tuple(model_outputs.drivable_segmentation_logits.shape) if model_outputs.drivable_segmentation_logits is not None else None}")
