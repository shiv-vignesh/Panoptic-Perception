"""
Smoke test — runs a real Trainer with fake data for 2 epochs on CPU.

Goal: surface AttributeError, TypeError, missing imports, and other
runtime crashes that unit tests with mocks cannot catch.

This test creates:
- A StubTaskModel that mimics BaseTaskModel's forward signature
- Fake DataLoaders matching the BDDPreprocessor.collate_fn output format
- Real Trainer, real Callbacks (CheckpointCallback + EvalMetricsCallback)
- Disabled WandB
"""

import os
import shutil
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from panoptic_perception.trainer.trainer_args import TrainingArgument
from panoptic_perception.trainer.trainer_refactor import Trainer
from panoptic_perception.trainer.callbacks import (
    Callbacks, CheckpointCallback, EvalMetricsCallback,
)
from panoptic_perception.trainer.utils import EvalMetrics
from panoptic_perception.models.models import BaseTaskModel
from panoptic_perception.models.types import PanopticModelOutputs
from panoptic_perception.utils.logger import Logger
from panoptic_perception.utils.wandb_logger import WandBLogger


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

BS = 2
NC = 3        # image channels
IMAGE_H = 32  # small for speed
IMAGE_W = 48
NUM_CLASSES = 3
NUM_DRIVABLE_CLASSES = 2
MAX_DET_PER_IMAGE = 5


# ─────────────────────────────────────────────
# Stub model
# ─────────────────────────────────────────────

class StubTaskModel(BaseTaskModel):
    """
    Minimal model that returns PanopticModelOutputs with losses
    and predictions shaped correctly for the callbacks.
    """

    def __init__(self):
        super().__init__()
        # Need at least one parameter so the optimizer has something
        self.conv = nn.Conv2d(NC, 16, 3, padding=1)
        self.det_head = nn.Linear(16, 6)  # (x, y, w, h, conf, cls)
        self.seg_head = nn.Conv2d(16, NUM_DRIVABLE_CLASSES, 1)

    def forward(self, images, targets=None):
        b, c, h, w = images.shape
        feat = self.conv(images)

        # Detection: produce (B, num_anchors, 6) predictions
        det_flat = feat.mean(dim=[2, 3])  # (B, 16)
        det_pred = self.det_head(det_flat).unsqueeze(1).expand(b, 10, 6)
        det_pred = det_pred.clone()
        # Ensure conf scores are in [0,1] for NMS
        det_pred[:, :, 4] = torch.sigmoid(det_pred[:, :, 4])
        det_pred[:, :, 5] = 0  # class 0

        # Segmentation: produce (B, num_classes, H, W)
        seg_pred = self.seg_head(feat)

        # Compute dummy losses
        det_loss = det_flat.mean() * 0.01  # small, has grad
        seg_loss = seg_pred.mean() * 0.01

        return PanopticModelOutputs(
            detection_loss=det_loss,
            detection_predictions=det_pred,
            drivable_segmentation_loss=seg_loss,
            drivable_segmentation_predictions=seg_pred,
            lane_segmentation_loss=None,
            lane_segmentation_predictions=None,
        )

    def get_active_tasks(self):
        return ["detection", "drivable_segmentation"]

    def get_param_groups(self, optimizer_kwargs=None):
        return [{"params": list(self.parameters()), "name": "stub", "lr_scale": 1.0}]


# ─────────────────────────────────────────────
# Fake dataset mimicking BDDPreprocessor.collate_fn output
# ─────────────────────────────────────────────

class FakeBDDDataset(Dataset):
    """
    Returns per-image items. The collate_fn assembles them into the
    same format that BDDPreprocessor.collate_fn produces.
    """

    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        images = torch.randn(NC, IMAGE_H, IMAGE_W)

        # Per-image detection targets: (n_gt, 5) — [cls, cx, cy, w, h] normalized
        # batch_idx is prepended by collate_fn (same as the real pipeline)
        n_gt = torch.randint(1, MAX_DET_PER_IMAGE + 1, (1,)).item()
        dets = torch.zeros(n_gt, 5)
        dets[:, 0] = torch.randint(0, NUM_CLASSES, (n_gt,)).float()  # class
        dets[:, 1:5] = torch.rand(n_gt, 4).clamp(0.05, 0.95)  # cx, cy, w, h

        drivable_area_seg = torch.randint(0, NUM_DRIVABLE_CLASSES, (IMAGE_H, IMAGE_W))

        return {
            "images": images,
            "detection_targets": dets,
            "drivable_area_seg": drivable_area_seg,
            "segmentation_masks": None,
            "clean_images": None,
            "image_paths": f"fake_image_{idx}.jpg",
        }


def fake_collate_fn(batch):
    """
    Mimics BDDPreprocessor.collate_fn output format.
    detections: (total_targets_in_batch, 6) where col 0 = batch_idx.
    """
    images = torch.stack([item["images"] for item in batch])

    # Build detections the same way the real collate_fn does:
    # prepend batch_idx column, then cat across images
    batch_targets = []
    for batch_idx, item in enumerate(batch):
        dets = item["detection_targets"]  # (n_gt, 5)
        if dets is not None and dets.shape[0] > 0:
            batch_idx_col = torch.full((dets.shape[0], 1), batch_idx, dtype=dets.dtype)
            batch_targets.append(torch.cat([batch_idx_col, dets], dim=1))  # (n_gt, 6)

    detections = torch.cat(batch_targets, dim=0) if batch_targets else None

    drivable_area_seg = torch.stack([item["drivable_area_seg"] for item in batch])

    image_paths = [item["image_paths"] for item in batch]

    return {
        "images": images,
        "detections": detections,
        "drivable_area_seg": drivable_area_seg,
        "segmentation_masks": None,
        "clean_images": None,
        "image_paths": image_paths,
    }


def make_train_dataloader(size=8):
    return DataLoader(
        FakeBDDDataset(size=size),
        batch_size=BS,
        shuffle=True,
        collate_fn=fake_collate_fn,
    )


def make_val_dataloaders(size=4):
    return {
        "val": DataLoader(
            FakeBDDDataset(size=size),
            batch_size=BS,
            shuffle=False,
            collate_fn=fake_collate_fn,
        )
    }


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def output_dir(tmp_path):
    d = str(tmp_path / "smoke_run")
    os.makedirs(d, exist_ok=True)
    yield d
    # cleanup
    if os.path.exists(d):
        shutil.rmtree(d)


@pytest.fixture
def logger(output_dir):
    return Logger(
        log_file_path=os.path.join(output_dir, "test.log"),
        logger_name="smoke_test",
    )


@pytest.fixture
def wandb_logger():
    return WandBLogger(
        project_name="smoke_test",
        run_name="smoke",
        config={},
        enabled=False,
    )


@pytest.fixture
def training_args(output_dir):
    return TrainingArgument(
        output_dir=output_dir,
        epochs=2,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        warmup_epochs=1,
        monitor_train=True,
        monitor_val=True,
        first_val_epoch=0,
        wandb_enabled=False,
        eval_visualize_outputs=False,  # skip file I/O for speed
        checkpoint_idx=1,
        lr_scheduler_start_epoch=0,
    )


@pytest.fixture
def model():
    return StubTaskModel()


# ─────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────

class TestSmokeTrainOnly:
    """Train for 2 epochs without eval — catches training-loop crashes."""

    def test_train_only_no_crash(self, model, training_args, logger, wandb_logger):
        training_args.monitor_val = False

        trainer = Trainer(
            model=model,
            train_dataloader=make_train_dataloader(size=6),
            training_args=training_args,
            logger=logger,
            wandb_logger=wandb_logger,
        )
        trainer.callbacks.add_callback(CheckpointCallback())

        trainer.train()  # should complete without error

        assert trainer.cur_epoch == training_args.epochs


class TestSmokeTrainAndEval:
    """Train + eval for 2 epochs — catches callback interactions."""

    def test_train_and_eval_no_crash(self, model, training_args, logger, wandb_logger, output_dir):
        val_dls = make_val_dataloaders(size=4)

        trainer = Trainer(
            model=model,
            train_dataloader=make_train_dataloader(size=6),
            val_dataloaders=val_dls,
            training_args=training_args,
            logger=logger,
            wandb_logger=wandb_logger,
        )

        trainer.callbacks.add_callback(CheckpointCallback())
        trainer.callbacks.add_callback(EvalMetricsCallback(
            num_drivable_classes=NUM_DRIVABLE_CLASSES,
            visualize_idx=999,  # skip visualization I/O
        ))

        trainer.train()

        assert trainer.cur_epoch == training_args.epochs

        # Checkpoint should have been saved
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        assert os.path.exists(ckpt_dir), "Checkpoint directory should exist"


class TestSmokeEvalOnly:
    """Eval only (monitor_train=False) — catches eval-path crashes."""

    def test_eval_only_no_crash(self, model, training_args, logger, wandb_logger):
        training_args.monitor_train = False
        val_dls = make_val_dataloaders(size=4)

        trainer = Trainer(
            model=model,
            val_dataloaders=val_dls,
            training_args=training_args,
            logger=logger,
            wandb_logger=wandb_logger,
        )

        trainer.callbacks.add_callback(CheckpointCallback())
        trainer.callbacks.add_callback(EvalMetricsCallback(
            num_drivable_classes=NUM_DRIVABLE_CLASSES,
            visualize_idx=999,
        ))

        trainer.train()

        assert trainer.cur_epoch == training_args.epochs


class TestSmokeWarmup:
    """Verify warmup runs without crash during first epoch."""

    def test_warmup_no_crash(self, model, training_args, logger, wandb_logger):
        training_args.warmup_epochs = 2
        training_args.monitor_val = False

        trainer = Trainer(
            model=model,
            train_dataloader=make_train_dataloader(size=6),
            training_args=training_args,
            logger=logger,
            wandb_logger=wandb_logger,
        )

        trainer.train()
        assert trainer.cur_epoch == training_args.epochs


class TestSmokeGradientAccumulation:
    """Verify gradient accumulation doesn't crash."""

    def test_grad_accum_no_crash(self, model, training_args, logger, wandb_logger):
        training_args.gradient_accumulation_steps = 4
        training_args.monitor_val = False

        trainer = Trainer(
            model=model,
            train_dataloader=make_train_dataloader(size=8),
            training_args=training_args,
            logger=logger,
            wandb_logger=wandb_logger,
        )

        trainer.train()
        assert trainer.cur_epoch == training_args.epochs
