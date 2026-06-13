"""
Unit tests for Trainer.__init__ — validation, attribute setup, optimizer/scheduler creation.
Uses a lightweight stub model to avoid loading real weights.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

from panoptic_perception.trainer.trainer_args import TrainingArgument
from panoptic_perception.trainer.trainer_refactor import Trainer
from panoptic_perception.trainer.utils import EvalMetrics
from panoptic_perception.models.models import BaseTaskModel
from panoptic_perception.models.types import PanopticModelOutputs
from panoptic_perception.utils.logger import Logger
from panoptic_perception.utils.wandb_logger import WandBLogger


# ─────────────────────────────────────────────
# Stub model for testing (no real weights/config)
# ─────────────────────────────────────────────

class StubTaskModel(BaseTaskModel):
    """Minimal BaseTaskModel subclass for trainer tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, images, targets=None):
        b = images.shape[0]
        return PanopticModelOutputs(
            detection_loss=torch.tensor(1.0, requires_grad=True),
            detection_predictions=torch.randn(b, 10, 6),
            drivable_segmentation_loss=torch.tensor(0.5, requires_grad=True),
            drivable_segmentation_predictions=torch.randn(b, 2, 8, 8),
        )

    def get_active_tasks(self):
        return ["detection", "drivable_segmentation"]

    def get_param_groups(self, optimizer_kwargs=None):
        return [{"params": list(self.parameters()), "name": "stub", "lr_scale": 1.0}]


def _make_logger(tmp_path):
    log_path = str(tmp_path / "test.log")
    return Logger(log_file_path=log_path, logger_name="test_trainer_init")


def _make_wandb_logger():
    return WandBLogger(
        project_name="test", run_name="test",
        config={}, enabled=False
    )


def _make_val_dataloaders():
    ds = TensorDataset(torch.randn(4, 3, 8, 8), torch.randn(4, 6))
    return {"val": DataLoader(ds, batch_size=2)}


# ─────────────────────────────────────────────
# Validation errors
# ─────────────────────────────────────────────

class TestTrainerValidation:

    def test_model_none_raises(self, tmp_path):
        with pytest.raises(ValueError, match="requires a model"):
            Trainer(
                model=None,
                logger=_make_logger(tmp_path),
                wandb_logger=_make_wandb_logger(),
            )

    def test_logger_none_raises(self):
        model = StubTaskModel()
        with pytest.raises(ValueError, match="requires a logger"):
            Trainer(model=model, logger=None, wandb_logger=_make_wandb_logger())

    def test_wandb_logger_none_raises(self, tmp_path):
        model = StubTaskModel()
        with pytest.raises(ValueError, match="requires a wandb_logger"):
            Trainer(model=model, logger=_make_logger(tmp_path), wandb_logger=None)


# ─────────────────────────────────────────────
# Attribute setup
# ─────────────────────────────────────────────

class TestTrainerAttributes:

    def test_default_training_args(self, tmp_path):
        model = StubTaskModel()
        trainer = Trainer(
            model=model,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.training_args is not None
        assert isinstance(trainer.training_args, TrainingArgument)

    def test_custom_training_args(self, tmp_path):
        model = StubTaskModel()
        args = TrainingArgument(epochs=5, output_dir="runs/custom")
        trainer = Trainer(
            model=model,
            training_args=args,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.training_args.epochs == 5
        assert trainer.training_args.output_dir == "runs/custom"

    def test_device_from_model(self, tmp_path):
        model = StubTaskModel()  # on CPU
        trainer = Trainer(
            model=model,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.device == torch.device("cpu")

    def test_has_enhancement_false(self, tmp_path):
        model = StubTaskModel()
        trainer = Trainer(
            model=model,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.has_enhancement is False

    def test_cur_epoch_zero(self, tmp_path):
        model = StubTaskModel()
        trainer = Trainer(
            model=model,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.cur_epoch == 0

    def test_checkpoint_path_stored(self, tmp_path):
        model = StubTaskModel()
        trainer = Trainer(
            model=model,
            checkpoint_path="/some/ckpt.pt",
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.checkpoint_path == "/some/ckpt.pt"

    def test_callbacks_initialized(self, tmp_path):
        model = StubTaskModel()
        trainer = Trainer(
            model=model,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert hasattr(trainer, "callbacks")
        assert trainer.callbacks.callbacks == []


# ─────────────────────────────────────────────
# Optimizer and scheduler auto-creation
# ─────────────────────────────────────────────

class TestTrainerOptimizerScheduler:

    def test_auto_creates_optimizer(self, tmp_path):
        model = StubTaskModel()
        trainer = Trainer(
            model=model,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.optimizer is not None

    def test_auto_creates_scheduler(self, tmp_path):
        model = StubTaskModel()
        trainer = Trainer(
            model=model,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.lr_scheduler is not None

    def test_provided_optimizer_kept(self, tmp_path):
        model = StubTaskModel()
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(
            model=model,
            optimizer=optim,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.optimizer is optim


# ─────────────────────────────────────────────
# EvalMetrics initialization
# ─────────────────────────────────────────────

class TestTrainerEvalMetrics:

    def test_eval_metrics_created_for_val_dataloaders(self, tmp_path):
        model = StubTaskModel()
        val_dls = _make_val_dataloaders()
        trainer = Trainer(
            model=model,
            val_dataloaders=val_dls,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert "val" in trainer.eval_metrics
        assert isinstance(trainer.eval_metrics["val"], EvalMetrics)

    def test_eval_metrics_empty_when_no_val(self, tmp_path):
        model = StubTaskModel()
        trainer = Trainer(
            model=model,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert trainer.eval_metrics == {}

    def test_eval_metrics_multiple_prefixes(self, tmp_path):
        model = StubTaskModel()
        ds = TensorDataset(torch.randn(4, 3, 8, 8), torch.randn(4, 6))
        val_dls = {
            "val_clean": DataLoader(ds, batch_size=2),
            "val_foggy": DataLoader(ds, batch_size=2),
        }
        trainer = Trainer(
            model=model,
            val_dataloaders=val_dls,
            logger=_make_logger(tmp_path),
            wandb_logger=_make_wandb_logger(),
        )
        assert "val_clean" in trainer.eval_metrics
        assert "val_foggy" in trainer.eval_metrics
