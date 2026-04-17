"""
Unit tests for callback classes.
Tests the Callbacks composite, CheckpointCallback, EvalMetricsCallback,
and the create_callbacks factory from train_v2.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from collections import defaultdict

from panoptic_perception.trainer.callbacks import (
    TrainerCallback, Callbacks, CheckpointCallback,
    EnhancedImageLogger, EvalMetricsCallback,
)
from panoptic_perception.trainer.utils import EvalMetrics, EvalBatchContext
from panoptic_perception.scripts.train.train_v2 import (
    create_callbacks, CALLBACK_REGISTRY, DEFAULT_CALLBACKS,
)


# ─────────────────────────────────────────────
# Callbacks (composite)
# ─────────────────────────────────────────────

class TestCallbacksComposite:

    def test_empty_callbacks(self):
        cb = Callbacks()
        assert cb.callbacks == []
        # Should not raise on any hook with a mock trainer
        cb.on_train_begin(MagicMock())
        cb.on_epoch_end(MagicMock())

    def test_add_callback(self):
        cb = Callbacks()
        mock_cb = MagicMock(spec=TrainerCallback)
        cb.add_callback(mock_cb)
        assert len(cb.callbacks) == 1

    def test_hooks_dispatched(self):
        mock_cb1 = MagicMock(spec=TrainerCallback)
        mock_cb2 = MagicMock(spec=TrainerCallback)
        cb = Callbacks([mock_cb1, mock_cb2])

        trainer = MagicMock()
        cb.on_train_begin(trainer)
        mock_cb1.on_train_begin.assert_called_once_with(trainer)
        mock_cb2.on_train_begin.assert_called_once_with(trainer)

        cb.on_epoch_end(trainer)
        mock_cb1.on_epoch_end.assert_called_once_with(trainer)
        mock_cb2.on_epoch_end.assert_called_once_with(trainer)

    def test_all_hooks_exist(self):
        """Verify Callbacks exposes every hook from TrainerCallback."""
        hooks = [
            "on_train_begin", "on_train_end",
            "on_epoch_begin", "on_epoch_end",
            "on_step_begin", "on_step_end",
            "on_eval_begin", "on_eval_batch_end", "on_eval_end",
        ]
        cb = Callbacks()
        for hook in hooks:
            assert hasattr(cb, hook), f"Missing hook: {hook}"


# ─────────────────────────────────────────────
# CheckpointCallback
# ─────────────────────────────────────────────

class TestCheckpointCallback:

    def test_init_state(self):
        cb = CheckpointCallback()
        assert isinstance(cb.best_map, defaultdict)
        assert isinstance(cb.best_epoch, defaultdict)
        assert cb.best_map["any_key"] == 0.0
        assert cb.best_epoch["any_key"] == 0

    def test_resume_no_checkpoint_path(self):
        """Should return early without error when checkpoint_path is None."""
        cb = CheckpointCallback()
        trainer = MagicMock()
        trainer.checkpoint_path = None
        cb.resume_from_checkpoint(trainer)
        # No crash = pass

    def test_resume_nonexistent_file(self):
        """Should return early when file doesn't exist."""
        cb = CheckpointCallback()
        trainer = MagicMock()
        trainer.checkpoint_path = "/nonexistent/path/ckpt.pt"
        cb.resume_from_checkpoint(trainer)
        # No crash = pass

    def test_save_best_model_skips_when_no_eval(self):
        """_save_best_model should skip prefixes where ap_per_class is None."""
        cb = CheckpointCallback()
        trainer = MagicMock()
        trainer.eval_metrics = {
            "val": EvalMetrics(metric_prefix="val"),  # ap_per_class is None
        }
        trainer.training_args.output_dir = "/tmp/test_output"
        # Should not crash
        with patch("os.path.exists", return_value=True):
            with patch("os.makedirs"):
                cb._save_best_model(trainer)


# ─────────────────────────────────────────────
# EvalMetricsCallback
# ─────────────────────────────────────────────

class TestEvalMetricsCallback:

    def test_init_defaults(self):
        cb = EvalMetricsCallback()
        assert cb.conf_threshold == 0.001
        assert cb.iou_threshold == 0.45
        assert cb.max_detections == 500
        assert cb.num_drivable_classes == 2
        assert cb.total_val_loss == 0.0
        assert cb.global_image_idx == 0

    def test_init_custom_args(self):
        cb = EvalMetricsCallback(
            conf_threshold=0.5,
            iou_threshold=0.6,
            max_detections=100,
            num_drivable_classes=3,
        )
        assert cb.conf_threshold == 0.5
        assert cb.iou_threshold == 0.6
        assert cb.max_detections == 100
        assert cb.num_drivable_classes == 3

    def test_reset(self):
        cb = EvalMetricsCallback()
        cb.total_val_loss = 10.0
        cb.total_det_loss = 5.0
        cb.global_image_idx = 42
        cb._reset()
        assert cb.total_val_loss == 0.0
        assert cb.total_det_loss == 0.0
        assert cb.global_image_idx == 0

    def test_drivable_confusion_matrix_shape(self):
        cb = EvalMetricsCallback(num_drivable_classes=3)
        assert cb.drivable_confusion_matrix.shape == (3, 3)
        assert cb.drivable_confusion_matrix.sum() == 0

    def test_lane_confusion_matrix_starts_none(self):
        cb = EvalMetricsCallback()
        assert cb.lane_confusion_matrix is None


# ─────────────────────────────────────────────
# EnhancedImageLogger
# ─────────────────────────────────────────────

class TestEnhancedImageLogger:

    def test_init_defaults(self):
        cb = EnhancedImageLogger()
        assert cb.train_log_idx == 200
        assert cb.eval_log_idx == 200

    def test_init_custom(self):
        cb = EnhancedImageLogger(train_log_idx=50, eval_log_idx=100)
        assert cb.train_log_idx == 50
        assert cb.eval_log_idx == 100


# ─────────────────────────────────────────────
# create_callbacks (train_v2 factory)
# ─────────────────────────────────────────────

class TestCreateCallbacks:

    def test_defaults_when_no_config(self):
        callbacks = create_callbacks({})
        assert len(callbacks) == len(DEFAULT_CALLBACKS)
        types = [type(cb).__name__ for cb in callbacks]
        assert "CheckpointCallback" in types
        assert "EvalMetricsCallback" in types

    def test_explicit_config(self):
        config = {
            "callbacks": {
                "checkpoint": {},
                "enhanced_image_logger": {"train_log_idx": 50},
            }
        }
        callbacks = create_callbacks(config)
        assert len(callbacks) == 2
        types = [type(cb).__name__ for cb in callbacks]
        assert "CheckpointCallback" in types
        assert "EnhancedImageLogger" in types

    def test_unknown_callback_raises(self):
        config = {"callbacks": {"nonexistent_callback": {}}}
        with pytest.raises(KeyError, match="Unknown callback"):
            create_callbacks(config)

    def test_registry_keys(self):
        assert "checkpoint" in CALLBACK_REGISTRY
        assert "enhanced_image_logger" in CALLBACK_REGISTRY
        assert "eval_metrics" in CALLBACK_REGISTRY


# ─────────────────────────────────────────────
# EvalMetrics dataclass
# ─────────────────────────────────────────────

class TestEvalMetrics:

    def test_defaults(self):
        em = EvalMetrics(metric_prefix="val")
        assert em.ap_per_class is None
        assert em.stats_per_class is None
        assert em.drivable_metrics is None

    def test_reset(self):
        em = EvalMetrics(metric_prefix="val")
        em.ap_per_class = {"mAP": 0.5}
        em.stats_per_class = {"cls0": {}}
        em.reset()
        assert em.ap_per_class is None
        assert em.stats_per_class is None
        # metric_prefix should NOT be reset
        assert em.metric_prefix == "val"


# ─────────────────────────────────────────────
# EvalBatchContext dataclass
# ─────────────────────────────────────────────

class TestEvalBatchContext:

    def test_all_fields_none_by_default(self):
        ctx = EvalBatchContext()
        assert ctx.cur_eval_model_outputs is None
        assert ctx.cur_eval_image_h is None
        assert ctx.cur_eval_images is None
        assert ctx.ap_table_data is None
