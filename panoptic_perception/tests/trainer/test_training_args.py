"""
Unit tests for TrainingArgument and parse_config.
"""

import pytest
from panoptic_perception.trainer.trainer_args import TrainingArgument
from panoptic_perception.utils.config_parser import parse_config


# ─────────────────────────────────────────────
# TrainingArgument.from_config
# ─────────────────────────────────────────────

class TestTrainingArgumentFromConfig:

    def test_defaults_when_empty(self):
        args = TrainingArgument.from_config({})
        assert args.epochs == 100
        assert args.initial_lr == 1e-3
        assert args.optim_type == "AdamW"
        assert args.monitor_train is True

    def test_override_known_fields(self):
        args = TrainingArgument.from_config({
            "epochs": 50,
            "initial_lr": 5e-4,
            "output_dir": "runs/test",
            "wandb_enabled": False,
        })
        assert args.epochs == 50
        assert args.initial_lr == 5e-4
        assert args.output_dir == "runs/test"
        assert args.wandb_enabled is False

    def test_unknown_fields_ignored(self):
        args = TrainingArgument.from_config({
            "epochs": 10,
            "bogus_key": 999,
            "another_unknown": "hello",
        })
        assert args.epochs == 10
        assert not hasattr(args, "bogus_key")
        assert not hasattr(args, "another_unknown")

    def test_partial_override_keeps_defaults(self):
        args = TrainingArgument.from_config({"gradient_clipping": 5.0})
        assert args.gradient_clipping == 5.0
        assert args.epochs == 100  # default
        assert args.lr_type == "cosine"  # default


# ─────────────────────────────────────────────
# parse_config
# ─────────────────────────────────────────────

class TestParseConfig:

    def test_empty_config(self):
        result = parse_config({})
        assert result == {}

    def test_trainer_kwargs_passthrough(self):
        config = {
            "trainer_kwargs": {
                "output_dir": "runs/exp1",
                "epochs": 25,
                "wandb_enabled": False,
            }
        }
        result = parse_config(config)
        assert result["output_dir"] == "runs/exp1"
        assert result["epochs"] == 25
        assert result["wandb_enabled"] is False

    def test_optimizer_kwargs_selective_keys(self):
        config = {
            "optimizer_kwargs": {
                "initial_lr": 2e-4,
                "weight_decay": 0.05,
                "momentum": 0.9,
                "warmup_epochs": 5,
                # keys NOT in the allowed set should be skipped
                "groups": {"backbone": {"group": [0, 10], "trainable": True}},
            }
        }
        result = parse_config(config)
        assert result["initial_lr"] == 2e-4
        assert result["weight_decay"] == 0.05
        assert result["momentum"] == 0.9
        assert result["warmup_epochs"] == 5
        assert "groups" not in result

    def test_lr_scheduler_kwargs_remapping(self):
        config = {
            "lr_scheduler_kwargs": {
                "_type": "linear",
                "linear_lr_kwargs": {
                    "start_factor": 0.5,
                    "end_factor": 0.001,
                },
            }
        }
        result = parse_config(config)
        assert result["lr_type"] == "linear"
        assert result["linear_lr_start"] == 0.5
        assert result["linear_lr_end"] == 0.001

    def test_cosine_scheduler_remapping(self):
        config = {
            "lr_scheduler_kwargs": {
                "_type": "cosine",
                "cosine_annealing_lr_kwargs": {
                    "eta_min": 1e-7,
                },
            }
        }
        result = parse_config(config)
        assert result["lr_type"] == "cosine"
        assert result["cosine_annealing_eta_min"] == 1e-7

    def test_all_sections_merged(self):
        config = {
            "trainer_kwargs": {"epochs": 50, "output_dir": "runs/merge_test"},
            "optimizer_kwargs": {"initial_lr": 1e-4},
            "lr_scheduler_kwargs": {"_type": "linear"},
        }
        result = parse_config(config)
        assert result["epochs"] == 50
        assert result["initial_lr"] == 1e-4
        assert result["lr_type"] == "linear"


# ─────────────────────────────────────────────
# Round-trip: parse_config → from_config
# ─────────────────────────────────────────────

class TestRoundTrip:

    def test_full_pipeline(self):
        config = {
            "trainer_kwargs": {
                "output_dir": "runs/round_trip",
                "epochs": 30,
                "gradient_clipping": 2.0,
                "wandb_enabled": False,
            },
            "optimizer_kwargs": {
                "initial_lr": 5e-4,
                "weight_decay": 0.001,
                "warmup_epochs": 2,
            },
            "lr_scheduler_kwargs": {
                "_type": "cosine",
                "cosine_annealing_lr_kwargs": {"eta_min": 1e-8},
            },
        }

        flat = parse_config(config)
        args = TrainingArgument.from_config(flat)

        assert args.output_dir == "runs/round_trip"
        assert args.epochs == 30
        assert args.gradient_clipping == 2.0
        assert args.wandb_enabled is False
        assert args.initial_lr == 5e-4
        assert args.weight_decay == 0.001
        assert args.warmup_epochs == 2
        assert args.lr_type == "cosine"
        assert args.cosine_annealing_eta_min == 1e-8
        # defaults preserved
        assert args.monitor_train is True
        assert args.optim_type == "AdamW"
