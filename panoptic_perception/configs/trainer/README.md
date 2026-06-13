# Trainer Config Guide

This directory holds the JSON config files that drive training. Treat `train_kwargs_optimized_drivable.json` as the canonical example — start by copying it and editing the fields you care about.

## Launching a training run

```bash
python3 -m panoptic_perception.scripts.train.train_v2 \
    --config panoptic_perception/configs/trainer/train_kwargs_optimized_drivable.json
```

> `panoptic_perception/scripts/train/train.py` and `panoptic_perception/trainer/trainer.py` are **deprecated**. Use `train_v2.py` + `trainer_refactor.Trainer`. The legacy files emit a `DeprecationWarning` on import and will be removed.

The new entrypoint uses:
- `ModelFactory` for model construction (`model_type` is a registry key)
- `LossFactory` + `MultiTaskLoss` for composable loss configuration (`_type` is a registry key per task)
- `TrainingArgument` for config parsing and validation
- `Trainer` from `trainer_refactor.py` for the training loop

## Config structure

The JSON has five top-level sections. Each is documented below.

### 1. `model_kwargs`

```jsonc
"model_kwargs": {
    "model_type": "yolop",                              // registry key
    "cfg_path":  "<path>/yolo-768-1280-detection.cfg",  // architecture definition
    "device":    "cuda:0"
}
```

- `model_type`: one of `"yolop"`, `"yolov8p"`. For enhancement-wrapped variants, set `use_gdip: true` or `use_denet: true` and provide the matching `gdip_kwargs` / `denet_kwargs` block.
- `cfg_path`: model architecture file under `configs/models/`. Must match the resolution you intend to train at.

### 2. `dataset_kwargs`

```jsonc
"dataset_kwargs": {
    "images_dir":                     "<path>/100k/100k",
    "detection_annotations_dir":      "<path>/bdd100k_labels/100k",
    "segmentation_annotations_dir":   "",            // empty string = skip task
    "drivable_annotations_dir":       "<path>/drivable_maps/labels",

    "train_batch_size": 16,
    "train_shuffle":    true,
    "train_num_workers": 4,
    "train_preprocessor_kwargs": { ... },

    "val_batch_size":   16,
    "val_shuffle":      false,
    "val_num_workers":  4,
    "val_preprocessor_kwargs":   { ... }
}
```

Key behaviors:
- An **empty string** annotation dir drops that task from `active_tasks`. The matching loss will not be computed even if it appears in `loss_kwargs`.
- `train_preprocessor_kwargs.image_resize` must match the model's `cfg_path` resolution.
- Augmentations live under `train_preprocessor_kwargs.augment_params` (geometric, color, noise) and `train_preprocessor_kwargs.advanced_aug` (mosaic, mixup, copy-paste — all default 0 and only enable for long runs).
- `val_preprocessor_kwargs` should have `perform_augmentation` absent or false.

For the **foggy** dataset variant, add `"dataset_class": "foggy"` and an `adverse_params` block — see `panoptic_perception/dataset/__init__.py::DataLoaderBuilder` for the foggy-specific knobs (`fog_betas`, `darkness_gammas`, `depth_backend`, etc.). Foggy training forces `num_workers=0` because depth estimation runs on the main process GPU.

### 3. `optimizer_kwargs`

```jsonc
"optimizer_kwargs": {
    "_type":          "AdamW",   // "AdamW" | "SGD" | "Adagrad" (case-insensitive)
    "initial_lr":     1e-3,
    "weight_decay":   0.01,
    "momentum":       0.937,     // used by SGD; ignored by AdamW
    "warmup_bias_lr": 0.1,
    "warmup_momentum": 0.8,
    "warmup_epochs":  3,
    "groups": {
        "backbone":     { "group": [0, 9],   "trainable": true },
        "fpn":          { "group": [10, 16], "trainable": true },
        "det_pan":      { "group": [17, 23], "trainable": true },
        "detect_head":  { "group": [24],     "trainable": true },
        "segmentation": { "group": [25, 33], "trainable": true }
    }
}
```

`groups` controls per-layer-range trainability:
- `group`: `[start_idx, end_idx]` inclusive over the model's flat layer list (indices match the `.cfg` file).
- `trainable: false` sets `requires_grad=False` on every parameter in that range. **If the only active task's path is fully frozen, `loss.backward()` will fail with `element 0 of tensors does not require grad`** — make sure at least one group on the loss's path is trainable.
- Common patterns:
  - **Fine-tune everything**: all groups `trainable: true`
  - **Frozen backbone**: `backbone.trainable: false`, everything else true
  - **Head-only training**: only `detect_head` / `segmentation` true

### 4. `lr_scheduler_kwargs`

```jsonc
"lr_scheduler_kwargs": {
    "_type": "cosine",   // "cosine" | "linear" | "step" | "multistep"
                         // | "exponential" | "onecycle" | "cosine_restart"
    "cosine_annealing_lr_kwargs": { "eta_min": 5e-6 },
    "linear_lr_kwargs":           { "start_factor": 1.0, "end_factor": 0.01 }
}
```

Each scheduler reads its own `*_kwargs` sub-block. Supplying blocks for schedulers you aren't using is harmless. See `trainer_schedulers.py::LR_SCHEDULER_HANDLER` for the dispatch table.

### 5. `trainer_kwargs`

```jsonc
"trainer_kwargs": {
    "output_dir":          "yolop-ATSS",  // checkpoints + logs land here
    "is_training":         true,
    "epochs":              100,
    "first_val_epoch":     0,
    "lr_scheduler_start_epoch": 1,

    "gradient_clipping":           1.0,
    "gradient_accumulation_steps": 2,

    "checkpoint_idx":  2,        // every Nth epoch save a checkpoint
    "checkpoint_path": "<path>/best_model.pt",   // resume from here (optional)
    "reload_optimizer": false,
    "reload_optimizer_with_initial_lr": false,

    "monitor_train":     true,
    "monitor_val":       true,
    "metric_eval_mode":  "strict",   // "strict" | "lenient"
    "metric_average_mode": "macro",  // "macro"  | "micro"

    "wandb_enabled":  true,
    "wandb_project":  "yolop-ATSS",

    "use_ema":           false,
    "ema_decay":         0.9999,
    "ema_warmup_steps":  2000
}
```

Resuming a run: set `checkpoint_path` to the `.pt` file. `reload_optimizer: true` restores optimizer state; `reload_optimizer_with_initial_lr: true` keeps the schedule but resets to `initial_lr` (useful for LR-bump fine-tuning).

### 6. `loss_kwargs`

```jsonc
"loss_kwargs": {
    "detection":            { "_type": "detection-loss-ATSS", "kwargs": {} },
    "drivable_segmentation":{ "_type": "segmentation-loss",   "kwargs": {} },
    "lane_detection":       { "_type": "lane-detection-loss", "kwargs": {} },

    "loss_weights": {
        "detection":              1.0,
        "drivable_segmentation":  1.0,
        "lane_detection":         1.0,
        "lane_segmentation":      0.0   // set to 0.0 to disable
    }
}
```

`_type` registry keys (registered via `@LossFactory.register_loss_function(...)`):
- `detection-loss-anchor` — classic anchor-based YOLO detection loss
- `detection-loss-anchor-free` — anchor-free variant
- `detection-loss-ATSS` — Adaptive Training Sample Selection matcher (recommended)
- `segmentation-loss` — drivable area segmentation
- `lane-detection-loss` — lane keypoints/regression

`MultiTaskLoss` only computes losses for tasks where:
1. The annotation dir is non-empty in `dataset_kwargs` (drives `active_tasks`)
2. The task appears in `loss_kwargs` with a valid `_type`
3. `loss_weights[task] > 0.0`

Setting a weight to `0.0` is the runtime kill-switch — the loss factory still registers the function, but it contributes nothing to the gradient.

## Common modifications

| Goal | Edit |
|---|---|
| Train detection only | Empty `drivable_annotations_dir` and `segmentation_annotations_dir`; drop those entries from `loss_weights` (or set to 0.0) |
| Freeze backbone, fine-tune heads | `optimizer_kwargs.groups.backbone.trainable: false` |
| Smaller GPU | Lower `train_batch_size`, raise `gradient_accumulation_steps` proportionally |
| Switch to anchor-free | Change detection model's `cfg_path` to the anchor-free variant *and* `loss_kwargs.detection._type` to `"detection-loss-anchor-free"` — both must match |
| Resume from checkpoint | Set `checkpoint_path`; `reload_optimizer: true` if you want optimizer state preserved |
| Disable WandB | `wandb_enabled: false` (no `WANDB_API_KEY` needed) |

## Troubleshooting

- **`RuntimeError: element 0 of tensors does not require grad`** — every layer group in the active task's grad path is frozen. Flip the relevant group to `trainable: true`.
- **Trainer log shows fewer tasks than expected** — check that the annotation directories are non-empty strings and the directories actually exist on disk.
- **Loss explodes in epoch 0 with ATSS** — normal for the first few hundred iterations while the matcher stabilizes. If it doesn't recover by iter ~500, lower `initial_lr` or shorten `warmup_epochs`.
- **`KeyError: <task> not registered`** — the `_type` value doesn't match a registered loss factory key. Grep for `@LossFactory.register_loss_function` to see what's available.
