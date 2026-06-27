import argparse

import torch

import warnings
warnings.simplefilter("once", UserWarning)

from panoptic_perception.models.teacher_model import TeacherFusion
from panoptic_perception.trainer.teacher_trainer import (
    TeacherTrainer, load_yolop_into_backbones,
)

from panoptic_perception.utils.config_parser import load_json

# Reuse the train_v2 factories — single source of truth for arg parsing,
# loss, optimizer, scheduler, dataloader, callbacks, wandb, logger.
from panoptic_perception.scripts.train.train_v2 import (
    create_training_arguments,
    create_loss_function,
    create_model,
    create_optimizer,
    create_scheduler,
    create_logger,
    create_dataloader,
    create_callbacks,
    create_wandb_logger,
)


def create_teacher_model(image_model_kwargs: dict, depth_model_kwargs: dict,
                         fusion_kwargs: dict, loss_kwargs: dict):
    if not fusion_kwargs:
        raise ValueError("teacher training requires non-empty fusion_kwargs")

    image_model, device = create_model(image_model_kwargs, loss_kwargs)
    depth_model, _ = create_model(depth_model_kwargs, loss_kwargs)

    # Depth backbone is forward_backbone-only; loss function attached here would
    # be dead state and wastes optimizer/buffer memory.
    depth_model.loss_function = None

    teacher_model = TeacherFusion(
        image_model=image_model,
        depth_model=depth_model,
        fusion_kwargs=fusion_kwargs,
    )
    teacher_model.to(device)

    return teacher_model, device


def main(args: argparse.Namespace):

    config_fn = args.config
    config = load_json(config_fn)

    training_args = create_training_arguments(config)

    logger = create_logger(training_args)
    logger.log_line()
    logger.log_message("=== Teacher Training Arguments ===")
    logger.log_message(f"Output dir       : {training_args.output_dir}")
    logger.log_message(f"Epochs           : {training_args.epochs}")
    logger.log_message(f"Grad accumulation: {training_args.gradient_accumulation_steps}")
    logger.log_message(f"Grad clipping    : {training_args.gradient_clipping}")
    logger.log_message(f"Warmup epochs    : {training_args.warmup_epochs}")
    logger.log_new_line()

    wandb_logger = create_wandb_logger(config, training_args)
    logger.log_message("=== WandB Logger ===")
    logger.log_message(f"Enabled          : {training_args.wandb_enabled}")
    logger.log_message(f"Project          : {config.get('trainer_kwargs', {}).get('wandb_project', 'N/A')}")
    logger.log_new_line()

    # --- Dataloader ----------------------------------------------------------
    # For teacher training, dataset_kwargs MUST have:
    #   - dataset_class: "foggy"     (FoggyBDDPreprocessor provides depth_maps)
    #   - apply_fog:     false       (clean image + depth, no fog blend)
    dataset_kwargs = config["dataset_kwargs"]
    if dataset_kwargs.get("dataset_class") != "foggy":
        raise ValueError(
            f"teacher requires dataset_class='foggy' (for depth_maps); "
            f"got dataset_class={dataset_kwargs.get('dataset_class')!r}"
        )
    if dataset_kwargs.get("apply_fog", True):
        raise ValueError(
            "teacher requires apply_fog=false (privileged-information path: "
            "clean image + depth, no synthetic fog)"
        )

    logger.log_message("=== Building Datasets & DataLoaders ===")
    train_dataloader, val_dataloaders = create_dataloader(dataset_kwargs, logger=logger)

    # DataLoaderBuilder.build_val() returns {val_clean, val_foggy} when dataset_class=foggy.
    # For the teacher (apply_fog=False):
    #   - val_clean uses BDD100KDataset → no depth_maps → would crash _forward_model
    #   - val_foggy uses FoggyBDD100KDataset with apply_fog=False → emits clean+depth
    # So we keep ONLY the Foggy-built loader and rename it 'val_clean' (semantically
    # it IS clean data — no fog applied — just routed through the depth-aware preprocessor).
    # Dropping val_foggy also skips the wasted fog-eval pass.
    if "val_foggy" in (val_dataloaders or {}):
        val_dataloaders = {"val_clean": val_dataloaders["val_foggy"]}
    else:
        raise RuntimeError(
            "Expected 'val_foggy' in val_dataloaders for teacher mode (dataset_class=foggy "
            "+ apply_fog=False). DataLoaderBuilder.build_val() may have changed its key shape."
        )

    logger.log_message(f"Train Dataset    : {train_dataloader.dataset.__class__.__name__}")
    logger.log_message(f"Train batches    : {len(train_dataloader)}")
    logger.log_message(f"Train batch size : {dataset_kwargs.get('train_batch_size', 'N/A')}")
    logger.log_message(f"Train workers    : {dataset_kwargs.get('train_num_workers', 'N/A')}")
    for prefix, dl in (val_dataloaders or {}).items():
        logger.log_message(f"Val Dataset      : {prefix} -> {dl.dataset.__class__.__name__}")
        logger.log_message(f"Val batches      : {len(dl)}")
    logger.log_new_line()

    # --- Model ---------------------------------------------------------------
    required = ("image_model_kwargs", "depth_model_kwargs", "fusion_kwargs")
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"teacher config missing required keys: {missing}")

    logger.log_message("=== Creating Teacher Model ===")
    model, device = create_teacher_model(
        image_model_kwargs=config["image_model_kwargs"],
        depth_model_kwargs=config["depth_model_kwargs"],
        fusion_kwargs=config["fusion_kwargs"],
        loss_kwargs=config.get("loss_kwargs"),
    )
    logger.log_message(f"Model type       : {model.__class__.__name__}")
    logger.log_message(f"Device           : {device}")
    logger.log_message(f"Image params     : {sum(p.numel() for p in model.image_model.parameters()):,}")
    logger.log_message(f"Depth params     : {sum(p.numel() for p in model.depth_model.parameters()):,}")
    logger.log_message(f"Fusion params    : {sum(p.numel() for p in model.fusion_blocks.parameters()):,}")
    logger.log_message(f"Total params     : {sum(p.numel() for p in model.parameters()):,}")
    logger.log_message(f"Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.log_message(f"Active tasks     : {model.image_model.get_active_tasks()}")
    logger.log_new_line()

    # --- Warm-start backbones from YOLOP checkpoints (optional) --------------
    image_init = config["image_model_kwargs"].get("backbone_init")
    depth_init = config["depth_model_kwargs"].get("backbone_init")
    if image_init:
        logger.log_message("=== Loading YOLOP checkpoints into backbones ===")
        load_yolop_into_backbones(
            teacher_model=model,
            image_backbone_path=image_init,
            depth_backbone_path=depth_init,
            logger=logger,
        )
        logger.log_new_line()
    else:
        logger.log_message("=== Skipping backbone warm-start (no image_backbone_init in model_kwargs) ===")
        logger.log_new_line()

    # --- Loss ----------------------------------------------------------------
    loss_kwargs = config.get("loss_kwargs", {})
    logger.log_message("=== Loss Configuration ===")
    for task_name, task_cfg in loss_kwargs.items():
        if task_name == "loss_weights":
            continue
        loss_type = task_cfg.get("_type", "?")
        extra = task_cfg.get("kwargs", {})
        extra_str = f" ({extra})" if extra else ""
        logger.log_message(f"  {task_name:25} : {loss_type}{extra_str}")
    weights = loss_kwargs.get("loss_weights", {})
    if weights:
        logger.log_message("Loss weights:")
        for task_name, weight in weights.items():
            logger.log_message(f"  {task_name:25} : {weight}")
    logger.log_new_line()

    # --- Optimizer + Scheduler ----------------------------------------------
    # create_optimizer calls model.get_param_groups(...) — TeacherFusion's
    # implementation merges image + depth + fusion groups, with per-backbone
    # freeze knobs under optimizer_kwargs.image_backbone_kwargs / depth_backbone_kwargs.
    logger.log_message("=== Creating Optimizer ===")
    optimizer = create_optimizer(
        model,
        config["optimizer_kwargs"],
        training_args,
        logger,
    ) if "optimizer_kwargs" in config else None

    if optimizer:
        logger.log_message(f"Optimizer        : {optimizer.__class__.__name__}")
        logger.log_message(f"Initial LR       : {config['optimizer_kwargs'].get('initial_lr', 'N/A')}")
        logger.log_message(f"Weight decay     : {config['optimizer_kwargs'].get('weight_decay', 'N/A')}")
        logger.log_message(f"Param groups     : {len(optimizer.param_groups)}")
        logger.log_new_line()

        logger.log_message("=== Creating LR Scheduler ===")
        lr_scheduler = create_scheduler(optimizer, training_args)
        logger.log_message(f"Scheduler        : {lr_scheduler.__class__.__name__}")
    else:
        lr_scheduler = None
        logger.log_message("No optimizer provided, skipping scheduler")
    logger.log_new_line()

    # --- Callbacks -----------------------------------------------------------
    callbacks = create_callbacks(config)
    logger.log_message("=== Callbacks ===")
    for cb in callbacks:
        logger.log_message(f"  - {cb.__class__.__name__}")
    logger.log_new_line()

    # --- Trainer -------------------------------------------------------------
    logger.log_message("=== Creating Teacher Trainer ===")
    checkpoint_path = config.get("trainer_kwargs", {}).get("checkpoint_path")
    logger.log_message(f"Checkpoint (resume) : {checkpoint_path or 'None'}")

    trainer = TeacherTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training_args=training_args,
        wandb_logger=wandb_logger,
        logger=logger,
        checkpoint_path=checkpoint_path,
    )

    for cb in callbacks:
        trainer.callbacks.add_callback(cb)

    # --- WandB config dump (reproducibility) ---------------------------------
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb_logger.update_config({
        "runtime/torch_version":  torch.__version__,
        "runtime/cuda_available": torch.cuda.is_available(),
        "runtime/cuda_version":   torch.version.cuda if torch.cuda.is_available() else None,
        "runtime/gpu_name":       torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "runtime/device":         str(device),
        "model/class":            model.__class__.__name__,
        "model/total_params":     total_params,
        "model/trainable_params": trainable_params,
        "model/image_params":     sum(p.numel() for p in model.image_model.parameters()),
        "model/depth_params":     sum(p.numel() for p in model.depth_model.parameters()),
        "model/fusion_params":    sum(p.numel() for p in model.fusion_blocks.parameters()),
        "model/active_tasks":     model.image_model.get_active_tasks(),
        "model/image_cfg_path":   config["image_model_kwargs"].get("cfg_path"),
        "model/depth_cfg_path":   config["depth_model_kwargs"].get("cfg_path"),
        "model/image_backbone_init": image_init,
        "model/depth_backbone_init": depth_init,
        "fusion/type":            config["fusion_kwargs"].get("fusion_type"),
        "fusion/num_blocks":      config["fusion_kwargs"].get("num_fusion_blocks"),
        "fusion/weighted":        config["fusion_kwargs"].get("weighted_fusion"),
        "fusion/intercepts":      str(config["fusion_kwargs"].get("backbone_intercepts")),
        "fusion/aux_depth_recon_cfg": config["fusion_kwargs"].get("aux_depth_recon_cfg"),
        "optimizer/type":         optimizer.__class__.__name__ if optimizer else None,
        "optimizer/initial_lr":   config.get("optimizer_kwargs", {}).get("initial_lr"),
        "optimizer/weight_decay": config.get("optimizer_kwargs", {}).get("weight_decay"),
        "optimizer/param_groups": len(optimizer.param_groups) if optimizer else 0,
        "scheduler/type":         lr_scheduler.__class__.__name__ if lr_scheduler else None,
        "data/train_batches":     len(train_dataloader),
        "data/train_batch_size":  dataset_kwargs.get("train_batch_size"),
        "data/train_workers":     dataset_kwargs.get("train_num_workers"),
        "data/image_resize":      dataset_kwargs.get("train_preprocessor_kwargs", {}).get("image_resize"),
        "data/apply_fog":         dataset_kwargs.get("apply_fog", True),
        "data/depth_backend":     dataset_kwargs.get("adverse_params", {}).get("depth_backend"),
        "trainer/checkpoint_path":            checkpoint_path,
        "trainer/gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "trainer/gradient_clipping":          training_args.gradient_clipping,
        "trainer/warmup_epochs":              training_args.warmup_epochs,
    })

    logger.log_line()
    logger.log_message("=== Starting Teacher Training ===")
    logger.log_new_line()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Teacher Fusion (privileged-information: clean image + depth)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to teacher training config JSON file",
    )
    args = parser.parse_args()

    main(args)
