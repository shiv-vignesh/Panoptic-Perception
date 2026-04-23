import json
import os
import argparse

from typing import Union, Optional

import torch

from panoptic_perception.models import ModelFactory, BaseTaskModel, BaseEnhancementModel

from panoptic_perception.utils.logger import Logger
from panoptic_perception.utils.wandb_logger import WandBLogger
from panoptic_perception.utils.config_parser import load_json, parse_config

from panoptic_perception.dataset import DataLoaderBuilder

from panoptic_perception.trainer.trainer_args import TrainingArgument
from panoptic_perception.trainer.trainer_optimizer import OptimizerContext, build_optmizer
from panoptic_perception.trainer.trainer_schedulers import SchedulerContext, build_scheduler
from panoptic_perception.trainer.trainer_refactor import Trainer
from panoptic_perception.trainer.callbacks import (
    CheckpointCallback, EnhancedImageLogger, EvalMetricsCallback
)

CALLBACK_REGISTRY = {
    "checkpoint": CheckpointCallback,
    "enhanced_image_logger": EnhancedImageLogger,
    "eval_metrics": EvalMetricsCallback,
}

DEFAULT_CALLBACKS = ["checkpoint", "eval_metrics"]

def create_training_arguments(config:dict) -> TrainingArgument:
    return TrainingArgument.from_config(parse_config(config))

def create_model(model_kwargs:dict, loss_weights: dict = None) -> Union[BaseTaskModel, BaseEnhancementModel]:

    device = model_kwargs.get("device", "cuda")

    use_gdip = model_kwargs.get("use_gdip", False)
    use_denet = model_kwargs.get("use_denet", False)

    assert not (use_gdip and use_denet), "use_gdip and use_denet cannot both be True"
    if use_gdip:
        assert "gdip_kwargs" in model_kwargs and model_kwargs["gdip_kwargs"], \
            f'Key Error: gdip_kwargs missing'    

        model_kwargs["enhancement"] = "gdip-yolo"

    if use_denet:
        assert "denet_kwargs" in model_kwargs and model_kwargs["denet_kwargs"], \
            f'Key Error: denet_kwargs missing'

        model_kwargs["enhancement"] = "denet-yolo"

    model = ModelFactory.from_config(model_kwargs, loss_weights)
    device = torch.device(device) if torch.cuda.is_available() and "cuda" in device else torch.device("cpu")
    model.to(device)

    return model, device

def create_optimizer(model:Union[BaseTaskModel, BaseEnhancementModel], 
                    optimizer_kwargs:dict, 
                    training_args:TrainingArgument, 
                    logger:Logger) -> torch.optim:

    groups = optimizer_kwargs.get("groups", {})
    has_enhancement = isinstance(model, BaseEnhancementModel)

    if not groups:
        if has_enhancement:
            logger.log_message(f'{model.task_network.__class__.__name__} Full model training (all layers trainable)')
        else:
            logger.log_message(f'{model.__class__.__name__} Full model training (all layers trainable)')

    param_groups = model.get_param_groups(optimizer_kwargs)
    ctx = OptimizerContext(param_groups, training_args)

    return build_optmizer(ctx)

def create_scheduler(optimizer:torch.optim,
                    training_args:TrainingArgument) -> torch.optim.lr_scheduler:

    ctx = SchedulerContext(optimizer, training_args, training_args.epochs)
    return build_scheduler(ctx)

def create_logger(training_args:TrainingArgument) -> Logger:

    log_dir = training_args.output_dir
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, "training.log")
    return Logger(log_file_path=log_file_path, logger_name="panoptic_trainer")

def create_dataloader(dataset_kwargs, logger=None):
    
    builder = DataLoaderBuilder(dataset_kwargs, logger=logger)
    train_dataloader = builder.build_train()                                                                                          
    val_dataloaders = builder.build_val()
    
    return train_dataloader, val_dataloaders

def create_callbacks(config: dict) -> list:
    callbacks_config = config.get("callbacks", None)

    if callbacks_config is None:
        return [CALLBACK_REGISTRY[name]() for name in DEFAULT_CALLBACKS]

    callbacks = []
    for name, kwargs in callbacks_config.items():
        if name not in CALLBACK_REGISTRY:
            raise KeyError(f"Unknown callback: {name}")
        callbacks.append(CALLBACK_REGISTRY[name](**kwargs))

    return callbacks

def create_wandb_logger(config:dict, training_args:TrainingArgument) -> WandBLogger:

    project_name = config.get("trainer_kwargs", {}).get("wandb_project", "panoptic-perception")
    run_name = os.path.basename(training_args.output_dir)
    enabled = training_args.wandb_enabled

    return WandBLogger(
        project_name=project_name,
        run_name=run_name,
        config=config,
        enabled=enabled
    )

def main(args:argparse.Namespace):

    config_fn = args.config
    config = load_json(config_fn)

    training_args = create_training_arguments(config)

    logger = create_logger(training_args)
    logger.log_line()
    logger.log_message("=== Training Arguments ===")
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

    logger.log_message("=== Building Datasets & DataLoaders ===")
    train_dataloader, val_dataloaders = create_dataloader(
        config["dataset_kwargs"], logger=logger
    )
    logger.log_message(f'Train Dataset    : {train_dataloader.dataset.__class__.__name__}')
    logger.log_message(f"Train batches    : {len(train_dataloader)}")
    logger.log_message(f"Train batch size : {config['dataset_kwargs'].get('train_batch_size', 'N/A')}")
    logger.log_message(f"Train workers    : {config['dataset_kwargs'].get('train_num_workers', 'N/A')}")
    for prefix, dl in (val_dataloaders or {}).items():
        logger.log_message(f'Val Dataset    : {dl.dataset.__class__.__name__}')
        logger.log_message(f"Val batches    : {len(dl)}")
    logger.log_message(f"Val batch size   : {config['dataset_kwargs'].get('val_batch_size', 'N/A')}")
    logger.log_message(f"Val workers      : {config['dataset_kwargs'].get('val_num_workers', 'N/A')}")
    logger.log_new_line()

    if "model_kwargs" not in config:
        raise ValueError(f"Expected Model Kwargs dict")

    logger.log_message("=== Creating Model ===")
    model, device = create_model(config["model_kwargs"], config.get("loss_weights"))
    logger.log_message(f"Model type       : {model.__class__.__name__}")
    logger.log_message(f"Device           : {device}")
    logger.log_message(f"Parameters       : {sum(p.numel() for p in model.parameters()):,}")
    logger.log_message(f"Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.log_message(f"Active tasks     : {model.get_active_tasks()}")
    loss_weights = config.get("loss_weights", {})
    logger.log_message(f"Loss weights     : {', '.join(f'{k}={v}' for k, v in loss_weights.items() if not k.startswith('_'))}")
    logger.log_new_line()

    logger.log_message("=== Creating Optimizer ===")
    optimizer = create_optimizer(
            model,
            config["optimizer_kwargs"],
            training_args,
            logger
        ) if "optimizer_kwargs" in config else None

    if optimizer:
        logger.log_message(f"Optimizer        : {optimizer.__class__.__name__}")
        logger.log_message(f"Initial LR       : {config['optimizer_kwargs'].get('initial_lr', 'N/A')}")
        logger.log_message(f"Weight decay     : {config['optimizer_kwargs'].get('weight_decay', 'N/A')}")
        logger.log_message(f"Param groups     : {len(optimizer.param_groups)}")
        logger.log_new_line()

        logger.log_message("=== Creating LR Scheduler ===")
        lr_scheduler = create_scheduler(
            optimizer,
            training_args
        )
        logger.log_message(f"Scheduler        : {lr_scheduler.__class__.__name__}")
    else:
        lr_scheduler = None
        logger.log_message("No optimizer provided, skipping scheduler")
    logger.log_new_line()

    callbacks = create_callbacks(config)
    logger.log_message("=== Callbacks ===")
    for cb in callbacks:
        logger.log_message(f"  - {cb.__class__.__name__}")
    logger.log_new_line()

    logger.log_message("=== Creating Trainer ===")
    checkpoint_path = config.get("trainer_kwargs", {}).get("checkpoint_path")
    logger.log_message(f"Checkpoint       : {checkpoint_path or 'None'}")

    trainer = Trainer(
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

    # Log runtime details to wandb for reproducibility
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb_logger.update_config({
        "runtime/torch_version": torch.__version__,
        "runtime/cuda_available": torch.cuda.is_available(),
        "runtime/cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "runtime/gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "runtime/device": str(device),
        "model/class": model.__class__.__name__,
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/active_tasks": model.get_active_tasks(),
        "model/cfg_path": config["model_kwargs"].get("cfg_path"),
        "optimizer/type": optimizer.__class__.__name__ if optimizer else None,
        "optimizer/initial_lr": config.get("optimizer_kwargs", {}).get("initial_lr"),
        "optimizer/weight_decay": config.get("optimizer_kwargs", {}).get("weight_decay"),
        "optimizer/param_groups": len(optimizer.param_groups) if optimizer else 0,
        "scheduler/type": lr_scheduler.__class__.__name__ if lr_scheduler else None,
        "data/train_batches": len(train_dataloader),
        "data/train_batch_size": config["dataset_kwargs"].get("train_batch_size"),
        "data/train_workers": config["dataset_kwargs"].get("train_num_workers"),
        "data/image_resize": config["dataset_kwargs"].get("train_preprocessor_kwargs", {}).get("image_resize"),
        "trainer/checkpoint_path": checkpoint_path,
        "trainer/gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "trainer/gradient_clipping": training_args.gradient_clipping,
        "trainer/warmup_epochs": training_args.warmup_epochs,
    })

    logger.log_line()
    logger.log_message("=== Starting Training ===")
    logger.log_new_line()
    trainer.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOP Panoptic Perception Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="panoptic_perception/configs/trainer/train_kwargs.json",
        help="Path to training config JSON file"        
    )
    
    args = parser.parse_args()

    main(args)
    