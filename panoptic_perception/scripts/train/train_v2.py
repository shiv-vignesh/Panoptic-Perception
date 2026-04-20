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

    ctx = SchedulerContext(optimizer, training_args)
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
    wandb_logger = create_wandb_logger(config, training_args)
    
    train_dataloader, val_dataloaders = create_dataloader(
        config["dataset_kwargs"], logger=logger
    )

    if "model_kwargs" not in config:
        raise ValueError(f"Expected Model Kwargs dict")

    model, device = create_model(config["model_kwargs"], config.get("loss_weights"))

    optimizer = create_optimizer(
            model,
            config["optimizer_kwargs"],
            training_args,
            logger
        ) if "optimizer_kwargs" in config else None

    if optimizer:
        lr_scheduler = create_scheduler(
            optimizer,
            training_args
        )
    else:
        lr_scheduler = None    
    
    callbacks = create_callbacks(config)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training_args=training_args,
        wandb_logger=wandb_logger,
        logger=logger,
        checkpoint_path=config.get("trainer_kwargs", {}).get("checkpoint_path"),
    )

    for cb in callbacks:
        trainer.callbacks.add_callback(cb)

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
    