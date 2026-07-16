import os
import torch
import argparse

from panoptic_perception.models import ModelFactory
from panoptic_perception.losses.multi_task_loss import MultiTaskLoss
from panoptic_perception.utils.config_parser import load_json, parse_config
from panoptic_perception.dataset.coco_image_dataset import DataLoaderBuilder
from panoptic_perception.trainer.trainer import PanopticTrainer
from panoptic_perception.scripts.train.pretrain_swin_imagenet import (
    create_training_arguments, create_logger, 
    create_optimizer, create_scheduler, create_wandb_logger
)

from panoptic_perception.scripts.train.train import (
    create_loss_function, create_callbacks
)

def create_model(model_kwargs:dict, loss_kwargs:dict):

    device = model_kwargs.get("device", "cuda")

    model = ModelFactory.from_config(model_kwargs)
    device = torch.device(device) if torch.cuda.is_available() and "cuda" in device else torch.device("cpu")
    model.to(device)

    loss_func = create_loss_function(loss_kwargs)
    model.loss_function = loss_func

    return model, device

def create_dataloader(dataset_kwargs, logger=None):

    builder = DataLoaderBuilder(dataset_kwargs, logger=logger)

    train_dataloader = builder._build_train()
    val_dataloader = builder._build_val()

    return train_dataloader, val_dataloader

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
    logger.log_message(f"Train batch size : {train_dataloader.batch_size}")
    logger.log_message(f"Train workers    : {train_dataloader.num_workers}")

    logger.log_message(f'Val Dataset    : {val_dataloaders.dataset.__class__.__name__}')
    logger.log_message(f"Val batches    : {len(val_dataloaders)}")
    logger.log_message(f"Val batch size   : {val_dataloaders.batch_size}")
    logger.log_message(f"Val workers      : {val_dataloaders.num_workers}")
    logger.log_new_line()

    if "model_kwargs" not in config:
        raise ValueError(f"Expected Model Kwargs dict")

    logger.log_message("=== Creating Model ===")
    model, device = create_model(config["model_kwargs"], config.get("loss_kwargs"))
    logger.log_message(f"Model type       : {model.__class__.__name__}")
    logger.log_message(f"Device           : {device}")
    logger.log_message(f"Parameters       : {sum(p.numel() for p in model.parameters()):,}")
    logger.log_message(f"Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    loss_kwargs = config.get("loss_kwargs", {})
    
    logger.log_message("=== Loss Configuration ===")
    for task_name, task_cfg in loss_kwargs.items():
        if task_name == "loss_weights":
            continue   # printed separately below  
        loss_type = task_cfg.get("_type", "?")
        extra = task_cfg.get("kwargs", {})
        extra_str = f" ({extra})" if extra else "" 
        logger.log_message(f"  {task_name:25} : {loss_type}{extra_str}")

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

    logger.log_new_line()

    callbacks = create_callbacks(config, class_names_enum=val_dataloaders.dataset.class_names_enum)
    logger.log_message("=== Callbacks ===")
    for cb in callbacks:
        logger.log_message(f"  - {cb.__class__.__name__}")
    logger.log_new_line()

    logger.log_message("=== Creating Trainer ===")
    checkpoint_path = config.get("trainer_kwargs", {}).get("checkpoint_path")
    logger.log_message(f"Checkpoint       : {checkpoint_path or 'None'}")        

    trainer = PanopticTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders={"val": val_dataloaders},
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        training_args=training_args,
        wandb_logger=wandb_logger,
        logger=logger,
        checkpoint_path=checkpoint_path,
    )    

    for cb in callbacks:
        trainer.callbacks.add_callback(cb)    

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
    parser = argparse.ArgumentParser(description="Train Swin Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="panoptic_perception/configs/trainer/train_kwargs_swin_pretrain.json",
        help="Path to training config JSON file"
    )
    
    args = parser.parse_args()

    main(args)    