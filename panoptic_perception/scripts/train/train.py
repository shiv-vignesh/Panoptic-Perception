import json
import os
import argparse

import torch

from panoptic_perception.models.models import YOLOP, YOLOv8P, GDIPYolo
from panoptic_perception.trainer.trainer import Trainer


def create_model(model_kwargs: dict, loss_weights: dict = None):
    """Create YOLOP model with optional multi-task loss weights."""
    model_type = model_kwargs["model_type"]
    cfg_path = model_kwargs["cfg_path"]
    device = model_kwargs["device"]
    use_gdip = model_kwargs.get("use_gdip", False)

    assert os.path.exists(cfg_path), f'{cfg_path} does not exists'
    if model_type == "yolop":
        model = YOLOP(cfg_path, loss_weights=loss_weights)
    elif model_type == "yolov8p":
        model = YOLOv8P(cfg_path, loss_weights=loss_weights)
        
    if use_gdip:
        assert "gdip_kwargs" in model_kwargs and model_kwargs["gdip_kwargs"], \
            f'Key Error: gdip_kwargs missing'

        model = GDIPYolo(task_network=model, 
                        gdip_kwargs=model_kwargs["gdip_kwargs"])

    device = torch.device(device) if torch.cuda.is_available() and "cuda" in device else torch.device("cpu")
    model.to(device)

    return model, device


def create_trainer(trainer_kwargs_path: str):
    """Create trainer from config file."""
    assert os.path.exists(trainer_kwargs_path), f'{trainer_kwargs_path} does not exists'
    trainer_kwargs = json.load(open(trainer_kwargs_path))

    # Get loss weights from config (optional)
    loss_weights = trainer_kwargs.get("loss_weights", None)

    model, device = create_model(
        trainer_kwargs["model_kwargs"],
        loss_weights=loss_weights
    )

    trainer = Trainer(
        model=model,
        device=device,
        dataset_kwargs=trainer_kwargs["dataset_kwargs"],
        optimizer_kwargs=trainer_kwargs["optimizer_kwargs"],
        lr_scheduler_kwargs=trainer_kwargs["lr_scheduler_kwargs"],
        trainer_kwargs=trainer_kwargs["trainer_kwargs"]
    )

    return trainer, trainer_kwargs["trainer_kwargs"].get("checkpoint_path", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOP Panoptic Perception Model")
    parser.add_argument(
        "--config",
        type=str,
        default="panoptic_perception/configs/trainer/train_kwargs.json",
        help="Path to training config JSON file"
    )
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    trainer, checkpoint_path = create_trainer(args.config)

    trainer.train(checkpoint_path)