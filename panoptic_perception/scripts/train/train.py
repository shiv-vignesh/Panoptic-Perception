import json
import os

import torch

from panoptic_perception.models.models import YOLOP
from panoptic_perception.trainer.trainer import Trainer

def create_model(model_kwargs:dict):
    
    cfg_path = model_kwargs["cfg_path"]
    device = model_kwargs["device"]
    
    assert os.path.exists(cfg_path), f'{cfg_path} does not exists'
    model = YOLOP(cfg_path)
    
    device = torch.device("cuda") if torch.cuda.is_available() and device == "cuda" else torch.device("cpu")
    
    return model, device

def create_trainer(trainer_kwargs_path:str):
    
    assert os.path.exists(trainer_kwargs_path), f'{trainer_kwargs_path} does not exists'            
    trainer_kwargs = json.load(open(trainer_kwargs_path))
    
    model, device = create_model(
        trainer_kwargs["model_kwargs"]
    )
    
    trainer = Trainer(
        model=model,
        device=device,
        dataset_kwargs=trainer_kwargs["dataset_kwargs"],
        optimizer_kwargs=trainer_kwargs["optimizer_kwargs"],
        lr_scheduler_kwargs=trainer_kwargs["lr_scheduler_kwargs"],
        trainer_kwargs=trainer_kwargs["trainer_kwargs"]
    )
    
    return trainer

if __name__ == "__main__":
    
    trainer = create_trainer(
        "panoptic_perception/configs/trainer/train_kwargs.json"
    )
    
    trainer.train()