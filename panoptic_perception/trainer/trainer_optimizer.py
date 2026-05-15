import torch

from dataclasses import dataclass
from typing import Any

from .trainer_args import TrainingArgument

@dataclass
class OptimizerContext:
    param_groups : Any
    training_args : TrainingArgument

def _get_adamw_handler(ctx: OptimizerContext):
    args = ctx.training_args
    return torch.optim.AdamW, {
        "lr":args.initial_lr,
        "weight_decay":args.lr_decay if hasattr(args, "weight_decay") else 0.0,
        "betas": args.betas if hasattr(args, "betas") else (0.937, 0.999)
    }
    
def _get_sgd_handler(ctx:OptimizerContext):
    args = ctx.training_args
    return torch.optim.SGD, {
        "lr":args.initial_lr,
        "momentum":args.momentum,
        "weight_decay":args.weight_decay
    }
    
def _get_adagrad_handler(ctx:OptimizerContext):
    args = ctx.training_args
    return torch.optim.Adagrad, {
        "lr":args.initial_lr,
        "lr_decay": args.lr_decay if hasattr(args, "lr_decay") else 0.0,
        "weight_decay": args.lr_decay if hasattr(args, "weight_decay") else 0.0,
        "eps": args.lr_decay if hasattr(args, "weight_decay") else 0.0
    }
    
def build_optmizer(ctx:OptimizerContext):

    if not hasattr(ctx.training_args, "optim_type"):
        raise AttributeError("Attribute optim_type not found")

    optim_type = ctx.training_args.optim_type.lower()
    if optim_type not in OPTIMIZER_HANDLER:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")
    
    if not hasattr(ctx.training_args, "initial_lr"):
        raise AttributeError("Attribute initial_lr not found")
    
    if ctx.training_args.initial_lr is None:
        raise ValueError(f"Unsupported initial_lr: {ctx.training_args.initial_lr}")
    
    optim_class, kwargs = OPTIMIZER_HANDLER[optim_type](ctx)
    return optim_class(ctx.param_groups, **kwargs)
    
OPTIMIZER_HANDLER = {
    "adamw":_get_adamw_handler,
    "sgd":_get_sgd_handler,
    "adagrad":_get_adagrad_handler
}