import torch

from dataclasses import dataclass
from typing import Any

from .trainer_args import TrainingArgument

@dataclass
class SchedulerContext:
    optimizer: torch.optim.Optimizer
    training_args: TrainingArgument
    total_epochs: int


def step_handler(ctx: SchedulerContext):
    args = ctx.training_args

    return torch.optim.lr_scheduler.StepLR, {
        "step_size": getattr(args, "step_size", 30),
        "gamma": getattr(args, "gamma", 0.1),
    }
    
def multistep_handler(ctx: SchedulerContext):
    args = ctx.training_args

    return torch.optim.lr_scheduler.MultiStepLR, {
        "milestones": getattr(args, "milestones", [30, 60, 90]),
        "gamma": getattr(args, "gamma", 0.1),
    }    
    
def exponential_handler(ctx: SchedulerContext):
    args = ctx.training_args

    return torch.optim.lr_scheduler.ExponentialLR, {
        "gamma": getattr(args, "gamma", 0.95),
    }    

def onecycle_handler(ctx: SchedulerContext):
    args = ctx.training_args
    
    if not hasattr(ctx.training_args, "steps_per_epoch") or ctx.training_args.steps_per_epoch is None:
        raise ValueError("OneCycleLR requires steps_per_epoch")

    total_steps = ctx.training_args.steps_per_epoch * ctx.total_epochs

    return torch.optim.lr_scheduler.OneCycleLR, {
        "max_lr": getattr(args, "max_lr", args.initial_lr),
        "total_steps": total_steps,
        "pct_start": getattr(args, "pct_start", 0.3),
        "anneal_strategy": "cos",
    }

def cosine_restart_handler(ctx: SchedulerContext):
    args = ctx.training_args

    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, {
        "T_0": getattr(args, "T_0", 10),
        "T_mult": getattr(args, "T_mult", 2),
        "eta_min": args.cosine_annealing_eta_min,
    }
    
def cosine_handler(ctx: SchedulerContext):
    args = ctx.training_args

    return torch.optim.lr_scheduler.CosineAnnealingLR, {
        "T_max": ctx.total_epochs,
        "eta_min": args.cosine_annealing_eta_min,
    }    
    
def linear_handler(ctx: SchedulerContext):
    args = ctx.training_args

    start = args.linear_lr_start
    end = args.linear_lr_end
    total_epochs = ctx.total_epochs

    def lr_lambda(epoch):
        return start + (end - start) * (epoch / total_epochs)

    return torch.optim.lr_scheduler.LambdaLR, {
        "lr_lambda": lr_lambda
    }
    
def build_scheduler(ctx:SchedulerContext):
    
    if not hasattr(ctx.training_args, "lr_type"):
        raise AttributeError("Attribute lr_type not found")
    
    lr_type = ctx.training_args.lr_type.lower()
    if lr_type not in LR_SCHEDULER_HANDLER:
        raise ValueError(f"Unsupported lr scheduler type: {lr_type}")
    
    if ctx.optimizer is None:
        raise ValueError(f"Optimizer cannot be: {ctx.optimizer}")
    
    sched_class, kwargs = LR_SCHEDULER_HANDLER[lr_type](ctx)
    return sched_class(ctx.optimizer, **kwargs)
    

LR_SCHEDULER_HANDLER = {
    "cosine": cosine_handler,
    "linear": linear_handler,
    "step": step_handler,
    "multistep": multistep_handler,
    "exponential": exponential_handler,
    "onecycle": onecycle_handler,
    "cosine_restart": cosine_restart_handler,
}