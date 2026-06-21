import os
import torch
from dataclasses import dataclass, field, fields
from datetime import datetime

@dataclass
class TrainingArgument:

    @staticmethod
    def get_train_datetime():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    # ---- Optimizer Args ----
    
    optim_type : str = "AdamW"
    initial_lr : float = 1e-3
    weight_decay : float = 0.01
    lr_decay : float = 0.0
    eps : float = 1e-10
    momentum : float = 0.937
    warmup_bias_lr : float = 0.1
    warmup_momentum : float = 0.8
    warmup_epochs : int = 3
    betas : tuple = (0.937, 0.999)
    main_momentum : float = 0.937
    
    forbidden_layers : list = field(default_factory=lambda : [torch.nn.BatchNorm2d])
    
    # ---- Learning Rate Scheduler ----
    
    lr_type : str = "cosine"
    
    linear_lr_start : float = 1.0
    linear_lr_end : float = 0.01

    cosine_annealing_eta_min : float = 1e-6

    steps_per_epoch: int = None  # optional (for step-based schedulers)

    # ----- training Args -----

    output_dir : str = "runs"
    logs_dir : str = field(
                    default_factory=lambda:os.path.join(TrainingArgument.output_dir, 
                                f"train_{TrainingArgument.get_train_datetime()}")
                    )

    epochs : int = 100
    gradient_clipping : int = 1.0
    checkpoint_idx : int = 2
    gradient_accumulation_steps : int = 4
    lr_scheduler_start_epoch : int = 1

    monitor_train : bool = True
    monitor_val : bool = True
    first_val_epoch : int = 0

    wandb_enabled : bool = True
    reload_optimizer : bool = False

    eval_visualize_outputs : bool = True

    # ----- Smoke / debug iter caps -----
    # When set (int > 0), _train_one_epoch and _eval_one_epoch break after this
    # many batches. None disables the cap (production default).
    max_train_iters : int = None
    max_eval_iters  : int = None

    # When set, log per-iter loss every N steps (useful for smoke runs where the
    # default ten-percent log interval = total_train_batch//100 is too sparse).
    # None disables — falls back to existing ten-percent logging.
    log_every_n_iters : int = None

    # ----- Base -----
    lambda_defog : float = 0.2
    
    @classmethod
    def from_config(cls, config:dict):
      valid_fields = {f.name for f in fields(TrainingArgument)}
      filtered = {k: v for k, v in config.items() if k in valid_fields}                                                             
      return cls(**filtered)