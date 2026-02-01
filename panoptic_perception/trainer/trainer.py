import os, math, time, copy
from tqdm import tqdm
from datetime import datetime

from typing import Iterable, Union
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

import numpy as np

from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor


class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Keeps a moving average of model parameters for more stable predictions.

    Usage:
        ema = ModelEMA(model, decay=0.9999)
        # During training:
        ema.update(model)
        # For evaluation:
        ema.apply_shadow(model)  # Apply EMA weights
        # ... evaluate ...
        ema.restore(model)       # Restore original weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 2000):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (higher = slower update, more smoothing)
            warmup_steps: Number of steps before reaching full decay rate
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0

        # Create shadow copy of model weights
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self) -> float:
        """Get current decay rate with warmup."""
        if self.step < self.warmup_steps:
            # Linear warmup from 0 to decay
            return min(self.decay, (1 + self.step) / (10 + self.step))
        return self.decay

    def update(self, model: nn.Module):
        """Update EMA weights with current model weights."""
        self.step += 1
        decay = self._get_decay()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA update: shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (backup original weights first)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'step': self.step,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.step = state_dict['step']
        self.decay = state_dict.get('decay', self.decay)


from panoptic_perception.dataset.enums import BDD100KClassesReduced

from panoptic_perception.models.models import YOLOP, get_model_param_groups
from panoptic_perception.models.utils import PanopticModelOutputs, WeightsManager

from panoptic_perception.utils.logger import Logger
from panoptic_perception.utils.wandb_logger import WandBLogger
from panoptic_perception.utils.detection_utils import DetectionHelper
from panoptic_perception.utils.evaluation_helper import DetectionMetrics, SegmentationMetrics
from terminaltables import AsciiTable


class Trainer:
    
    def __init__(self, model:Union[YOLOP], device:torch.device,
                 dataset_kwargs:dict, 
                 optimizer_kwargs:dict, lr_scheduler_kwargs:dict,
                 trainer_kwargs:dict):
        
        self.model = model
        self.device = device
        
        self.output_dir = trainer_kwargs["output_dir"]
        self.is_training = trainer_kwargs["is_training"]
        self.first_val_epoch = trainer_kwargs["first_val_epoch"]

        self.epochs = trainer_kwargs["epochs"]
        self.monitor_train = trainer_kwargs["monitor_train"]
        self.monitor_val = trainer_kwargs["monitor_val"]
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]
        
        self.checkpoint_idx = trainer_kwargs['checkpoint_idx']
        self.gradient_accumulation_steps = trainer_kwargs['gradient_accumulation_steps']
        self.reload_optimizer_with_initial_lr = trainer_kwargs["reload_optimizer_with_initial_lr"]
        self.lr_scheduler_start_epoch = trainer_kwargs["lr_scheduler_start_epoch"]
        self.reload_optimizer = trainer_kwargs["reload_optimizer"]

        # EMA settings
        self.use_ema = trainer_kwargs.get('use_ema', True)
        self.ema_decay = trainer_kwargs.get('ema_decay', 0.9999)
        self.ema_warmup_steps = trainer_kwargs.get('ema_warmup_steps', 2000)
        self.ema = None  # Initialized in train()

        # Best model tracking
        self.best_map = 0.0
        self.best_epoch = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.logger = Logger(
            f'{self.output_dir}/training_logger_{datetime.now()}', f'training_logger_{datetime.now()}'
        )

        # Initialize WandB logger
        wandb_config = {
            "epochs": self.epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            **trainer_kwargs,
            **optimizer_kwargs,
            **lr_scheduler_kwargs
        }
        self.wandb_logger = WandBLogger(
            project_name=trainer_kwargs.get("wandb_project", "yolop-panoptic"),
            run_name=trainer_kwargs.get("wandb_run_name", f"run_{datetime.now()}"),
            config=wandb_config,
            entity=trainer_kwargs.get("wandb_entity", None),
            tags=trainer_kwargs.get("wandb_tags", ["yolop", "panoptic"]),
            enabled=trainer_kwargs.get("wandb_enabled", True)
        )

        # Watch model if WandB is enabled
        if self.wandb_logger.enabled:
            self.wandb_logger.watch_model(self.model, log_freq=100)

        self._init_dataloader(dataset_kwargs)
        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 100
        
        self.logger.log_message(f'  Training on BDD100k Dataset   ')
        self.logger.log_message(f'Images Dir: {self.train_dataloader.dataset.images_dir}')        
        self.logger.log_message(f'Detections Annotations Dir: {self.train_dataloader.dataset.detection_annotations_dir}')
        self.logger.log_message(f'Segmentations Dir: {self.train_dataloader.dataset.segmentation_annotations_dir}')
        self.logger.log_message(f'Drivable Segmentations Dir: {self.train_dataloader.dataset.drivable_annotations_dir}')
        
        self.logger.log_new_line()
        
        self.logger.log_message(f'Number of Train Images: {self.train_dataloader.dataset.__len__()}')
        self.logger.log_message(f'Train Batch Size: {self.train_dataloader.batch_size}')
        
        self.logger.log_new_line()
        
        self.logger.log_message(f'Number of Val Images: {self.val_dataloader.dataset.__len__()}')
        self.logger.log_message(f'Val Batch Size: {self.val_dataloader.batch_size}')
                
        self.logger.log_line()

        self.logger.log_line()

        self._init_optimizer(optimizer_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'  Optimizer: {self.optimizer.__class__.__name__}  ')        
        self.logger.log_new_line()        
        
        if lr_scheduler_kwargs:
            self._init_lr_scheduler(lr_scheduler_kwargs)          
    
    def _init_dataloader(self, dataset_kwargs:dict):
        
        def create_dataloader(images_dir:str, detection_annotations_dir:dict, 
                            segmentation_annotations_dir:dict, drivable_annotations_dir:dict,
                            preprocessor_kwargs:dict, dataset_type:str, batch_size:int, 
                            perform_augmentation:bool=False, shuffle:bool=False, num_workers:int=1):
            
            dataset = BDD100KDataset({
                "images_dir":images_dir, 
                "detection_annotations_dir":detection_annotations_dir,
                "segmentation_annotations_dir":segmentation_annotations_dir,
                "drivable_annotations_dir":drivable_annotations_dir,
                "preprocessor_kwargs":preprocessor_kwargs
            }, dataset_type=dataset_type, 
            perform_augmentation=perform_augmentation)
            
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=BDDPreprocessor.collate_fn
            )                    
        
        self.train_dataloader = create_dataloader(
            dataset_kwargs["images_dir"],
            dataset_kwargs["detection_annotations_dir"],
            dataset_kwargs["segmentation_annotations_dir"],
            dataset_kwargs["drivable_annotations_dir"],
            dataset_kwargs["train_preprocessor_kwargs"],
            dataset_type="train",
            batch_size=dataset_kwargs["train_batch_size"],
            shuffle=dataset_kwargs.get("train_shuffle", True),
            num_workers=dataset_kwargs.get("train_num_workers", 1),
            perform_augmentation=dataset_kwargs.get("train_preprocessor_kwargs", False).get("perform_augmentation", False)
        )
        
        self.val_dataloader = create_dataloader(
            dataset_kwargs["images_dir"],
            dataset_kwargs["detection_annotations_dir"],
            dataset_kwargs["segmentation_annotations_dir"],
            dataset_kwargs["drivable_annotations_dir"],
            dataset_kwargs["val_preprocessor_kwargs"],
            dataset_type="val",
            batch_size=dataset_kwargs["val_batch_size"],
            shuffle=dataset_kwargs.get("val_shuffle", True),
            num_workers=dataset_kwargs.get("val_num_workers", 1)
        )                
    
    def _init_optimizer(self, optimizer_kwargs):
        
        self.warmup_bias_lr = optimizer_kwargs.get("warmup_bias_lr", 0.1)
        self.warmup_momentum = optimizer_kwargs.get("warmup_momentum", 0.8)
        self.warmup_epochs = optimizer_kwargs.get("warmup_epochs", 3)
        self.main_momentum = optimizer_kwargs.get("momentum", 0.937)
        
        self.groups = optimizer_kwargs.get("groups", {})

        if not self.groups:
            # No custom groups: train full model
            param_groups = list(self.model.parameters())
            self.logger.log_message('Full model training (all layers trainable)')
            self.logger.log_new_line()
        else:
            # Custom groups: get param groups
            param_groups = get_model_param_groups(self.model, self.groups)
            self.logger.log_message('Training with specified groups:')
            for group_name in self.groups:
                group_info = self.groups[group_name]
                self.logger.log_message(
                    f'  Group name: {group_name} - '
                    f'trainable: {group_info["trainable"]} - '
                    f'layer start/end: {group_info["group"]}'
                )
            self.logger.log_new_line()

        if optimizer_kwargs["_type"] == "SGD":
            self.lr0 = optimizer_kwargs.get("initial_lr", 3e-5)
            if param_groups:
                for pg in param_groups:
                    pg['lr'] = self.lr0 * pg.pop("lr_scale", 1.0)
                    
                self.optimizer = torch.optim.SGD(
                    param_groups,
                    momentum=optimizer_kwargs.get("momentum", 0.7)
                )                
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=optimizer_kwargs.get("initial_lr", 3e-3),
                    momentum=optimizer_kwargs.get("momentum", 0.7)
                )                

            self.inital_lr = optimizer_kwargs.get("initial_lr", 3e-5)
            
        elif optimizer_kwargs["_type"] == "AdamW":
            self.lr0 = optimizer_kwargs.get("initial_lr", 3e-4)
            
            if param_groups:
                for pg in param_groups:
                    pg['lr'] = self.lr0 * pg.pop("lr_scale", 1.0)
                    
                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    weight_decay=optimizer_kwargs.get("weight_decay", 0.01),
                    betas=(0.937, 0.999)
                )
                
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.lr0,
                    weight_decay=optimizer_kwargs.get("weight_decay", 0.01),
                    betas=(0.937, 0.999)
                )                
            
            self.inital_lr = optimizer_kwargs.get("initial_lr", 3e-5)
    
    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):

        if lr_scheduler_kwargs['_type'] == "linear":
            lr_scheduler_kwargs = lr_scheduler_kwargs['linear_lr_kwargs']            
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=lr_scheduler_kwargs['start_factor'],
                end_factor=lr_scheduler_kwargs['end_factor'],
                total_iters=self.epochs
            )

            self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
            self.logger.log_message(f'LR Scheduler Start Factor: {lr_scheduler_kwargs["start_factor"]}')
            self.logger.log_message(f'LR Scheduler End Factor: {lr_scheduler_kwargs["end_factor"]}')
            self.logger.log_new_line()            

        elif lr_scheduler_kwargs['_type'] == "cosine":
            lr_scheduler_kwargs = lr_scheduler_kwargs['cosine_annealing_lr_kwargs']            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=lr_scheduler_kwargs['eta_min']
            )

            self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
            self.logger.log_message(f'LR Scheduler TMax: {self.epochs}')
            self.logger.log_message(f'LR Scheduler ETA Min: {lr_scheduler_kwargs["eta_min"]}')            
            self.logger.log_new_line()
    
    def resume_from_ckpt(self, ckpt_path:str):
                
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            missing, unexpected, loaded_keys = WeightsManager().load(self.model, ckpt_path)            
            # self.model.load_state_dict(ckpt["model_state"])
            self.logger.log_message("=== Weights Loaded ===")
            self.logger.log_message(f"Loaded     : {len(loaded_keys)} keys")
            self.logger.log_message(f"Missing    : {len(missing)} keys")
            self.logger.log_message(f"Unexpected : {len(unexpected)} keys")            

            if self.reload_optimizer:
                if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
                    self.optimizer.load_state_dict(ckpt["optimizer_state"])
                    
                    if self.reload_optimizer_with_initial_lr:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.inital_lr                    

                if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
                    if hasattr(self, "lr_scheduler"):
                        self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])

                self.start_epoch = ckpt.get("epoch", 1)
                self.logger.log_message(f"Resuming from epoch {self.start_epoch}")
    
    def train(self, checkpoint_path:str=None):
        self.logger.log_line()
        
        tasks = []
        
        if self.model.detection_head_idx != -1:
            tasks.append("Detection, ")
        
        if self.model.segmentation_head_idx != -1:
            tasks.append("Drivable Segmentation, ")
            
        if self.model.lane_segmentation_head_idx != -1:
            tasks.append("Lane Segmentation")
            
        tasks = 'Tasks: '.join(tasks)
        
        self.logger.log_message(
            f'Training: Max Epoch - {self.epochs} -- {tasks} -- Device: {self.device}'
        )
        self.logger.log_new_line()

        self.total_training_time = 0.0
        self.cur_epoch = 0
        # self.best_score = 0.0
        self.best_score = defaultdict(float)

        # Reset best model tracking
        self.best_map = 0.0
        self.best_epoch = 0

        self.start_epoch = 1
        self.resume_from_ckpt(checkpoint_path)

        # Initialize EMA after loading checkpoint
        if self.use_ema:
            self.ema = ModelEMA(self.model, decay=self.ema_decay, warmup_steps=self.ema_warmup_steps)
            self.logger.log_message(f'EMA enabled with decay={self.ema_decay}, warmup_steps={self.ema_warmup_steps}')

            # If resuming, try to load EMA state
            if checkpoint_path and os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if 'ema_state' in ckpt and ckpt['ema_state'] is not None:
                    self.ema.load_state_dict(ckpt['ema_state'])
                    self.logger.log_message('EMA state loaded from checkpoint')
                if 'best_map' in ckpt:
                    self.best_map = ckpt['best_map']
                    self.best_epoch = ckpt.get('best_epoch', 0)
                    self.logger.log_message(f'Best mAP from checkpoint: {self.best_map:.4f} (epoch {self.best_epoch})')

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.cur_epoch = epoch
            self.logger.log_line()

            if self.monitor_train:
                self.train_one_epoch()

                if (self.cur_epoch + 1) % self.checkpoint_idx == 0:
                    ckpt_dir = f'{self.output_dir}/ckpt_{self.cur_epoch}'
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)

                    checkpoint = {
                        "epoch": self.cur_epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "scheduler_state": self.lr_scheduler.state_dict() if hasattr(self, "lr_scheduler") else None,
                        "best_map": self.best_map,
                        "best_epoch": self.best_epoch,
                    }

                    # Save EMA state
                    if self.use_ema and self.ema is not None:
                        checkpoint["ema_state"] = self.ema.state_dict()

                    torch.save(checkpoint, f"{ckpt_dir}/ckpt-{self.cur_epoch}.pt")

            if self.monitor_val and self.cur_epoch >= self.first_val_epoch:
                self.eval_one_epoch()

        # Finish WandB run
        self.logger.log_line()
        self.logger.log_message("Training completed!")
        self.logger.log_line()
        self.wandb_logger.finish()

    
    def train_one_epoch(self):
        
        self.model.train()

        total_loss = 0.0
        ten_percent_batch_total_loss = 0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0
        
        train_iter = tqdm(self.train_dataloader, desc=f'Training Epoch: {self.cur_epoch}')
        for batch_idx, data_items in enumerate(train_iter):
            
            step_begin_time = time.time()            
            loss, model_outputs = self.train_one_step(data_items)
            step_end_time = time.time()

            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx == self.train_dataloader.__len__() - 1):

                self.optimizer.step()
                self.optimizer.zero_grad()
                current_lr = self.optimizer.param_groups[0]['lr']

                # Update EMA weights after optimizer step
                if self.use_ema and self.ema is not None:
                    self.ema.update(self.model)

            total_loss += loss.item()
            ten_percent_batch_total_loss += loss.item()

            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)
                
            if (batch_idx + 1) % self.ten_percent_train_batch == 0:

                if self.cur_epoch < self.warmup_epochs:
                    warmup_factor = (self.cur_epoch + (batch_idx + 1)/self.total_train_batch) / self.warmup_epochs
                    warmup_factor = min(1.0, warmup_factor)

                    # LR warmup per param group (bias gets special warmup LR)
                    for pg in self.optimizer.param_groups:
                        if 'bias' in pg.get('name',''):
                            pg['lr'] = self.warmup_bias_lr + warmup_factor * (self.lr0 - self.warmup_bias_lr)
                        else:
                            pg['lr'] = warmup_factor * self.lr0

                    # Momentum warmup if optimizer supports momentum
                    if 'momentum' in self.optimizer.param_groups[0]:
                        self.optimizer.param_groups[0]['momentum'] = \
                            self.warmup_momentum + warmup_factor * (self.main_momentum - self.warmup_momentum)

                average_loss = ten_percent_batch_total_loss/self.ten_percent_train_batch
                average_time = ten_percent_training_time/self.ten_percent_train_batch

                message = f'Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - total loss {average_loss:.4f} -- current_lr: {current_lr}'
                self.logger.log_message(message=message)

                # Log to WandB
                self.wandb_logger.log_metrics({
                    "train/loss_10pct": average_loss,
                    "train/lr": current_lr,
                    "train/avg_step_time": average_time
                }, step=self.cur_epoch * self.total_train_batch + batch_idx)

                ten_percent_batch_total_loss = 0
                ten_percent_training_time = 0.0
                ten_percent_metric_per_grid = defaultdict(lambda:defaultdict(int))
                
        avg_epoch_loss = total_loss / self.total_train_batch

        self.logger.log_message(
            f'Epoch {self.cur_epoch} - Average Loss {avg_epoch_loss:.4f} -- current_lr: {current_lr}'
        )

        # Step the learning rate scheduler at the end of epoch
        if hasattr(self, 'lr_scheduler'):
            if self.lr_scheduler_start_epoch != -1 and self.cur_epoch > self.lr_scheduler_start_epoch:
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

        # Log epoch-level metrics to WandB
        self.wandb_logger.log_metrics({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_time": epoch_training_time,
            "train/epoch": self.cur_epoch
        }, step=self.cur_epoch)
                
    
    def train_one_step(self, data_items:dict) -> PanopticModelOutputs:
        
        for k, v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(self.device)
                
        outputs = self.model(
            data_items["images"],
            targets={
                "drivable_area_seg": data_items.get("drivable_area_seg"),
                "lane_seg": data_items.get("segmentation_masks"),
                "detections": data_items["detections"]
            }            
        )        
        
        loss = torch.zeros(1, device=self.device)
        
        if outputs.detection_loss is not None:
            loss += outputs.detection_loss
            
        if outputs.drivable_segmentation_loss is not None:
            loss += outputs.drivable_segmentation_loss
            
        if outputs.lane_segmentation_loss is not None:
            loss += outputs.lane_segmentation_loss

        loss.backward()

        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
        return loss, outputs

    def _compute_confusion_matrix(self, preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Compute confusion matrix for a batch of predictions and targets."""
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Filter valid indices (in case of ignore labels)
        mask = (targets_flat >= 0) & (targets_flat < num_classes)
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]

        # Compute confusion matrix using bincount
        indices = targets_flat * num_classes + preds_flat
        conf_matrix = torch.bincount(indices, minlength=num_classes * num_classes)
        return conf_matrix.reshape(num_classes, num_classes)

    def _compute_metrics_from_confusion_matrix(self, conf_matrix: torch.Tensor, num_classes: int) -> tuple:
        """Compute IoU and Dice metrics from confusion matrix."""
        iou_dict = {}
        dice_dict = {}

        iou_per_class = []
        dice_per_class = []

        for cls in range(num_classes):
            tp = conf_matrix[cls, cls].float()
            fp = conf_matrix[:, cls].sum().float() - tp
            fn = conf_matrix[cls, :].sum().float() - tp

            # IoU = TP / (TP + FP + FN)
            iou = tp / (tp + fp + fn + 1e-10)
            iou_dict[f'IoU_class_{cls}'] = iou.item()
            iou_per_class.append(iou.item())

            # Dice = 2*TP / (2*TP + FP + FN)
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-10)
            dice_dict[f'Dice_class_{cls}'] = dice.item()
            dice_per_class.append(dice.item())

        iou_dict['mIoU'] = np.mean(iou_per_class)
        dice_dict['mDice'] = np.mean(dice_per_class)

        return iou_dict, dice_dict

    def eval_one_epoch(self):
        """Evaluate model on validation set."""

        # Apply EMA weights for evaluation
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
            self.logger.log_message('Using EMA weights for evaluation')

        self.model.eval()
        import gc

        # Collect detection predictions (smaller memory footprint)
        # all_detections = []
        # all_detection_targets = []
        
        dets_by_image = defaultdict(None) #image_id -> (num_dets, 6)
        gt_by_image = defaultdict(None) #image_id -> (num_gt, 5)

        # Use confusion matrices for segmentation (memory efficient)
        num_drivable_classes = 3
        num_lane_classes = 2  # Will be updated dynamically if needed
        drivable_confusion_matrix = None
        lane_confusion_matrix = None

        total_val_loss = 0.0
        total_det_loss = 0.0
        total_drivable_loss = 0.0
        total_lane_loss = 0.0
        global_image_idx = 0

        self.logger.log_line()
        self.logger.log_message(f'Evaluating Epoch {self.cur_epoch}')

        val_iter = tqdm(self.val_dataloader, desc=f'Validation Epoch: {self.cur_epoch}')

        with torch.no_grad():
            for batch_idx, data_items in enumerate(val_iter):
                # Move data to device
                for k, v in data_items.items():
                    if torch.is_tensor(v):
                        data_items[k] = v.to(self.device)

                # Forward pass
                outputs = self.model(
                    data_items["images"],
                    targets={
                        "drivable_area_seg": data_items.get("drivable_area_seg"),
                        "lane_seg": data_items.get("segmentation_masks"),
                        "detections": data_items["detections"]
                    }
                )

                # Accumulate losses
                if outputs.detection_loss is not None:
                    total_det_loss += outputs.detection_loss.item()
                    total_val_loss += outputs.detection_loss.item()

                if outputs.drivable_segmentation_loss is not None:
                    total_drivable_loss += outputs.drivable_segmentation_loss.item()
                    total_val_loss += outputs.drivable_segmentation_loss.item()

                if outputs.lane_segmentation_loss is not None:
                    total_lane_loss += outputs.lane_segmentation_loss.item()
                    total_val_loss += outputs.lane_segmentation_loss.item()

                # Process detection predictions - concatenate layer outputs and apply NMS
                if outputs.detection_predictions is not None:
                    detection_preds = outputs.detection_predictions
                    batch_size, _, image_h, image_w = data_items["images"].shape

                    # Concatenate predictions from all detection layers
                    batch_predictions = []
                    for layer_pred in detection_preds:
                        b, na, h, w, nc = layer_pred.shape
                        layer_pred_flat = layer_pred.view(b, na * h * w, nc)
                        batch_predictions.append(layer_pred_flat)

                    concatenated_preds = torch.cat(batch_predictions, dim=1)

                    # Apply NMS
                    nms_results = DetectionHelper.non_max_suppression(
                        concatenated_preds,
                        conf_threshold=0.25,
                        iou_threshold=0.45,
                        max_detections=500
                    )

                    for image_idx, dets in enumerate(nms_results):
                        if dets is not None:
                            dets_by_image[global_image_idx] = dets
                        else:
                            dets_by_image[global_image_idx] = None
                        
                        mask = data_items["detections"][:, 0] == image_idx
                        img_targets = data_items["detections"][mask]
                        
                        if img_targets.shape[0] > 0:
                            boxes_xywh = img_targets[:, 2:6]
                            boxes_xywh[:, [0,2]] *= image_w
                            boxes_xywh[:, [1,3]] *= image_h
                            boxes_xyxy = DetectionHelper.xywh2xyxy(boxes_xywh)
                            classes = img_targets[:, 1:2]
                            gts = torch.cat([boxes_xyxy, classes], dim=1)
                            gt_by_image[global_image_idx] = gts
                        else:
                            gt_by_image[global_image_idx] = None

                        global_image_idx += 1

                # Process segmentation with running confusion matrix (memory efficient)
                if outputs.drivable_segmentation_predictions is not None:
                    drivable_preds = torch.argmax(outputs.drivable_segmentation_predictions, dim=1)
                    if data_items.get("drivable_area_seg") is not None:
                        drivable_targets = data_items["drivable_area_seg"]
                        # Initialize confusion matrix on first batch
                        if drivable_confusion_matrix is None:
                            drivable_confusion_matrix = torch.zeros(
                                num_drivable_classes, num_drivable_classes, dtype=torch.int64
                            )
                        # Update confusion matrix
                        drivable_confusion_matrix += self._compute_confusion_matrix(
                            drivable_preds.cpu(), drivable_targets.cpu(), num_drivable_classes
                        )

                if outputs.lane_segmentation_predictions is not None:
                    lane_preds = torch.argmax(outputs.lane_segmentation_predictions, dim=1)
                    if data_items.get("segmentation_masks") is not None:
                        lane_targets = data_items["segmentation_masks"]
                        # Dynamically determine number of lane classes
                        max_class = max(lane_preds.max().item(), lane_targets.max().item()) + 1
                        if max_class > num_lane_classes:
                            # Expand confusion matrix if needed
                            if lane_confusion_matrix is not None:
                                old_matrix = lane_confusion_matrix
                                lane_confusion_matrix = torch.zeros(
                                    max_class, max_class, dtype=torch.int64
                                )
                                lane_confusion_matrix[:old_matrix.shape[0], :old_matrix.shape[1]] = old_matrix
                            num_lane_classes = max_class
                        # Initialize confusion matrix on first batch
                        if lane_confusion_matrix is None:
                            lane_confusion_matrix = torch.zeros(
                                num_lane_classes, num_lane_classes, dtype=torch.int64
                            )
                        # Update confusion matrix
                        lane_confusion_matrix += self._compute_confusion_matrix(
                            lane_preds.cpu(), lane_targets.cpu(), num_lane_classes
                        )

        # Compute metrics
        num_batches = len(self.val_dataloader)

        # Average losses
        avg_det_loss = total_det_loss / num_batches if num_batches > 0 else 0.0
        avg_drivable_loss = total_drivable_loss / num_batches if num_batches > 0 else 0.0
        avg_lane_loss = total_lane_loss / num_batches if num_batches > 0 else 0.0
        avg_total_loss = (total_det_loss + total_drivable_loss + total_lane_loss) / num_batches

        # Detection metrics
        detection_metrics = {}
        num_classes = len(BDD100KClassesReduced)
        
        stats_iou_threshold=0.25
        ap_results, stats_per_class = DetectionMetrics.compute_stats(
            dets_by_image,
            gt_by_image,
            iou_threshold=stats_iou_threshold,
            num_classes=num_classes
        )

        detection_metrics = ap_results

        # Create AP table for logging with class names
        ap_table_data = [["Class", "AP"]]
        for cls in range(num_classes):
            class_name = BDD100KClassesReduced(cls).name
            ap_value = ap_results.get(f'AP_class_{cls}', 0.0)
            ap_table_data.append([f"{cls}: {class_name}", f"{ap_value:.4f}"])
        ap_table_data.append(["mAP@0.5", f"{ap_results['mAP']:.4f}"])

        ap_table_string = AsciiTable(ap_table_data).table
        self.logger.log_message("\nDetection Metrics (AP@0.5):")
        self.logger.log_message(ap_table_string)
        
        #Create Stats (TP, FP, FN)
        stats_table_data = [["Class", "total GT", f"TP", f"FP", f"FN"]]
        for cls in range(num_classes):
            class_name = BDD100KClassesReduced(cls).name            
            class_stats = stats_per_class[cls]        

            total_gt = class_stats.get("total_gt", 0.0)
            true_positives = class_stats.get("true_positives", 0.0)
            false_positives = class_stats.get("false_positives", 0.0)
            false_negatives = class_stats.get("false_negatives", 0.0)
            
            stats_table_data.append([f'{cls}: {class_name}', total_gt, 
                                    true_positives, false_positives, false_negatives])
            
        stats_table_string = AsciiTable(stats_table_data).table
        self.logger.log_message(f"\nDetection Metrics @{stats_iou_threshold}:")
        self.logger.log_message(stats_table_string)

        # Log AP table to WandB
        wandb_ap_data = [[f"{cls}: {BDD100KClassesReduced(cls).name}", ap_results.get(f'AP_class_{cls}', 0.0)] for cls in range(num_classes)]
        wandb_ap_data.append(["mAP@0.5", ap_results['mAP']])
        self.wandb_logger.log_table(
            "val/detection_ap",
            columns=["Class", "AP"],
            data=wandb_ap_data,
            step=self.cur_epoch
        )

        # Free detection memory before computing segmentation metrics
        del dets_by_image, gt_by_image
        gc.collect()
        torch.cuda.empty_cache()

        # Drivable segmentation metrics (computed from confusion matrix)
        drivable_metrics = {}
        if drivable_confusion_matrix is not None:
            drivable_iou, drivable_dice = self._compute_metrics_from_confusion_matrix(
                drivable_confusion_matrix, num_drivable_classes
            )
            drivable_metrics = {**drivable_iou, **drivable_dice}

            # Create drivable metrics table
            drivable_table_data = [["Metric", "Value"]]
            drivable_table_data.append(["mIoU", f"{drivable_iou['mIoU']:.4f}"])
            drivable_table_data.append(["mDice", f"{drivable_dice['mDice']:.4f}"])
            for cls in range(num_drivable_classes):
                drivable_table_data.append([f"IoU Class {cls}", f"{drivable_iou.get(f'IoU_class_{cls}', 0.0):.4f}"])

            drivable_table_string = AsciiTable(drivable_table_data).table

            self.logger.log_message("\nDrivable Segmentation Metrics:")
            self.logger.log_message(drivable_table_string)

            # Log to WandB
            wandb_drivable_data = [
                ["mIoU", drivable_iou['mIoU']],
                ["mDice", drivable_dice['mDice']]
            ]
            for cls in range(num_drivable_classes):
                wandb_drivable_data.append([f"IoU_class_{cls}", drivable_iou.get(f'IoU_class_{cls}', 0.0)])

            self.wandb_logger.log_table(
                "val/drivable_metrics",
                columns=["Metric", "Value"],
                data=wandb_drivable_data,
                step=self.cur_epoch
            )

        # Lane segmentation metrics (computed from confusion matrix)
        lane_metrics = {}
        if lane_confusion_matrix is not None:
            lane_iou, lane_dice = self._compute_metrics_from_confusion_matrix(
                lane_confusion_matrix, num_lane_classes
            )
            lane_metrics = {**lane_iou, **lane_dice}

            self.logger.log_message(f"\nLane Segmentation mIoU: {lane_iou['mIoU']:.4f}, mDice: {lane_dice['mDice']:.4f}")

        # Log all metrics to WandB
        wandb_metrics = {
            "val/total_loss": avg_total_loss,
            "val/detection_loss": avg_det_loss,
            "val/drivable_loss": avg_drivable_loss,
            "val/lane_loss": avg_lane_loss,
            "val/epoch": self.cur_epoch
        }

        # Add detection metrics
        if detection_metrics:
            wandb_metrics["val/mAP"] = detection_metrics["mAP"]
            for cls in range(num_classes):
                wandb_metrics[f"val/AP_class_{cls}"] = detection_metrics.get(f'AP_class_{cls}', 0.0)

        # Add drivable metrics
        if drivable_metrics:
            wandb_metrics["val/drivable_mIoU"] = drivable_metrics["mIoU"]
            wandb_metrics["val/drivable_mDice"] = drivable_metrics["mDice"]

        # Add lane metrics
        if lane_metrics:
            wandb_metrics["val/lane_mIoU"] = lane_metrics["mIoU"]
            wandb_metrics["val/lane_mDice"] = lane_metrics["mDice"]

        self.wandb_logger.log_metrics(wandb_metrics, step=self.cur_epoch)

        # Log summary
        self.logger.log_line()
        self.logger.log_message(
            f'Validation Epoch {self.cur_epoch} - Avg Loss: {avg_total_loss:.4f} | '
            f'Det Loss: {avg_det_loss:.4f} | Drivable Loss: {avg_drivable_loss:.4f}'
        )
        if detection_metrics:
            self.logger.log_message(f'  mAP@0.5: {detection_metrics["mAP"]:.4f}')
        if drivable_metrics:
            self.logger.log_message(f'  Drivable mIoU: {drivable_metrics["mIoU"]:.4f}')

        # Save best model checkpoint
        current_map = detection_metrics.get("mAP", 0.0) if detection_metrics else 0.0
        if current_map > self.best_map:
            self.best_map = current_map
            self.best_epoch = self.cur_epoch
            self._save_best_checkpoint(current_map, detection_metrics, drivable_metrics)

        self.logger.log_message(f'  Best mAP: {self.best_map:.4f} (epoch {self.best_epoch})')
        self.logger.log_line()

        # Restore original weights after EMA evaluation
        if self.use_ema and self.ema is not None:
            self.ema.restore(self.model)

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        self.model.train()

    def _save_best_checkpoint(self, current_map: float, detection_metrics: dict,
                               drivable_metrics: dict = None):
        """Save the best model checkpoint."""
        best_ckpt_path = f'{self.output_dir}/best_model.pt'

        checkpoint = {
            "epoch": self.cur_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.lr_scheduler.state_dict() if hasattr(self, "lr_scheduler") else None,
            "best_map": current_map,
            "best_epoch": self.cur_epoch,
            "detection_metrics": detection_metrics,
            "drivable_metrics": drivable_metrics,
        }

        # Save EMA state if using EMA
        if self.use_ema and self.ema is not None:
            checkpoint["ema_state"] = self.ema.state_dict()
            # Also save EMA weights as separate key (for easy inference loading)
            checkpoint["ema_model_state"] = {k: v.clone() for k, v in self.ema.shadow.items()}

        torch.save(checkpoint, best_ckpt_path)

        self.logger.log_message(f'New best model saved! mAP: {current_map:.4f} -> {best_ckpt_path}')

        # Log to WandB
        self.wandb_logger.log_metrics({
            "val/best_map": current_map,
            "val/best_epoch": self.cur_epoch
        }, step=self.cur_epoch)
