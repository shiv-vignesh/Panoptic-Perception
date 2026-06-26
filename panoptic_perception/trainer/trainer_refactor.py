
import torch
from torch.utils.data.dataloader import DataLoader

import os
import time
from typing import Iterable, Union, Optional, List
from collections import defaultdict

from tqdm import tqdm

from panoptic_perception.models.models import BaseTaskModel, BaseEnhancementModel
from panoptic_perception.models.types import PanopticModelOutputs

from panoptic_perception.utils.logger import Logger
from panoptic_perception.utils.wandb_logger import WandBLogger

from panoptic_perception.trainer.trainer_optimizer import build_optmizer, OptimizerContext
from panoptic_perception.trainer.trainer_schedulers import build_scheduler, SchedulerContext
from panoptic_perception.trainer.trainer_args import TrainingArgument
from panoptic_perception.trainer.callbacks import Callbacks
from panoptic_perception.trainer.utils import listify, to_numpy, EvalMetrics, EvalBatchContext

class Trainer:
    
    def __init__(self, model:Optional[Union[BaseTaskModel, BaseEnhancementModel]],
                train_dataloader:DataLoader=None,
                val_dataloaders:dict[str, DataLoader]=None, 
                optimizer:torch.optim=None,
                lr_scheduler:torch.optim.lr_scheduler=None,
                training_args:TrainingArgument = None,
                wandb_logger:WandBLogger = None,
                logger:Logger = None,
                checkpoint_path:str = None):

        if training_args is None:
            training_args = TrainingArgument(output_dir="tmp_trainer")

        self.training_args = training_args

        if model is None:
            raise ValueError(f"Trainer: requires a model ")

        self.model = model
        self.has_enhancement = isinstance(model, BaseEnhancementModel)
        self.device = next(model.parameters()).device

        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self._create_optimizer_and_scheduler()

        self.eval_metrics = {metric_prefix : EvalMetrics(metric_prefix=metric_prefix) 
                                for metric_prefix in (val_dataloaders or {})}
        
        self.callbacks = Callbacks()
        self.checkpoint_path = checkpoint_path

        if logger is None:
            raise ValueError("Trainer: requires a logger")
        if wandb_logger is None:
            raise ValueError("Trainer: requires a wandb_logger")

        self.logger = logger        
        self.wandb_logger = wandb_logger

        self.cur_epoch = 0

    def _create_optimizer_and_scheduler(self):
        
        if self.optimizer is None:
            self._create_optimizer()
            
        if self.lr_scheduler is None:
            self._create_scheduler()
    
    def _create_optimizer(self):
        param_groups = self.model.get_param_groups()
        ctx = OptimizerContext(
            param_groups, self.training_args
        )
        
        self.optimizer = build_optmizer(ctx)
        
    def _create_scheduler(self):
        
        ctx = SchedulerContext(
            self.optimizer, 
            self.training_args,
            total_epochs=self.training_args.epochs
        )
        
        self.lr_scheduler = build_scheduler(ctx)
        
    def train(self):

        if self.train_dataloader is None and self.training_args.monitor_train:
            raise ValueError("Trainer: training requires a train_dataset.")

        tasks = self.model.get_active_tasks()        
        self.logger.log_message(
            f'Training: Max Epoch - {self.training_args.epochs} -- {tasks} -- Device: {self.device}'
        )
        self.logger.log_new_line()
        self.callbacks.on_train_begin(self)

        if self.training_args.eval_before_train and self.val_dataloaders and self.training_args.monitor_val:
            self.logger.log_line()
            self.logger.log_message("=== Pre-Training Evaluation ===")
            for prefix, dataloader in self.val_dataloaders.items():
                self.eval_metrics[prefix].reset()
                self._eval_one_epoch(dataloader, prefix)

        self.start_epoch = self.cur_epoch
        for epoch in range(self.cur_epoch, self.training_args.epochs + 1):
            self.cur_epoch = epoch
            self.logger.log_line()

            if self.training_args.monitor_train:
                self.callbacks.on_epoch_begin(self)
                self._train_one_epoch()                

            if self.val_dataloaders and self.training_args.monitor_val:
                if self.cur_epoch >= self.training_args.first_val_epoch:                
                    for prefix, dataloader in self.val_dataloaders.items():
                        self.eval_metrics[prefix].reset()
                        self._eval_one_epoch(dataloader, prefix)

            self.callbacks.on_epoch_end(self)

    def _train_one_epoch(self):

        self.model.train()

        total_loss = 0.0
        ten_percent_batch_total_loss = 0

        ten_percent_det_loss = 0.0
        ten_percent_drivable_loss = 0.0
        ten_percent_lane_seg_loss = 0.0
        ten_percent_lane_det_loss = 0.0
        ten_percent_lane_det_items = {}

        epoch_training_time = 0.0
        ten_percent_training_time = 0.0
        
        self.train_batch_idx = 0 #used by callbacks
        self.batch_images = None
        
        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = max(1, self.total_train_batch // 100)

        current_lr = self.optimizer.param_groups[0]['lr']

        train_iter = tqdm(self.train_dataloader, desc=f'Training Epoch: {self.cur_epoch}')
        for batch_idx, data_items in enumerate(train_iter):

            self.train_batch_idx = batch_idx
            self._apply_warmup()
            current_lr = self.optimizer.param_groups[0]['lr']

            step_begin_time = time.time()
            loss, model_outputs = self._train_one_step(data_items)
            step_end_time = time.time()

            self.batch_images = data_items["images"]

            if ((batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0) or (batch_idx == self.total_train_batch - 1):
                if self.training_args.gradient_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_args.gradient_clipping
                    )

                    if not torch.isfinite(grad_norm):
                        self.optimizer.zero_grad()
                        continue

                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            ten_percent_batch_total_loss += loss.item()

            if model_outputs.detection_loss is not None:
                ten_percent_det_loss += model_outputs.detection_loss.item()
            if model_outputs.drivable_segmentation_loss is not None:
                ten_percent_drivable_loss += model_outputs.drivable_segmentation_loss.item()
            if model_outputs.lane_segmentation_loss is not None:
                ten_percent_lane_seg_loss += model_outputs.lane_segmentation_loss.item()
            if model_outputs.lane_detection_loss is not None:
                ten_percent_lane_det_loss += model_outputs.lane_detection_loss.item()
            if model_outputs.lane_detection_loss_items is not None:
                for k, v in model_outputs.lane_detection_loss_items.items():
                    ten_percent_lane_det_items[k] = ten_percent_lane_det_items.get(k, 0.0) + v

            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)

            if (batch_idx + 1) % self.ten_percent_train_batch == 0:
                n = self.ten_percent_train_batch
                average_loss = ten_percent_batch_total_loss / n
                average_time = ten_percent_training_time / n
                avg_det = ten_percent_det_loss / n
                avg_drv = ten_percent_drivable_loss / n
                avg_lane_seg = ten_percent_lane_seg_loss / n
                avg_lane_det = ten_percent_lane_det_loss / n

                parts = [f'total {average_loss:.4f}']
                if avg_det > 0:
                    parts.append(f'det {avg_det:.4f}')
                if avg_drv > 0:
                    parts.append(f'drv {avg_drv:.4f}')
                if avg_lane_seg > 0:
                    parts.append(f'lane_seg {avg_lane_seg:.4f}')
                if avg_lane_det > 0:
                    parts.append(f'lane_det {avg_lane_det:.4f}')

                loss_str = ' | '.join(parts)
                message = f'Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - {loss_str} -- lr: {current_lr}'
                self.logger.log_message(message=message)

                # Log to WandB
                wandb_metrics = {
                    "train/loss_10pct": average_loss,
                    "train/lr": current_lr,
                    "train/avg_step_time": average_time,
                }
                if avg_det > 0:
                    wandb_metrics["train/det_loss"] = avg_det
                if avg_drv > 0:
                    wandb_metrics["train/drivable_loss"] = avg_drv
                if avg_lane_seg > 0:
                    wandb_metrics["train/lane_seg_loss"] = avg_lane_seg
                if avg_lane_det > 0:
                    wandb_metrics["train/lane_det_loss"] = avg_lane_det
                if ten_percent_lane_det_items:
                    for k, v in ten_percent_lane_det_items.items():
                        wandb_metrics[f"train/{k}"] = v / n

                self.wandb_logger.log_metrics(wandb_metrics, step=self.cur_epoch * self.total_train_batch + batch_idx)

                ten_percent_batch_total_loss = 0
                ten_percent_det_loss = 0.0
                ten_percent_drivable_loss = 0.0
                ten_percent_lane_seg_loss = 0.0
                ten_percent_lane_det_loss = 0.0
                ten_percent_lane_det_items = {}
                ten_percent_training_time = 0.0
                
            self.callbacks.on_step_end(self)

        avg_epoch_loss = total_loss / self.total_train_batch

        self.logger.log_message(
            f'Epoch {self.cur_epoch} - Average Loss {avg_epoch_loss:.4f} -- current_lr: {current_lr}'
        )

        # Step the learning rate scheduler at the end of epoch
        if hasattr(self, 'lr_scheduler'):
            if self.training_args.lr_scheduler_start_epoch != -1 and \
                    self.cur_epoch > self.training_args.lr_scheduler_start_epoch:

                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

        # Log epoch-level metrics to WandB
        self.wandb_logger.log_metrics({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_time": epoch_training_time,
            "train/epoch": self.cur_epoch
        }, step=self.cur_epoch)
            
    def _train_one_step(self, data_items:dict) -> PanopticModelOutputs:
        
        for k, v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(self.device)
                
        outputs = self.model(
            data_items["images"],
            targets={
                "drivable_area_seg": data_items.get("drivable_area_seg"),
                "lane_seg": data_items.get("segmentation_masks"),
                "detections": data_items["detections"],
                "lanes_detections": data_items.get("lanes_detections"),
                "lane_seg_masks": data_items.get("lane_seg_masks"),
                "clean_images": data_items.get("clean_images")
            }
        )

        loss = torch.zeros(1, device=self.device)

        if outputs.detection_loss is not None:
            loss += outputs.detection_loss

        if outputs.drivable_segmentation_loss is not None:
            loss += outputs.drivable_segmentation_loss

        if outputs.lane_segmentation_loss is not None:
            loss += outputs.lane_segmentation_loss

        if outputs.lane_detection_loss is not None:
            loss += outputs.lane_detection_loss

        if self.has_enhancement:
            if hasattr(outputs, "defogging_loss") and outputs.defogging_loss is not None:
                loss += self.training_args.lambda_defog * outputs.defogging_loss
                
        accumulation_steps = max(1, self.training_args.gradient_accumulation_steps)
        (loss / accumulation_steps).backward()

        return loss, outputs
    
    def _eval_one_epoch(self, dataloader=None, metric_prefix="val"):
        
        if not dataloader or len(dataloader) == 0:
            return
        
        self.model.eval()
        
        self.logger.log_line()
        self.logger.log_message(f'[{metric_prefix}] Evaluating Epoch {self.cur_epoch}')

        val_iter = tqdm(dataloader, desc=f'[{metric_prefix}] Epoch: {self.cur_epoch}')
        
        self.eval_batch_idx = 0 #used by callbacks
        self.eval_metric_prefix = metric_prefix
        
        self.total_eval_batch = len(dataloader)
        self.eval_batch_ctx = EvalBatchContext()

        for batch_idx, data_items in enumerate(val_iter):
            # Move data to device
            for k, v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    data_items["images"],
                    targets={
                        "drivable_area_seg": data_items.get("drivable_area_seg"),
                        "lane_seg": data_items.get("segmentation_masks"),
                        "detections": data_items["detections"],
                        "lanes_detections": data_items.get("lanes_detections"),
                        "lane_seg_masks": data_items.get("lane_seg_masks"),
                        "clean_images": data_items.get("clean_images")
                    }
                )

            self.eval_batch_idx = batch_idx
            self.eval_batch_ctx.cur_eval_model_outputs = outputs

            _, _, image_h, image_w = data_items["images"].shape
            self.eval_batch_ctx.cur_eval_image_h = image_h
            self.eval_batch_ctx.cur_eval_image_w = image_w

            self.eval_batch_ctx.cur_eval_images = data_items["images"]
            self.eval_batch_ctx.cur_eval_image_paths = data_items.get("image_paths", [])
            self.eval_batch_ctx.cur_eval_gt_detections = data_items["detections"]
            self.eval_batch_ctx.cur_eval_gt_drivable_area_seg = data_items.get("drivable_area_seg")
            self.eval_batch_ctx.cur_eval_gt_lane_seg = data_items.get("segmentation_masks")
            self.eval_batch_ctx.cur_eval_gt_lane_detections = data_items.get("lanes_detections")

            self.callbacks.on_eval_batch_end(self)

        self.callbacks.on_eval_end(self)

    def _apply_warmup(self):

        if self.cur_epoch < self.training_args.warmup_epochs:
            warmup_factor = (self.cur_epoch + (self.train_batch_idx + 1)/self.total_train_batch) / self.training_args.warmup_epochs
            warmup_factor = min(1.0, warmup_factor)

            # LR warmup per param group (bias gets special warmup LR)
            # Skip groups frozen by staged training (lr == 0)
            for pg in self.optimizer.param_groups:
                if pg['lr'] == 0.0:
                    continue
                scale = pg.get('lr_scale', 1.0)
                if 'bias' in pg.get('name',''):
                    pg['lr'] = (self.training_args.warmup_bias_lr + warmup_factor * (self.training_args.initial_lr - self.training_args.warmup_bias_lr)) * scale
                else:
                    pg['lr'] = warmup_factor * self.training_args.initial_lr * scale

            # Momentum warmup if optimizer supports momentum
            if 'momentum' in self.optimizer.param_groups[0]:
                self.optimizer.param_groups[0]['momentum'] = \
                    self.training_args.warmup_momentum + warmup_factor * (self.training_args.main_momentum - self.training_args.warmup_momentum)
