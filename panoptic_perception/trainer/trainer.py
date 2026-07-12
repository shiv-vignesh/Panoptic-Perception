import time
from typing import Union, Optional

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from panoptic_perception.models.models import BaseTaskModel, BaseEnhancementModel
from panoptic_perception.models.types import PanopticModelOutputs

from panoptic_perception.utils.logger import Logger
from panoptic_perception.utils.wandb_logger import WandBLogger

from panoptic_perception.trainer.trainer_optimizer import build_optmizer, OptimizerContext
from panoptic_perception.trainer.trainer_schedulers import build_scheduler, SchedulerContext
from panoptic_perception.trainer.trainer_args import TrainingArgument
from panoptic_perception.trainer.callbacks import Callbacks
from panoptic_perception.trainer.utils import EvalMetrics, EvalBatchContext


class Trainer:

    def __init__(self, model: Optional[Union[BaseTaskModel, BaseEnhancementModel]],
                 train_dataloader: DataLoader = None,
                 val_dataloaders: dict = None,
                 optimizer: torch.optim = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 training_args: TrainingArgument = None,
                 wandb_logger: WandBLogger = None,
                 logger: Logger = None,
                 checkpoint_path: str = None):

        if training_args is None:
            training_args = TrainingArgument(output_dir="tmp_trainer")

        self.training_args = training_args

        if model is None:
            raise ValueError("Trainer: requires a model")

        self.model = model
        self.has_enhancement = isinstance(model, BaseEnhancementModel)
        self.device = next(model.parameters()).device

        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self._create_optimizer_and_scheduler()

        self.eval_metrics = {
            metric_prefix: EvalMetrics(metric_prefix=metric_prefix)
            for metric_prefix in (val_dataloaders or {})
        }

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
        ctx = OptimizerContext(param_groups, self.training_args)
        self.optimizer = build_optmizer(ctx)

    def _create_scheduler(self):
        ctx = SchedulerContext(
            self.optimizer, self.training_args,
            total_epochs=self.training_args.epochs,
        )
        self.lr_scheduler = build_scheduler(ctx)

    # ---- outer loop --------------------------------------------------------

    def train(self):
        if self.train_dataloader is None and self.training_args.monitor_train:
            raise ValueError("Trainer: training requires a train_dataset.")

        if hasattr(self.model, "get_active_tasks"):
            tasks = self.model.get_active_tasks()
            self.logger.log_message(
                f'Training: Max Epoch - {self.training_args.epochs} -- {tasks} -- Device: {self.device}'
            )
        else:
            self.logger.log_message(
                f'Training: Max Epoch - {self.training_args.epochs} -- Device: {self.device}'
            )

        self.logger.log_new_line()
        self.callbacks.on_train_begin(self)

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

    # ---- task-agnostic epoch scaffold --------------------------------------

    def _train_one_epoch(self):
        self.model.train()

        self._init_train_window()
        total_loss = 0.0
        epoch_training_time = 0.0

        self.train_batch_idx = 0
        self.batch_images = None
        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = max(1, self.total_train_batch // 100)

        current_lr = self.optimizer.param_groups[0]['lr']

        train_iter = tqdm(self.train_dataloader, desc=f'Training Epoch: {self.cur_epoch}')
        for batch_idx, data_items in enumerate(train_iter):
            self.train_batch_idx = batch_idx
            self._apply_warmup()

            step_begin = time.time()
            loss, model_outputs = self._train_one_step(data_items)
            step_time = time.time() - step_begin

            self.batch_images = data_items.get("images")

            if ((batch_idx + 1) % self.training_args.gradient_accumulation_steps == 0) or (batch_idx == self.total_train_batch - 1):
                if self.training_args.gradient_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_args.gradient_clipping,
                    )
                    if not torch.isfinite(grad_norm):
                        self.optimizer.zero_grad()
                        continue
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            epoch_training_time += step_time

            self._accumulate_train_iter(loss, model_outputs, step_time)

            if (batch_idx + 1) % self.ten_percent_train_batch == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self._log_train_window(current_lr, batch_idx)
                self._init_train_window()

            self.callbacks.on_step_end(self)

        avg_epoch_loss = total_loss / self.total_train_batch

        if hasattr(self, 'lr_scheduler'):
            if self.training_args.lr_scheduler_start_epoch != -1 and \
                    self.cur_epoch > self.training_args.lr_scheduler_start_epoch:
                self.lr_scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

        self._log_train_epoch(avg_epoch_loss, current_lr, epoch_training_time)

    def _eval_one_epoch(self, dataloader=None, metric_prefix="val"):
        if not dataloader or len(dataloader) == 0:
            return

        self.model.eval()
        self.logger.log_line()
        self.logger.log_message(f'[{metric_prefix}] Evaluating Epoch {self.cur_epoch}')

        val_iter = tqdm(dataloader, desc=f'[{metric_prefix}] Epoch: {self.cur_epoch}')

        self.eval_batch_idx = 0
        self.eval_metric_prefix = metric_prefix
        self.total_eval_batch = len(dataloader)
        self.eval_batch_ctx = EvalBatchContext()

        self._init_eval_state(metric_prefix)

        for batch_idx, data_items in enumerate(val_iter):
            for k, v in data_items.items():
                if torch.is_tensor(v):
                    data_items[k] = v.to(self.device)

            try:
                with torch.no_grad():
                    outputs = self._forward_model(data_items)
            except ValueError as e:
                if "model produced no outputs" not in str(e):
                    raise
                self._log_eval_skip(batch_idx, metric_prefix, data_items)
                continue

            self.eval_batch_idx = batch_idx
            self.eval_batch_ctx.cur_eval_model_outputs = outputs

            self._populate_eval_batch_ctx(data_items, outputs)
            self._accumulate_eval_iter(outputs, data_items)

            self.callbacks.on_eval_batch_end(self)

        self.callbacks.on_eval_end(self)
        self._log_eval_epoch(metric_prefix)

    def _apply_warmup(self):
        if self.cur_epoch < self.training_args.warmup_epochs:
            warmup_factor = (self.cur_epoch + (self.train_batch_idx + 1) / self.total_train_batch) / self.training_args.warmup_epochs
            warmup_factor = min(1.0, warmup_factor)

            for pg in self.optimizer.param_groups:
                if pg['lr'] == 0.0:
                    continue
                scale = pg.get('lr_scale', 1.0)
                if 'bias' in pg.get('name', ''):
                    pg['lr'] = (self.training_args.warmup_bias_lr + warmup_factor * (self.training_args.initial_lr - self.training_args.warmup_bias_lr)) * scale
                else:
                    pg['lr'] = warmup_factor * self.training_args.initial_lr * scale

            if 'momentum' in self.optimizer.param_groups[0]:
                self.optimizer.param_groups[0]['momentum'] = \
                    self.training_args.warmup_momentum + warmup_factor * (self.training_args.main_momentum - self.training_args.warmup_momentum)

    # ---- forward / loss surface (subclass extends) -------------------------
    def _forward_model(self, data_items: dict):
        raise NotImplementedError("Subclass must implement _forward_model")

    def _build_targets(self, data_items: dict) -> dict:
        return {}

    def _train_one_step(self, data_items: dict):
        # Overrides for multi-task loss summing.
        for k, v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(self.device)

        outputs = self._forward_model(data_items)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError(
                f"{type(outputs).__name__}.loss is None; base _train_one_step "
                f"requires a scalar `.loss` on model outputs. Override in subclass "
                f"or ensure the model computes loss when targets are provided."
            )
        loss.backward()
        return loss, outputs

    # ---- train hooks (default = generic loss + timing) ---------------------
    def _init_train_window(self):
        self._window_loss = 0.0
        self._window_step_time = 0.0
        self._window_batches = 0

    def _accumulate_train_iter(self, loss, model_outputs, step_time):
        self._window_loss += loss.item()
        self._window_step_time += step_time
        self._window_batches += 1

    def _log_train_iter(self, batch_idx, loss, model_outputs, current_lr):
        pass

    def _log_train_window(self, current_lr, batch_idx):
        n = self._window_batches or 1
        avg_loss = self._window_loss / n
        avg_time = self._window_step_time / n
        self.logger.log_message(
            f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} "
            f"- total {avg_loss:.4f} -- lr: {current_lr}"
        )
        self.wandb_logger.log_metrics({
            "train/loss_10pct": avg_loss,
            "train/lr": current_lr,
            "train/avg_step_time": avg_time,
        }, step=self.cur_epoch * self.total_train_batch + batch_idx)

    def _log_train_epoch(self, avg_epoch_loss, current_lr, epoch_time):
        self.logger.log_message(
            f'Epoch {self.cur_epoch} - Average Loss {avg_epoch_loss:.4f} -- current_lr: {current_lr}'
        )
        self.wandb_logger.log_metrics({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_time": epoch_time,
            "train/epoch": self.cur_epoch,
        }, step=self.cur_epoch)

    # ---- eval hooks (Override) --------------------------------------
    def _init_eval_state(self, prefix):
        pass

    def _accumulate_eval_iter(self, outputs, data_items):
        pass

    def _log_eval_epoch(self, prefix):
        pass

    def _populate_eval_batch_ctx(self, data_items, outputs):
        # Base populates only the image-shape fields common to any task.
        images = data_items.get("images")
        if images is not None and images.dim() == 4:
            _, _, image_h, image_w = images.shape
            self.eval_batch_ctx.cur_eval_image_h = image_h
            self.eval_batch_ctx.cur_eval_image_w = image_w
        self.eval_batch_ctx.cur_eval_images = images
        self.eval_batch_ctx.cur_eval_image_paths = data_items.get("image_paths", [])

    def _log_eval_skip(self, batch_idx, metric_prefix, data_items):
        paths = data_items.get("image_paths", [])
        self.logger.log_message(
            f"[skip-batch] eval iter {batch_idx} ({metric_prefix}): "
            f"empty loss_items — skipping. paths={paths}"
        )


class PanopticTrainer(Trainer):

    def _build_targets(self, data_items: dict) -> dict:
        return {
            "drivable_area_seg": data_items.get("drivable_area_seg"),
            "lane_seg": data_items.get("segmentation_masks"),
            "detections": data_items["detections"],
            "lanes_detections": data_items.get("lanes_detections"),
            "lane_seg_masks": data_items.get("lane_seg_masks"),
            "clean_images": data_items.get("clean_images"),
        }

    def _forward_model(self, data_items: dict) -> PanopticModelOutputs:
        return self.model(
            data_items["images"],
            targets=self._build_targets(data_items),
        )

    def _train_one_step(self, data_items: dict):
        for k, v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(self.device)

        try:
            outputs = self._forward_model(data_items)
        except ValueError as e:
            if "model produced no outputs" not in str(e):
                raise
            paths = data_items.get("image_paths", [])
            present = {
                k: (None if data_items.get(k) is None
                    else (tuple(data_items[k].shape) if torch.is_tensor(data_items[k])
                          else "non-tensor"))
                for k in ("detections", "drivable_area_seg", "segmentation_masks",
                          "lanes_detections", "lane_seg_masks")
            }
            self.logger.log_message(
                f"[skip-batch] iter {getattr(self, 'train_batch_idx', '?')}: "
                f"empty loss_items — skipping. paths={paths} targets={present}"
            )
            return torch.zeros(1, device=self.device), None

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

        loss.backward()
        return loss, outputs

    def _init_train_window(self):
        super()._init_train_window()
        self._window_det = 0.0
        self._window_drv = 0.0
        self._window_lane_seg = 0.0
        self._window_lane_det = 0.0
        self._window_lane_items = {}

    def _accumulate_train_iter(self, loss, model_outputs, step_time):
        super()._accumulate_train_iter(loss, model_outputs, step_time)
        if model_outputs is None:
            return
        if model_outputs.detection_loss is not None:
            self._window_det += model_outputs.detection_loss.item()
        if model_outputs.drivable_segmentation_loss is not None:
            self._window_drv += model_outputs.drivable_segmentation_loss.item()
        if model_outputs.lane_segmentation_loss is not None:
            self._window_lane_seg += model_outputs.lane_segmentation_loss.item()
        if model_outputs.lane_detection_loss is not None:
            self._window_lane_det += model_outputs.lane_detection_loss.item()
        if model_outputs.lane_detection_loss_items is not None:
            for k, v in model_outputs.lane_detection_loss_items.items():
                self._window_lane_items[k] = self._window_lane_items.get(k, 0.0) + v

    def _log_train_window(self, current_lr, batch_idx):
        n = self._window_batches or 1
        average_loss = self._window_loss / n
        average_time = self._window_step_time / n
        avg_det = self._window_det / n
        avg_drv = self._window_drv / n
        avg_lane_seg = self._window_lane_seg / n
        avg_lane_det = self._window_lane_det / n

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
        self.logger.log_message(
            f'Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - {loss_str} -- lr: {current_lr}'
        )

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
        for k, v in self._window_lane_items.items():
            wandb_metrics[f"train/{k}"] = v / n

        self.wandb_logger.log_metrics(
            wandb_metrics,
            step=self.cur_epoch * self.total_train_batch + batch_idx,
        )

    def _populate_eval_batch_ctx(self, data_items, outputs):
        super()._populate_eval_batch_ctx(data_items, outputs)
        self.eval_batch_ctx.cur_eval_gt_detections = data_items["detections"]
        self.eval_batch_ctx.cur_eval_gt_drivable_area_seg = data_items.get("drivable_area_seg")
        self.eval_batch_ctx.cur_eval_gt_lane_seg = data_items.get("segmentation_masks")
        self.eval_batch_ctx.cur_eval_gt_lane_detections = data_items.get("lanes_detections")

    def _log_eval_skip(self, batch_idx, metric_prefix, data_items):
        paths = data_items.get("image_paths", [])
        present = {
            k: (None if data_items.get(k) is None
                else (tuple(data_items[k].shape) if torch.is_tensor(data_items[k])
                      else "non-tensor"))
            for k in ("detections", "drivable_area_seg", "segmentation_masks",
                      "lanes_detections", "lane_seg_masks")
        }
        self.logger.log_message(
            f"[skip-batch] eval iter {batch_idx} ({metric_prefix}): "
            f"empty loss_items — skipping. paths={paths} targets={present}"
        )
