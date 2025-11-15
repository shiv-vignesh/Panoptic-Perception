import os, math, time
from tqdm import tqdm
from datetime import datetime

from typing import Iterable, Union
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np

from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor

from panoptic_perception.models.models import YOLOP
from panoptic_perception.models.utils import PanopticModelOutputs

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
            run_name=trainer_kwargs.get("wandb_run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
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
                            shuffle:bool=False, num_workers:int=1):
            
            dataset = BDD100KDataset({
                "images_dir":images_dir, 
                "detection_annotations_dir":detection_annotations_dir,
                "segmentation_annotations_dir":segmentation_annotations_dir,
                "drivable_annotations_dir":drivable_annotations_dir,
                "preprocessor_kwargs":preprocessor_kwargs
            }, dataset_type=dataset_type)
            
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
            num_workers=dataset_kwargs.get("train_num_workers", 1)
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
        
        if optimizer_kwargs["_type"] == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_kwargs.get("initial_lr", 3e-5),
                momentum=optimizer_kwargs.get("momentum", 0.7)
            )
            
        elif optimizer_kwargs["_type"] == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_kwargs.get("initial_lr", 3e-4),
                weight_decay=optimizer_kwargs.get("weight_decay", 0.01)
            )                        
    
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
    
    def train(self):
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

        for epoch in range(1, self.epochs + 1):
            self.cur_epoch = epoch
            self.logger.log_line()

            if self.monitor_train:
                self.train_one_epoch()

                if (self.cur_epoch + 1) % self.checkpoint_idx == 0:
                    ckpt_dir = f'{self.output_dir}/ckpt_{self.cur_epoch}'
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)

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
                
            total_loss += loss.item()
            ten_percent_batch_total_loss += loss.item()

            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)
                
            if (batch_idx + 1) % self.ten_percent_train_batch == 0:
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

        # Step the learning rate scheduler at the end of epoch
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

        self.logger.log_message(
            f'Epoch {self.cur_epoch} - Average Loss {avg_epoch_loss:.4f} -- current_lr: {current_lr}'
        )

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
        
        loss = torch.tensor(0.0, device=self.device)
        
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
    
    def eval_one_epoch(self):
        """Evaluate model on validation set."""
        self.model.eval()

        # Collect all predictions and ground truths
        all_detections = []
        all_detection_targets = []
        all_drivable_preds = []
        all_drivable_targets = []
        all_lane_preds = []
        all_lane_targets = []

        total_val_loss = 0.0
        total_det_loss = 0.0
        total_drivable_loss = 0.0
        total_lane_loss = 0.0

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
                    batch_size = data_items["images"].shape[0]

                    # Concatenate predictions from all detection layers
                    all_predictions = []
                    for layer_pred in detection_preds:
                        b, na, h, w, nc = layer_pred.shape
                        layer_pred_flat = layer_pred.view(b, na * h * w, nc)
                        all_predictions.append(layer_pred_flat)

                    concatenated_preds = torch.cat(all_predictions, dim=1)

                    # Apply NMS
                    nms_results = DetectionHelper.non_max_suppression(
                        concatenated_preds,
                        conf_threshold=0.25,
                        iou_threshold=0.45,
                        max_detections=100
                    )
                    all_detections.extend(nms_results)

                    # Process ground truth detections
                    for i in range(batch_size):
                        mask = data_items["detections"][:, 0] == i
                        img_targets = data_items["detections"][mask]

                        if img_targets.shape[0] > 0:
                            boxes_xywh = img_targets[:, 2:6]
                            boxes_xyxy = DetectionHelper.xywh2xyxy(boxes_xywh)
                            classes = img_targets[:, 1:2]
                            gts = torch.cat([boxes_xyxy, classes], dim=1)
                            all_detection_targets.append(gts)
                        else:
                            all_detection_targets.append(None)

                # Process segmentation predictions
                if outputs.drivable_segmentation_predictions is not None:
                    drivable_preds = torch.argmax(outputs.drivable_segmentation_predictions, dim=1)
                    all_drivable_preds.append(drivable_preds.cpu())
                    if data_items.get("drivable_masks") is not None:
                        all_drivable_targets.append(data_items["drivable_masks"].cpu())

                if outputs.lane_segmentation_predictions is not None:
                    lane_preds = torch.argmax(outputs.lane_segmentation_predictions, dim=1)
                    all_lane_preds.append(lane_preds.cpu())
                    if data_items.get("segmentation_masks") is not None:
                        all_lane_targets.append(data_items["segmentation_masks"].cpu())

        # Compute metrics
        num_batches = len(self.val_dataloader)

        # Average losses
        avg_det_loss = total_det_loss / num_batches if num_batches > 0 else 0.0
        avg_drivable_loss = total_drivable_loss / num_batches if num_batches > 0 else 0.0
        avg_lane_loss = total_lane_loss / num_batches if num_batches > 0 else 0.0
        avg_total_loss = (total_det_loss + total_drivable_loss + total_lane_loss) / num_batches

        # Detection metrics
        detection_metrics = {}
        if len(all_detections) > 0 and len(all_detection_targets) > 0:
            ap_results = DetectionMetrics.calculate_ap(
                all_detections,
                all_detection_targets,
                iou_threshold=0.5,
                num_classes=10
            )
            detection_metrics = ap_results

            # Create AP table for logging
            ap_table_data = [["Class", "AP"]]
            for cls in range(10):
                ap_table_data.append([f"Class {cls}", f"{ap_results.get(f'AP_class_{cls}', 0.0):.4f}"])
            ap_table_data.append(["mAP@0.5", f"{ap_results['mAP']:.4f}"])

            ap_table_string = AsciiTable(ap_table_data).table

            self.logger.log_message("\nDetection Metrics (AP@0.5):")
            self.logger.log_message(ap_table_string)

            # Log AP table to WandB
            wandb_ap_data = [[f"Class {cls}", ap_results.get(f'AP_class_{cls}', 0.0)] for cls in range(10)]
            wandb_ap_data.append(["mAP@0.5", ap_results['mAP']])
            self.wandb_logger.log_table(
                "val/detection_ap",
                columns=["Class", "AP"],
                data=wandb_ap_data,
                step=self.cur_epoch
            )

        # Drivable segmentation metrics
        drivable_metrics = {}
        if len(all_drivable_preds) > 0 and len(all_drivable_targets) > 0:
            drivable_preds_cat = torch.cat(all_drivable_preds, dim=0)
            drivable_targets_cat = torch.cat(all_drivable_targets, dim=0)

            drivable_iou = SegmentationMetrics.compute_iou(
                drivable_preds_cat, drivable_targets_cat, num_classes=3
            )
            drivable_dice = SegmentationMetrics.compute_dice(
                drivable_preds_cat, drivable_targets_cat, num_classes=3
            )
            drivable_metrics = {**drivable_iou, **drivable_dice}

            # Create drivable metrics table
            drivable_table_data = [["Metric", "Value"]]
            drivable_table_data.append(["mIoU", f"{drivable_iou['mIoU']:.4f}"])
            drivable_table_data.append(["mDice", f"{drivable_dice['mDice']:.4f}"])
            for cls in range(3):
                drivable_table_data.append([f"IoU Class {cls}", f"{drivable_iou.get(f'IoU_class_{cls}', 0.0):.4f}"])

            drivable_table_string = AsciiTable(drivable_table_data).table

            self.logger.log_message("\nDrivable Segmentation Metrics:")
            self.logger.log_message(drivable_table_string)

            # Log to WandB
            wandb_drivable_data = [
                ["mIoU", drivable_iou['mIoU']],
                ["mDice", drivable_dice['mDice']]
            ]
            for cls in range(3):
                wandb_drivable_data.append([f"IoU_class_{cls}", drivable_iou.get(f'IoU_class_{cls}', 0.0)])

            self.wandb_logger.log_table(
                "val/drivable_metrics",
                columns=["Metric", "Value"],
                data=wandb_drivable_data,
                step=self.cur_epoch
            )

        # Lane segmentation metrics
        lane_metrics = {}
        if len(all_lane_preds) > 0 and len(all_lane_targets) > 0:
            lane_preds_cat = torch.cat(all_lane_preds, dim=0)
            lane_targets_cat = torch.cat(all_lane_targets, dim=0)

            # Determine number of lane classes dynamically
            num_lane_classes = int(lane_preds_cat.max().item()) + 1

            lane_iou = SegmentationMetrics.compute_iou(
                lane_preds_cat, lane_targets_cat, num_classes=num_lane_classes
            )
            lane_dice = SegmentationMetrics.compute_dice(
                lane_preds_cat, lane_targets_cat, num_classes=num_lane_classes
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
            for cls in range(10):
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
        self.logger.log_line()

        self.model.train()
        