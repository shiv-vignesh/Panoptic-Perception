from __future__ import annotations

import os
import torch

from typing import Any, Iterable, Dict, TYPE_CHECKING
from collections import defaultdict
from terminaltables import AsciiTable

if TYPE_CHECKING:
    from panoptic_perception.trainer.trainer_refactor import Trainer

from panoptic_perception.dataset.enums import BDD100KClassesReduced
from panoptic_perception.models.models import BaseTaskModel, BaseEnhancementModel

from panoptic_perception.models.utils import WeightsManager
from panoptic_perception.utils.detection_utils import DetectionHelper
from panoptic_perception.utils.segmentation_utils import SegmentationUtils
from panoptic_perception.utils.evaluation_helper import DetectionMetrics
from panoptic_perception.utils.lane_utils import lane_nms, predictions_to_lanes, _make_activated_gt, polyline_iou

from panoptic_perception.trainer.utils import listify

CLASS_NAMES = [cls.name for cls in BDD100KClassesReduced]

class TrainerCallback:
    def on_train_begin(self, trainer:Trainer): pass
    def on_train_end(self, trainer:Trainer): pass
    def on_epoch_begin(self, trainer:Trainer): pass
    def on_epoch_end(self, trainer:Trainer): pass
    def on_step_begin(self, trainer:Trainer): pass
    def on_step_end(self, trainer:Trainer): pass
    def on_eval_begin(self, trainer:Trainer): pass
    def on_eval_batch_end(self, trainer:Trainer): pass
    def on_eval_end(self, trainer:Trainer): pass
    
class Callbacks(TrainerCallback):
    """Class that combines multiple callbacks into one. For internal use only"""

    def __init__(self, callbacks=None):
        self.callbacks : Iterable[TrainerCallback] = listify(callbacks)

    def add_callback(self, callback:TrainerCallback):
        self.callbacks.append(callback)

    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer)

    def on_epoch_end(self, trainer):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def on_step_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_step_begin(trainer)

    def on_step_end(self, trainer):
        for callback in self.callbacks:
            callback.on_step_end(trainer)

    def on_eval_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_eval_begin(trainer)

    def on_eval_batch_end(self, trainer):
        for callback in self.callbacks:
            callback.on_eval_batch_end(trainer)

    def on_eval_end(self, trainer):
        for callback in self.callbacks:
            callback.on_eval_end(trainer)
    
class CheckpointCallback(TrainerCallback):
    
    def __init__(self):

        #prefix : metric/best-epoch
        self.best_map = defaultdict(float)
        self.best_epoch = defaultdict(int)
    
    def on_train_begin(self, trainer):
        self.resume_from_checkpoint(trainer)

    def resume_from_checkpoint(self, trainer:Trainer):

        if not trainer.checkpoint_path:
            return
        
        if os.path.exists(trainer.checkpoint_path):
            try:
                checkpoint = torch.load(trainer.checkpoint_path, map_location=trainer.device, weights_only=False)
            except Exception as e:
                trainer.logger.log_message(f'Error loading checkpoint: {trainer.checkpoint_path} - {e}')
                return
        else:
            return

        key_prefix = None
        if isinstance(trainer.model, BaseEnhancementModel):
            ckpt_state = checkpoint.get("model_state", checkpoint)
            sample_key = next(iter(ckpt_state), "")
            if not sample_key.startswith("task_network."):
                key_prefix = "task_network"
                
        missing, unexpected, loaded_keys = WeightsManager().load(trainer.model, 
                                                                trainer.checkpoint_path, key_prefix=key_prefix)

        trainer.logger.log_message("=== Weights Loaded ===")
        trainer.logger.log_message(f"Loaded     : {len(loaded_keys)} keys")
        trainer.logger.log_message(f"Missing    : {len(missing)} keys")
        trainer.logger.log_message(f"Unexpected : {len(unexpected)} keys")
        trainer.logger.log_new_line()

        if "epoch" in checkpoint:        
            trainer.cur_epoch = checkpoint["epoch"] + 1

        if trainer.training_args.reload_optimizer:
            reloaded_optim = False
            if "optimizer_state" not in checkpoint or checkpoint["optimizer_state"] is None:
                trainer.logger.log_message(f'Key - optimizer_state missing from checkpoint or None. \
                                           \nStarting Fresh')
                trainer.logger.log_new_line()

            elif "optimizer_type" not in checkpoint and checkpoint["optimizer_state"] is not None:
                try:
                    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
                    reloaded_optim = True
                except (ValueError, KeyError) as e:
                    trainer.logger.log_message(f'Could not load optimizer_state: {e} \
                                            \nStarting Fresh')
                    trainer.logger.log_new_line()

            else:
                ckpt_optim_type = checkpoint.get("optimizer_type")
                current_optim_type = trainer.optimizer.__class__.__name__
                
                if ckpt_optim_type and ckpt_optim_type != current_optim_type:
                    trainer.logger.log_message(f"Optimizer mismatch: checkpoint={ckpt_optim_type},  \
                                            current={current_optim_type}. Skipping optimizer state reload.")
                    trainer.logger.log_new_line()

                else:
                    trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
                    reloaded_optim = True
                    
            if reloaded_optim:
                if "scheduler_state" not in checkpoint or checkpoint["scheduler_state"] is None:
                    trainer.logger.log_message(f'Key - scheduler_state missing from checkpoint or None. \
                                            \nStarting Fresh')
                    trainer.logger.log_new_line()

                elif "scheduler_type" not in checkpoint and checkpoint["scheduler_state"] is not None:
                    if hasattr(trainer, "lr_scheduler"):
                        try:
                            trainer.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
                        except (ValueError, KeyError) as e:
                            trainer.logger.log_message(f'Could not load scheduler_state: {e} \
                                                    \nStarting Fresh')
                            trainer.logger.log_new_line()

                else:
                    ckpt_scheduler_type = checkpoint.get("scheduler_type")
                    current_scheduler_type = trainer.lr_scheduler.__class__.__name__
                    
                    if ckpt_scheduler_type and ckpt_scheduler_type != current_scheduler_type:
                        trainer.logger.log_message(f"Scheduler mismatch: checkpoint={ckpt_scheduler_type},  \
                                                current={current_scheduler_type}. Skipping optimizer state reload.")
                        trainer.logger.log_new_line()                             
                    else:
                        trainer.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
                        
    def on_epoch_end(self, trainer):

        self._save_checkpoint(trainer)
        self._save_best_model(trainer)
    
    def _save_checkpoint(self, trainer:Trainer):

        if (trainer.cur_epoch + 1) % trainer.training_args.checkpoint_idx == 0:
            ckpt_dir = f"{trainer.training_args.output_dir}/checkpoints/ckpt_{trainer.cur_epoch}"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            checkpoint = {
                "epoch": trainer.cur_epoch,
                "model_state": trainer.model.state_dict(),
                "optimizer_type":trainer.optimizer.__class__.__name__,
                "optimizer_state": trainer.optimizer.state_dict(),
                "scheduler_type": trainer.lr_scheduler.__class__.__name__,
                "scheduler_state": trainer.lr_scheduler.state_dict() if hasattr(trainer, "lr_scheduler") else None
            }

            torch.save(checkpoint, f"{ckpt_dir}/ckpt-{trainer.cur_epoch}.pt")
    
    def _save_best_model(self, trainer:Trainer):

        output_dir = f"{trainer.training_args.output_dir}/best_models"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for metric_prefix, eval_metrics in trainer.eval_metrics.items():
            if eval_metrics.ap_per_class is None:
                continue

            current_map = eval_metrics.ap_per_class.get("mAP", 0.0)
            if current_map > self.best_map[metric_prefix]:
                self.best_map[metric_prefix] = current_map
                self.best_epoch[metric_prefix] = trainer.cur_epoch
                
                metric_dir = f"{output_dir}/{metric_prefix}"
                if not os.path.exists(metric_dir):
                    os.makedirs(metric_dir)
                output_path = f"{metric_dir}/best_model.pt"

                checkpoint = {
                    "epoch": trainer.cur_epoch,
                    "model_state": trainer.model.state_dict(),
                    "optimizer_state": trainer.optimizer.state_dict(),
                    "scheduler_state": trainer.lr_scheduler.state_dict() if hasattr(trainer, "lr_scheduler") else None,
                    "best_map": current_map,
                    "best_epoch": trainer.cur_epoch,
                    "detection_metrics": eval_metrics.ap_per_class,
                    "drivable_metrics": eval_metrics.drivable_metrics,
                }
                
                torch.save(checkpoint, output_path)

                trainer.logger.log_message(f'New best model saved! mAP: {current_map:.4f} -> {output_path}')
                # Log to WandB
                trainer.wandb_logger.log_metrics({
                    "val/best_map": current_map,
                    "val/best_epoch": trainer.cur_epoch
                }, step=trainer.cur_epoch)
                
                trainer.wandb_logger.log_artifact(
                    artifact_path=output_path,
                    artifact_type="model",
                    name=f"best_model_{metric_prefix}_epoch{trainer.cur_epoch}"
                )

class EnhancedImageLogger(TrainerCallback):
    
    def __init__(self, train_log_idx=200, eval_log_idx:int=200):
        self.train_log_idx = train_log_idx
        self.eval_log_idx = eval_log_idx

    def _log_images(self, trainer:Trainer, key:str, step, caption:str, images:torch.Tensor):
        if hasattr(trainer.model, "enhanced_image") and trainer.model.enhanced_image is not None:

            enhanced_imgs = trainer.model.enhanced_image.clamp(0, 1)
            original_images = images.clamp(0, 1)

            combined = torch.cat([original_images, enhanced_imgs], dim=3)
            combined = (combined * 255).to(torch.uint8)

            trainer.wandb_logger.log_images(
                key,
                combined,
                step=step,
                caption=caption
            )
            trainer.model.enhanced_image = None

        #GDIPYolo 
        if hasattr(trainer.model, "gates") and trainer.model.gates is not None:
            gate_names = ["white_balance", "gamma", "identity", "sharpening", "defog", "contrast", "tone"]

            if isinstance(trainer.model.gates, torch.Tensor):
                # Single-level GDIP: gates shape (batch_size, 7)
                gate_means = trainer.model.gates.mean(dim=0)
                gate_metrics = {f"train/gates/{name}": gate_means[i].item() for i, name in enumerate(gate_names)}
                trainer.wandb_logger.log_metrics(gate_metrics, step=step)
            elif isinstance(trainer.model.gates, list):
                # Multi-level GDIP: list of (batch_size, 7) tensors
                for block_idx, block_gates in enumerate(trainer.model.gates):
                    gate_means = block_gates.mean(dim=0)
                    gate_metrics = {f"train/gates/block{block_idx}/{name}": gate_means[i].item() for i, name in enumerate(gate_names)}
                    trainer.wandb_logger.log_metrics(gate_metrics, step=step)

            trainer.model.gates = None        
    
    def on_step_end(self, trainer):
        
        if trainer.has_enhancement and (trainer.train_batch_idx + 1) % self.train_log_idx == 0:
            step = trainer.cur_epoch * trainer.total_train_batch + trainer.train_batch_idx
            caption = f'train_{trainer.cur_epoch}_batch_{trainer.train_batch_idx}'
            
            self._log_images(trainer=trainer,
                            key="train/original_vs_enhanced",
                            step=step,
                            caption=caption,
                            images=trainer.batch_images)
                
                
    def on_eval_batch_end(self, trainer):

        if trainer.has_enhancement and (trainer.eval_batch_idx + 1) % self.eval_log_idx == 0:

            step = trainer.cur_epoch * trainer.total_eval_batch + trainer.eval_batch_idx
            caption = f'{trainer.eval_metric_prefix}_{trainer.cur_epoch}_batch_{trainer.eval_batch_idx}'

            self._log_images(
                trainer, 
                key=f"{trainer.eval_metric_prefix}/enhanced_images",
                step=step,
                caption=caption,
                images=trainer.eval_batch_ctx.cur_eval_images
            )
            
            
class EvalMetricsCallback(TrainerCallback):
    
    def __init__(self, conf_threshold = 0.001,
                iou_threshold = 0.45,
                max_detections = 500,
                stats_iou_threshold : float = 0.25,
                num_drivable_classes = 2,
                num_lane_classes = 2,  # Will be updated dynamically if needed
                lane_det_conf_threshold = 0.5,
                lane_det_nms_threshold = 0.5,
                lane_det_iou_threshold = 0.5,
                visualize_idx:int = 100):

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.visualize_idx = visualize_idx
        self.stats_iou_threshold = stats_iou_threshold

        self.num_drivable_classes = num_drivable_classes
        self.num_lane_classes = num_lane_classes

        self.lane_det_conf_threshold = lane_det_conf_threshold
        self.lane_det_nms_threshold = lane_det_nms_threshold
        self.lane_det_iou_threshold = lane_det_iou_threshold

        self._reset()

    def _reset(self):

        self.total_val_loss = 0.0
        self.total_det_loss = 0.0
        self.total_drivable_loss = 0.0
        self.total_lane_loss = 0.0
        self.total_lane_det_loss = 0.0

        self.global_image_idx = 0

        self.dets_by_image = defaultdict(None) #image_id -> (num_dets, 6)
        self.gt_by_image = defaultdict(None) #image_id -> (num_gt, 5)

        self.drivable_confusion_matrix = torch.zeros(
                        self.num_drivable_classes,
                        self.num_drivable_classes, dtype=torch.int64
                    )

        self.lane_confusion_matrix = None

        self.lane_det_preds = []   # list of list[dict] per image
        self.lane_det_gts = []     # list of list[dict] per image

    def _post_process_detections(self, trainer:Trainer):
        
        outputs = trainer.eval_batch_ctx.cur_eval_model_outputs
        
        if outputs.detection_predictions is not None:
            detection_preds = outputs.detection_predictions
            
            if isinstance(detection_preds, torch.Tensor):
                # yolov8 anchor-free postProcess
                nms_results = DetectionHelper.non_max_suppression_v8(
                    detection_preds, 
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections
                )

            else:
                batch_predictions = []
                for layer_pred in detection_preds:
                    b, na, h, w, nc = layer_pred.shape
                    layer_pred_flat = layer_pred.view(b, na * h * w, nc)
                    batch_predictions.append(layer_pred_flat)

                concatenated_preds = torch.cat(batch_predictions, dim=1)

                # Apply NMS
                nms_results = DetectionHelper.non_max_suppression(
                    concatenated_preds,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                    max_detections=self.max_detections
                )

            for image_idx, dets in enumerate(nms_results):
                if dets is not None:
                    self.dets_by_image[self.global_image_idx] = dets
                else:
                    self.dets_by_image[self.global_image_idx] = None
                
                mask = trainer.eval_batch_ctx.cur_eval_gt_detections[:, 0] == image_idx
                img_targets = trainer.eval_batch_ctx.cur_eval_gt_detections[mask]
                gts = None
                
                if img_targets.shape[0] > 0:
                    print(img_targets.shape)
                    boxes_xywh = img_targets[:, 2:6]
                    boxes_xywh[:, [0,2]] *= trainer.eval_batch_ctx.cur_eval_image_w
                    boxes_xywh[:, [1,3]] *= trainer.eval_batch_ctx.cur_eval_image_h
                    boxes_xyxy = DetectionHelper.xywh2xyxy(boxes_xywh)
                    classes = img_targets[:, 1:2]
                    gts = torch.cat([boxes_xyxy, classes], dim=1)
                    self.gt_by_image[self.global_image_idx] = gts
                else:
                    self.gt_by_image[self.global_image_idx] = None

                self.global_image_idx += 1
                
                if trainer.training_args.eval_visualize_outputs and \
                    (trainer.eval_batch_idx + 1) % self.visualize_idx == 0:

                        if not os.path.exists(f'{trainer.training_args.output_dir}/visualizations/detections'):
                            os.makedirs(f'{trainer.training_args.output_dir}/visualizations/detections')

                        epoch_vis_dir = f'{trainer.training_args.output_dir}/visualizations/detections/eval_epoch_{trainer.cur_epoch}'
                        if not os.path.exists(epoch_vis_dir):
                            os.makedirs(epoch_vis_dir)

                        if trainer.eval_batch_ctx.cur_eval_image_paths and image_idx < len(trainer.eval_batch_ctx.cur_eval_image_paths):
                            img_name = os.path.basename(trainer.eval_batch_ctx.cur_eval_image_paths[image_idx])
                            save_path = os.path.join(epoch_vis_dir, f'vis_{img_name}.png')
                        else:
                            save_path = os.path.join(epoch_vis_dir, f'vis_{image_idx}.png')
                            
                        DetectionHelper.visualize_detections(
                            image=trainer.eval_batch_ctx.cur_eval_images[image_idx],
                            predictions=dets,
                            targets=gts,
                            class_names=CLASS_NAMES,
                            save_path=save_path
                        )                    
    
    def _post_process_drivable_segmentation_predictions(self, trainer:Trainer):
        
        drivable_preds = torch.argmax(
                            trainer.eval_batch_ctx.cur_eval_model_outputs.drivable_segmentation_predictions, 
                            dim=1)

        if trainer.eval_batch_ctx.cur_eval_gt_drivable_area_seg is not None:
            drivable_targets = trainer.eval_batch_ctx.cur_eval_gt_drivable_area_seg
            # Initialize confusion matrix on first batch

            # Update confusion matrix
            self.drivable_confusion_matrix += SegmentationUtils._compute_confusion_matrix(
                drivable_preds.cpu(), 
                drivable_targets.cpu(), 
                self.num_drivable_classes
            )

            if trainer.training_args.eval_visualize_outputs and \
                (trainer.eval_batch_idx + 1) % self.visualize_idx == 0:

                    if not os.path.exists(f'{trainer.training_args.output_dir}/visualizations/drivable_area'):
                        os.makedirs(f'{trainer.training_args.output_dir}/visualizations/drivable_area')

                    epoch_vis_dir = f'{trainer.training_args.output_dir}/visualizations/drivable_area/eval_epoch_{trainer.cur_epoch}'
                    if not os.path.exists(epoch_vis_dir):
                        os.makedirs(epoch_vis_dir)

                    batch_drivable_preds = SegmentationUtils.transparent_overlay(
                        original_imgs=trainer.eval_batch_ctx.cur_eval_images,
                        masks=drivable_preds
                    )

                    batch_drivable_gts = SegmentationUtils.transparent_overlay(
                        original_imgs=trainer.eval_batch_ctx.cur_eval_images,
                        masks=drivable_targets
                    )

                    image_paths = trainer.eval_batch_ctx.cur_eval_image_paths
                    for image_idx, (pred_overlay, gt_overlay) in enumerate(zip(batch_drivable_preds, batch_drivable_gts)):
                        if image_paths and image_idx < len(image_paths):
                            img_name = os.path.basename(image_paths[image_idx])
                            pred_save_path = os.path.join(epoch_vis_dir, f'vis_{img_name}_pred.png')
                            gt_save_path = os.path.join(epoch_vis_dir, f'vis_{img_name}_gt.png')
                        else:
                            pred_save_path = os.path.join(epoch_vis_dir, f'vis_{image_idx}_pred.png')
                            gt_save_path = os.path.join(epoch_vis_dir, f'vis_{image_idx}_gt.png')

                        SegmentationUtils.save_overlay_image(
                            vis_image=pred_overlay,
                            save_path=pred_save_path
                        )

                        SegmentationUtils.save_overlay_image(
                            vis_image=gt_overlay,
                            save_path=gt_save_path
                        )                            

    def _post_process_lane_segmentation_predictions(self, trainer:Trainer):
        #TODO, not tested or vetted
        lane_preds = torch.argmax(trainer.eval_batch_ctx.cur_eval_model_outputs.lane_segmentation_predictions, 
                                dim=1)

        if trainer.eval_batch_ctx.cur_eval_gt_lane_seg is not None:
            lane_targets = trainer.eval_batch_ctx.cur_eval_gt_lane_seg
            # Dynamically determine number of lane classes
            max_class = max(lane_preds.max().item(), lane_targets.max().item()) + 1
            if max_class > self.num_lane_classes:
                # Expand confusion matrix if needed
                if self.lane_confusion_matrix is not None:
                    old_matrix = self.lane_confusion_matrix
                    self.lane_confusion_matrix = torch.zeros(
                        max_class, max_class, dtype=torch.int64
                    )
                    self.lane_confusion_matrix[:old_matrix.shape[0], :old_matrix.shape[1]] = old_matrix
                num_lane_classes = max_class
            # Initialize confusion matrix on first batch
            if self.lane_confusion_matrix is None:
                self.lane_confusion_matrix = torch.zeros(
                    num_lane_classes, num_lane_classes, dtype=torch.int64
                )
            # Update confusion matrix
            self.lane_confusion_matrix += SegmentationUtils._compute_confusion_matrix(
                lane_preds.cpu(), lane_targets.cpu(), num_lane_classes
            )

    def _post_process_lane_detection_predictions(self, trainer: Trainer):
        outputs = trainer.eval_batch_ctx.cur_eval_model_outputs
        if outputs.lane_detection_predictions is None:
            return

        preds = outputs.lane_detection_predictions                     # (bs, 192, 78)
        img_h = trainer.eval_batch_ctx.cur_eval_image_h
        img_w = trainer.eval_batch_ctx.cur_eval_image_w
        gt_lanes = trainer.eval_batch_ctx.cur_eval_gt_lane_detections  # (bs, max_lanes, 78) or None

        for b in range(preds.shape[0]):
            nms_preds = lane_nms(preds[b], img_w,
                                 conf_threshold=self.lane_det_conf_threshold,
                                 nms_threshold=self.lane_det_nms_threshold)
            pred_lanes = predictions_to_lanes(nms_preds, img_h, img_w)

            gt_lane_list = []
            if gt_lanes is not None:
                gt_b = gt_lanes[b]                                     # (max_lanes, 78)
                valid = gt_b[gt_b[:, 0] == 1]                         # (T, 78) valid GTs
                if len(valid) > 0:
                    gt_lane_list = predictions_to_lanes(
                        _make_activated_gt(valid), img_h, img_w)

            self.lane_det_preds.append(pred_lanes)
            self.lane_det_gts.append(gt_lane_list)

    def on_eval_batch_end(self, trainer):
        
        outputs = trainer.eval_batch_ctx.cur_eval_model_outputs

        # Accumulate losses
        if outputs.detection_loss is not None:
            self.total_det_loss += outputs.detection_loss.item()
            self.total_val_loss += outputs.detection_loss.item()

        if outputs.drivable_segmentation_loss is not None:
            self.total_drivable_loss += outputs.drivable_segmentation_loss.item()
            self.total_val_loss += outputs.drivable_segmentation_loss.item()

        if outputs.lane_segmentation_loss is not None:
            self.total_lane_loss += outputs.lane_segmentation_loss.item()
            self.total_val_loss += outputs.lane_segmentation_loss.item()

        if outputs.lane_detection_loss is not None:
            self.total_lane_det_loss += outputs.lane_detection_loss.item()
            self.total_val_loss += outputs.lane_detection_loss.item()

        # Process detection predictions - concatenate layer outputs and apply NMS
        self._post_process_detections(trainer)

        if trainer.eval_batch_ctx.cur_eval_model_outputs.drivable_segmentation_predictions is not None:
            self._post_process_drivable_segmentation_predictions(trainer)

        if trainer.eval_batch_ctx.cur_eval_model_outputs.lane_segmentation_predictions is not None:
            self._post_process_lane_segmentation_predictions(trainer)

        self._post_process_lane_detection_predictions(trainer)
        
    def _compute_detection_metrics(self, trainer:Trainer):

        # Average losses
        ap_results, stats_per_class = DetectionMetrics.compute_stats(
            self.dets_by_image,
            self.gt_by_image,
            iou_threshold=self.stats_iou_threshold,
            num_classes=len(CLASS_NAMES)
        )
        
        metric_prefix = trainer.eval_metric_prefix
        num_classes = len(BDD100KClassesReduced)

        # Create AP table for logging with class names
        ap_table_data = [["Class", "AP"]]
        for cls in range(num_classes):
            class_name = BDD100KClassesReduced(cls).name
            ap_value = ap_results.get(f'AP_class_{cls}', 0.0)
            ap_table_data.append([f"{cls}: {class_name}", f"{ap_value:.4f}"])
        ap_table_data.append(["mAP@0.5", f"{ap_results['mAP']:.4f}"])

        trainer.eval_batch_ctx.ap_table_data = ap_table_data
        
        ap_table_string = AsciiTable(trainer.eval_batch_ctx.ap_table_data).table
        trainer.logger.log_message(f"\n[{metric_prefix}] Detection Metrics (AP@0.5):")
        trainer.logger.log_message(ap_table_string)        

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

        trainer.eval_batch_ctx.stats_table_data = stats_table_data
        trainer.eval_batch_ctx.stats_iou_threshold = self.stats_iou_threshold
        
        stats_table_string = AsciiTable(trainer.eval_batch_ctx.stats_table_data).table
        trainer.logger.log_message(f"\n[{metric_prefix}] Detection Metrics @{trainer.eval_batch_ctx.stats_iou_threshold}:")
        trainer.logger.log_message(stats_table_string)
        trainer.logger.log_line()

        # Log AP table to WandB
        wandb_ap_data = [[f"{cls}: {BDD100KClassesReduced(cls).name}", ap_results.get(f'AP_class_{cls}', 0.0)] for cls in range(num_classes)]
        wandb_ap_data.append([f"mAP@{self.stats_iou_threshold}", ap_results['mAP']])

        trainer.eval_batch_ctx.wandb_ap_data = wandb_ap_data
        trainer.wandb_logger.log_table(
            f"{metric_prefix}/detection_ap",
            columns=["Class", "AP"],
            data=trainer.eval_batch_ctx.wandb_ap_data,
            step=trainer.cur_epoch
        )        
        
        trainer.eval_metrics[metric_prefix].ap_per_class = ap_results

    def _compute_drivable_metrics(self, trainer:Trainer):

        metric_prefix = trainer.eval_metric_prefix
        if self.drivable_confusion_matrix.sum() > 0:
            drivable_iou, drivable_dice = SegmentationUtils._compute_metrics_from_confusion_matrix(
                self.drivable_confusion_matrix, 
                self.num_drivable_classes
            )

            trainer.eval_metrics[metric_prefix].drivable_metrics = {**drivable_iou, **drivable_dice}

            drivable_table_data = [["Metric", "Value"]]
            drivable_table_data.append(["mIoU", f"{drivable_iou['mIoU']:.4f}"])
            drivable_table_data.append(["mDice", f"{drivable_dice['mDice']:.4f}"])
            for cls in range(self.num_drivable_classes):
                drivable_table_data.append([f"IoU Class {cls}", f"{drivable_iou.get(f'IoU_class_{cls}', 0.0):.4f}"])

            trainer.eval_batch_ctx.drivable_table_data = drivable_table_data
            drivable_table_string = AsciiTable(drivable_table_data).table

            trainer.logger.log_message(f"\n[{metric_prefix}] Drivable Segmentation Metrics:")
            trainer.logger.log_message(drivable_table_string)
            trainer.logger.log_line()

            # Log to WandB
            wandb_drivable_data = [
                ["mIoU", drivable_iou['mIoU']],
                ["mDice", drivable_dice['mDice']]
            ]

            for cls in range(self.num_drivable_classes):
                wandb_drivable_data.append([f"IoU_class_{cls}", 
                                            drivable_iou.get(f'IoU_class_{cls}', 0.0)])

            trainer.wandb_logger.log_table(
                f"{metric_prefix}/drivable_metrics",
                columns=["Metric", "Value"],
                data=wandb_drivable_data,
                step=trainer.cur_epoch
            )

    def _compute_lane_detection_metrics(self, trainer: Trainer):
        if not self.lane_det_preds:
            return

        tp, fp, fn = 0, 0, 0

        for pred_lanes, gt_lanes in zip(self.lane_det_preds, self.lane_det_gts):
            matched_gt = set()
            for pred in pred_lanes:
                best_iou, best_gt_idx = 0.0, -1
                for g_idx, gt in enumerate(gt_lanes):
                    if g_idx in matched_gt:
                        continue
                    iou = polyline_iou(
                        pred['points'], gt['points'],
                        trainer.eval_batch_ctx.cur_eval_image_h,
                        trainer.eval_batch_ctx.cur_eval_image_w
                    )
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, g_idx
                if best_iou >= self.lane_det_iou_threshold and best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            fn += len(gt_lanes) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metric_prefix = trainer.eval_metric_prefix
        metrics = {'F1': f1, 'Precision': precision, 'Recall': recall,
                   'TP': tp, 'FP': fp, 'FN': fn}

        trainer.eval_metrics[metric_prefix].lane_detection_metrics = metrics

        lane_table_data = [["Metric", "Value"]]
        lane_table_data.append(["F1", f"{f1:.4f}"])
        lane_table_data.append(["Precision", f"{precision:.4f}"])
        lane_table_data.append(["Recall", f"{recall:.4f}"])
        lane_table_data.append(["TP / FP / FN", f"{tp} / {fp} / {fn}"])

        lane_table_string = AsciiTable(lane_table_data).table
        trainer.logger.log_message(f"\n[{metric_prefix}] Lane Detection Metrics @IoU={self.lane_det_iou_threshold}:")
        trainer.logger.log_message(lane_table_string)
        trainer.logger.log_line()

        wandb_lane_data = [
            ["F1", f1], ["Precision", precision], ["Recall", recall],
            ["TP", tp], ["FP", fp], ["FN", fn]
        ]
        trainer.wandb_logger.log_table(
            f"{metric_prefix}/lane_detection_metrics",
            columns=["Metric", "Value"],
            data=wandb_lane_data,
            step=trainer.cur_epoch
        )

    def on_eval_end(self, trainer):

        if trainer.total_eval_batch > 0:
            avg_det_loss = self.total_det_loss / trainer.total_eval_batch
            avg_drivable_loss = self.total_drivable_loss / trainer.total_eval_batch
            avg_lane_loss = self.total_lane_loss / trainer.total_eval_batch
            avg_lane_det_loss = self.total_lane_det_loss / trainer.total_eval_batch
            avg_total_loss = (self.total_det_loss + self.total_drivable_loss +
                              self.total_lane_loss + self.total_lane_det_loss) / trainer.total_eval_batch

        # Log summary
        trainer.logger.log_line()
        trainer.logger.log_message(
            f'[{trainer.eval_metric_prefix}] Epoch {trainer.cur_epoch} - Avg Loss: {avg_total_loss:.4f} | '
            f'Det Loss: {avg_det_loss:.4f} | Drivable Loss: {avg_drivable_loss:.4f} | '
            f'Lane Seg Loss: {avg_lane_loss:.4f} | Lane Det Loss: {avg_lane_det_loss:.4f}'
        )

        trainer.logger.log_new_line()

        self._compute_detection_metrics(trainer)
        self._compute_drivable_metrics(trainer)
        self._compute_lane_detection_metrics(trainer)

        self._reset()
