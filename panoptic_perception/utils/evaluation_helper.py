"""
Evaluation helper functions for object detection and segmentation.

Includes:
    - AP/mAP calculation
    - Confusion matrices
    - Visualization utilities
    - Segmentation metrics (IoU, Dice, etc.)

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import cv2

from panoptic_perception.utils.detection_utils import DetectionHelper


# ============================================================================
# DETECTION METRICS
# ============================================================================

class DetectionMetrics:
    """Metrics for object detection evaluation (AP, mAP, etc.)."""

    
    @staticmethod
    def compute_stats(
        all_detections: Dict[int, torch.Tensor],
        all_ground_truths: Dict[int, torch.Tensor],
        iou_threshold: float = 0.5,
        num_classes: int = 10        
    ):
        """
            PR Curve - 
                For each class Calculate Average Precision (AP) and mean AP (mAP)
                    Collect all detections for that class across all images
                    Sort them by confidence
                    Walk the list one detection at a time
                    After each detection, update cumulative TP / FP
                    
            General Stats : 
                True Positives, False Positives, False Negatives per class

        Args:
            all_detections: Dict of detection tensors per image 
                            image_id -> (num_dets, 6)
                            Each: (num_dets, 6) [x1, y1, x2, y2, conf, class]
            all_ground_truths: List of ground truth tensors per image
                            image_id -> (num_gt, 5)
                            Each: (num_gt, 5) [x1, y1, x2, y2, class]
            iou_threshold: IoU threshold for matching
            num_classes: Number of classes

        Returns:
            Dictionary with AP per class and mAP
        """
        
        ap_per_class = defaultdict(float)
        stats_per_class = defaultdict(lambda : defaultdict(float))

        tp_img = np.zeros(shape=(len(all_detections), num_classes)) #per image
        fp_img = np.zeros(shape=(len(all_detections), num_classes)) #per image
        fn_img = np.zeros(shape=(len(all_detections), num_classes))

        for cls in range(num_classes):
            all_scores = []
            all_is_tp = []
            total_gt = 0
            
            for image_idx in all_detections:
                detetctions = all_detections[image_idx]
                gts = all_ground_truths.get(image_idx)

                if detetctions is None:
                    continue
                if gts is None:
                    gts = detetctions.new_zeros((0, 5))
                
                cls_dets = detetctions[detetctions[:, -1] == cls]
                cls_gts = gts[gts[:, -1] == cls]
                total_gt += cls_gts.shape[0]
                
                if cls_dets.shape[0] == 0:
                    #false negatives
                    if cls_gts.shape[0] > 0:
                        fn_img[image_idx][cls] += cls_gts.shape[0]                    
                
                    continue
                
                #false postives
                if cls_dets.shape[0] > 0 and cls_gts.shape[0] == 0:
                    fp_img[image_idx][cls] += cls_dets.shape[0]
                    
                    all_scores.extend(cls_dets[:, 4].cpu().tolist())
                    all_is_tp.extend([0]*cls_dets.shape[0])
                    
                    continue
                
                ious = DetectionHelper.box_iou(
                    cls_dets[:, :4],
                    cls_gts[:, :4])
                
                sorted_idx = torch.argsort(
                    cls_dets[:, 4], descending=True
                )
                
                cls_dets = cls_dets[sorted_idx]
                ious = ious[sorted_idx]
                
                matched_gt = torch.zeros(cls_gts.shape[0], dtype=torch.bool)
                
                for det_i in range(cls_dets.shape[0]):
                    max_iou, gt_i = ious[det_i].max(dim=0)
                    if max_iou >= iou_threshold and not matched_gt[gt_i]:
                        matched_gt[gt_i] = True
                        all_is_tp.append(1)
                        tp_img[image_idx][cls] += 1
                        
                    else:
                        all_is_tp.append(0)
                        fp_img[image_idx][cls] += 1
                        
                    all_scores.append(cls_dets[det_i, 4].item())
                        
                fn_img[image_idx][cls] += (~matched_gt).sum().item()
                
            if not all_scores:
                ap_per_class[f"AP_class_{cls}"] = 0.0
                continue
            
            sorted_idx = np.argsort(-np.array(all_scores))
            tp = np.array(all_is_tp)[sorted_idx]
            fp = 1 - tp
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recall = tp_cumsum / (total_gt + 1e-5)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-5)
            
            ap_per_class[f"AP_class_{cls}"] = DetectionMetrics.compute_ap_11point(
                recall, precision
            )
            
            
        tp_per_cls = tp_img.sum(axis=0)
        fp_per_cls = fp_img.sum(axis=0)
        fn_per_cls = fn_img.sum(axis=0)
        
        for cls in range(num_classes):
            tp = tp_per_cls[cls].item()
            fp = fp_per_cls[cls].item()
            fn = fn_per_cls[cls].item()
            
            stats_per_class[cls]["total_gt"] = tp + fn
            stats_per_class[cls][f"true_positives"] = tp
            stats_per_class[cls][f"false_positives"] = fp
            stats_per_class[cls][f"false_negatives"] = fn
            
        return ap_per_class, stats_per_class
    
    @staticmethod
    def calculate_ap(
        detections: List[torch.Tensor],
        ground_truths: List[torch.Tensor],
        iou_threshold: float = 0.5,
        num_classes: int = 10
    ) -> Dict[str, float]:
        """
        Calculate Average Precision (AP) for each class and mean AP (mAP).

        Args:
            detections: List of detection tensors per image
                       Each: (num_dets, 6) [x1, y1, x2, y2, conf, class]
            ground_truths: List of ground truth tensors per image
                          Each: (num_gt, 5) [x1, y1, x2, y2, class]
            iou_threshold: IoU threshold for matching
            num_classes: Number of classes

        Returns:
            Dictionary with AP per class and mAP
        """
        ap_per_class = {}

        for cls in range(num_classes):
            # Collect all detections and ground truths for this class
            all_detections = []
            all_ground_truths = []

            for img_idx in range(len(detections)):
                # Get detections for this class
                if detections[img_idx] is not None:
                    cls_mask = detections[img_idx][:, 5] == cls
                    cls_dets = detections[img_idx][cls_mask]

                    # Store: [image_id, confidence, x1, y1, x2, y2]
                    for det in cls_dets:
                        all_detections.append([img_idx, det[4].item()] + det[:4].tolist())

                # Get ground truths for this class
                if ground_truths[img_idx] is not None:
                    cls_mask = ground_truths[img_idx][:, 4] == cls
                    cls_gts = ground_truths[img_idx][cls_mask]

                    for gt in cls_gts:
                        all_ground_truths.append([img_idx] + gt[:4].tolist())

            if len(all_detections) == 0:
                ap_per_class[f'AP_class_{cls}'] = 0.0
                continue

            # Sort detections by confidence
            all_detections = sorted(all_detections, key=lambda x: x[1], reverse=True)

            # Match detections to ground truths
            tp = np.zeros(len(all_detections))
            fp = np.zeros(len(all_detections))

            # Track which ground truths have been matched
            gt_matched = defaultdict(set)

            for det_idx, det in enumerate(all_detections):
                img_id = int(det[0])
                det_box = torch.tensor(det[2:6]).unsqueeze(0)

                # Get ground truths for this image
                img_gts = [gt for gt in all_ground_truths if gt[0] == img_id]

                if len(img_gts) == 0:
                    fp[det_idx] = 1
                    continue

                # Compute IoU with all ground truths
                gt_boxes = torch.tensor([gt[1:5] for gt in img_gts])
                ious = DetectionHelper.box_iou(det_box, gt_boxes)[0]

                # Find best matching ground truth
                max_iou, max_idx = ious.max(0)

                if max_iou >= iou_threshold and max_idx.item() not in gt_matched[img_id]:
                    tp[det_idx] = 1
                    gt_matched[img_id].add(max_idx.item())
                else:
                    fp[det_idx] = 1

            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / (len(all_ground_truths) + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            # Compute AP using 11-point interpolation
            ap = DetectionMetrics.compute_ap_11point(recalls, precisions)
            ap_per_class[f'AP_class_{cls}'] = ap

        # Compute mAP
        ap_values = [v for k, v in ap_per_class.items() if k.startswith('AP_class_')]
        ap_per_class['mAP'] = np.mean(ap_values) if ap_values else 0.0

        return ap_per_class

    @staticmethod
    def compute_ap_11point(recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Compute AP using 11-point interpolation.

        Args:
            recall: Recall values
            precision: Precision values

        Returns:
            Average Precision
        """
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11

        return ap

    @staticmethod
    def confusion_matrix_detection(
        detections: List[torch.Tensor],
        ground_truths: List[torch.Tensor],
        num_classes: int = 10,
        iou_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Compute confusion matrix for object detection.

        Args:
            detections: List of detections per image
            ground_truths: List of ground truths per image
            num_classes: Number of classes
            iou_threshold: IoU threshold for matching

        Returns:
            Confusion matrix (num_classes+1, num_classes+1)
            Rows: Ground truth, Cols: Predictions
            Last row/col is background/no detection
        """
        matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

        for img_idx in range(len(detections)):
            gt = ground_truths[img_idx]
            det = detections[img_idx]

            if gt is None or len(gt) == 0:
                # No ground truth, count as background
                if det is not None and len(det) > 0:
                    for d in det:
                        pred_cls = int(d[5].item())
                        matrix[num_classes, pred_cls] += 1  # FP
                continue

            gt_matched = set()

            if det is not None and len(det) > 0:
                # Compute IoU between all detections and ground truths
                ious = DetectionHelper.box_iou(det[:, :4], gt[:, :4])

                for det_idx in range(len(det)):
                    max_iou, max_gt_idx = ious[det_idx].max(0)

                    pred_cls = int(det[det_idx, 5].item())

                    if max_iou >= iou_threshold:
                        gt_cls = int(gt[max_gt_idx, 4].item())
                        matrix[gt_cls, pred_cls] += 1
                        gt_matched.add(max_gt_idx.item())
                    else:
                        # False positive (predicted but no match)
                        matrix[num_classes, pred_cls] += 1

            # Count unmatched ground truths as false negatives
            for gt_idx in range(len(gt)):
                if gt_idx not in gt_matched:
                    gt_cls = int(gt[gt_idx, 4].item())
                    matrix[gt_cls, num_classes] += 1  # FN

        return matrix


# ============================================================================
# SEGMENTATION METRICS
# ============================================================================

class SegmentationMetrics:
    """Metrics for semantic segmentation evaluation."""

    @staticmethod
    def compute_iou(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int,
        ignore_index: int = 255
    ) -> Dict[str, float]:
        """
        Compute IoU (Intersection over Union) for semantic segmentation.

        Args:
            predictions: (N, H, W) predicted class labels
            targets: (N, H, W) ground truth class labels
            num_classes: Number of classes
            ignore_index: Index to ignore in computation

        Returns:
            Dictionary with IoU per class and mean IoU
        """
        ious = {}

        # Flatten
        predictions = predictions.flatten()
        targets = targets.flatten()

        # Remove ignore index
        mask = targets != ignore_index
        predictions = predictions[mask]
        targets = targets[mask]

        for cls in range(num_classes):
            pred_mask = predictions == cls
            target_mask = targets == cls

            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()

            if union == 0:
                iou = float('nan')
            else:
                iou = (intersection / union).item()

            ious[f'IoU_class_{cls}'] = iou

        # Compute mean IoU (excluding NaN values)
        iou_values = [v for v in ious.values() if not np.isnan(v)]
        ious['mIoU'] = np.mean(iou_values) if iou_values else 0.0

        return ious

    @staticmethod
    def compute_dice(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int
    ) -> Dict[str, float]:
        """
        Compute Dice coefficient for segmentation.

        Args:
            predictions: (N, H, W) predicted class labels
            targets: (N, H, W) ground truth labels
            num_classes: Number of classes

        Returns:
            Dictionary with Dice per class and mean Dice
        """
        dice_scores = {}

        predictions = predictions.flatten()
        targets = targets.flatten()

        for cls in range(num_classes):
            pred_mask = (predictions == cls).float()
            target_mask = (targets == cls).float()

            intersection = (pred_mask * target_mask).sum()
            dice = (2. * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-6)

            dice_scores[f'Dice_class_{cls}'] = dice.item()

        # Mean Dice
        dice_values = list(dice_scores.values())
        dice_scores['mDice'] = np.mean(dice_values) if dice_values else 0.0

        return dice_scores

    @staticmethod
    def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute pixel accuracy.

        Args:
            predictions: (N, H, W) predicted labels
            targets: (N, H, W) ground truth labels

        Returns:
            Pixel accuracy (0-1)
        """
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        return correct / total

    @staticmethod
    def confusion_matrix_segmentation(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int
    ) -> np.ndarray:
        """
        Compute confusion matrix for segmentation.

        Args:
            predictions: (N, H, W) predicted labels
            targets: (N, H, W) ground truth labels
            num_classes: Number of classes

        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        predictions = predictions.flatten().cpu().numpy()
        targets = targets.flatten().cpu().numpy()

        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        for t, p in zip(targets, predictions):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                matrix[t, p] += 1

        return matrix


# ============================================================================
# SEGMENTATION HELPER
# ============================================================================

class SegmentationHelper:
    """Helper functions for segmentation visualization and post-processing."""

    @staticmethod
    def colorize_mask(mask: np.ndarray, color_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert segmentation mask to RGB image.

        Args:
            mask: (H, W) segmentation mask with class indices
            color_map: (num_classes, 3) RGB colors for each class

        Returns:
            (H, W, 3) RGB image
        """
        if color_map is None:
            # Default colormap
            color_map = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))[:, :3] * 255
            color_map = color_map.astype(np.uint8)

        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id in np.unique(mask):
            if cls_id < len(color_map):
                rgb[mask == cls_id] = color_map[cls_id]

        return rgb


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

class Visualizer:
    """Visualization utilities for detections and segmentation."""

    @staticmethod
    def draw_bboxes(
        image: np.ndarray,
        boxes: np.ndarray,
        labels: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: (H, W, 3) RGB image
            boxes: (N, 4) boxes in [x1, y1, x2, y2] format
            labels: (N,) class labels
            scores: (N,) confidence scores
            class_names: List of class names
            color: Box color (B, G, R)
            thickness: Line thickness

        Returns:
            Image with drawn boxes
        """
        img = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Prepare label text
            label_text = ""
            if labels is not None and class_names is not None:
                label_text = class_names[int(labels[i])]
            if scores is not None:
                label_text += f" {scores[i]:.2f}"

            if label_text:
                # Draw label background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img,
                    (x1, y1 - text_height - baseline),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                # Draw label text
                cv2.putText(
                    img,
                    label_text,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return img

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
        normalize: bool = False
    ):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            save_path: Path to save figure
            normalize: Whether to normalize by row
        """
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_metrics(
        metrics_dict: Dict[str, List[float]],
        title: str = "Training Metrics",
        save_path: Optional[str] = None
    ):
        """
        Plot training metrics over epochs.

        Args:
            metrics_dict: Dictionary of metric_name -> list of values
            title: Plot title
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 6))

        for metric_name, values in metrics_dict.items():
            plt.plot(values, label=metric_name, marker='o')

        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_pr_curve(
        precisions: np.ndarray,
        recalls: np.ndarray,
        ap: float,
        class_name: str = "",
        save_path: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve.

        Args:
            precisions: Precision values
            recalls: Recall values
            ap: Average Precision
            class_name: Name of class
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve{" - " + class_name if class_name else ""}\nAP = {ap:.3f}')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    @staticmethod
    def visualize_segmentation(
        image: np.ndarray,
        prediction: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        color_map: Optional[np.ndarray] = None,
        alpha: float = 0.5,
        save_path: Optional[str] = None
    ):
        """
        Visualize segmentation results.

        Args:
            image: (H, W, 3) RGB image
            prediction: (H, W) predicted segmentation mask
            ground_truth: (H, W) ground truth mask (optional)
            color_map: Color map for classes
            alpha: Transparency for overlay
            save_path: Path to save visualization
        """
        num_plots = 2 if ground_truth is None else 3

        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
        if num_plots == 2:
            axes = [axes[0], axes[1]]

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Predicted segmentation
        pred_colored = SegmentationHelper.colorize_mask(prediction, color_map)
        overlay = cv2.addWeighted(image, 1-alpha, pred_colored, alpha, 0)
        axes[1].imshow(overlay)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        # Ground truth (if provided)
        if ground_truth is not None:
            gt_colored = SegmentationHelper.colorize_mask(ground_truth, color_map)
            overlay_gt = cv2.addWeighted(image, 1-alpha, gt_colored, alpha, 0)
            axes[2].imshow(overlay_gt)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()
