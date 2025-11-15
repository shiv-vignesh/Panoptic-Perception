import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
from panoptic_perception.models.models import YOLOP
from panoptic_perception.utils.detection_utils import DetectionHelper
from panoptic_perception.utils.evaluation_helper import (
    DetectionMetrics,
    SegmentationMetrics,
    SegmentationHelper,
    Visualizer
)

def create_dataloader():
    
    print("\n" + "="*60)
    print("TEST: Dataset Creation")
    print("="*60)

    # Define paths (adjust these to your actual data paths)
    dataset_kwargs = {
        "images_dir": "panoptic_perception/BDD100k/100k/100k",
        "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
        "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }    
    
    try:
        dataset = BDD100KDataset(dataset_kwargs, dataset_type='train')
        print(f"✓ Dataset created successfully")
        print(f"  Dataset type: {dataset.dataset_type}")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Images directory: {dataset.images_dir}")
        print(f"  First few image IDs: {dataset.image_ids[:5]}")

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging, increase for training
            collate_fn=BDDPreprocessor.collate_fn
        )

        print(f"✓ DataLoader created successfully")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Number of batches: {len(dataloader)}")
        print(f"  Shuffle: {dataloader.shuffle if hasattr(dataloader, 'shuffle') else 'N/A'}")

        return dataloader
        
    except Exception as e:
        print(f"⚠ Dataset creation failed (expected if data not present)")
        print(f"  Error: {e}")
        print(f"  Note: Update paths in test_dataset_creation() to point to actual BDD100K data")
        
        exit(1)
        
def test_integration():
    dataloader = create_dataloader()
    
    print("\n" + "="*60)
    print("TEST: Model Forward Pass with DataLoader")
    print("="*60)
    
    cfg_path = 'panoptic_perception/configs/models/yolop.cfg'
    model = YOLOP(cfg_path)
    module_defs = model.module_defs

    assert isinstance(module_defs, list), "Module definitions should be a list"
    assert len(module_defs) > 0, "Module definitions should not be empty"
    
    model.eval()
    
    batch = next(iter(dataloader))
    
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch images shape: {batch['images'].shape}")
    if batch.get('detections') is not None:
        print(f"detections targets shape: {batch['detections'].shape}")

    # Set model to training mode to compute loss
    model.eval()

    outputs = model(
        batch["images"],
        targets={
            "drivable_area_seg": batch.get("drivable_area_seg"),
            "lane_seg": batch.get("segmentation_masks"),
            "detections": batch["detections"]
        }
    )

    print(f"\nModel outputs: {outputs.keys()}")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    print("\n" + "="*60)
    print("TEST: NaN Check")
    print("="*60)            

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print("NaN in", name)
            break

    for name, buffer in model.named_buffers():
        if torch.isnan(buffer).any():
            print("NaN buffer in", name)
            break
    

def test_evaluation_helpers():
    """Test evaluation helper functions with model outputs."""

    print("\n" + "="*60)
    print("TEST: Evaluation Helper Functions")
    print("="*60)

    # Create dataloader and model
    dataloader = create_dataloader()
    cfg_path = 'panoptic_perception/configs/models/yolop.cfg'
    model = YOLOP(cfg_path)
    model.eval()

    # Get a batch and run forward pass
    batch = next(iter(dataloader))

    with torch.no_grad():
        outputs = model(batch["images"])

    # ========================================================================
    # TEST 1: Non-Maximum Suppression
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 1: Non-Maximum Suppression")
    print("-"*60)

    # Use actual model detection predictions
    # detection_predictions is a list of tensors, one per detection layer
    # Each has shape: (batch_size, num_anchors, grid_h, grid_w, 5+num_classes)
    # We need to reshape and concatenate them for NMS

    detection_preds = outputs.detection_predictions
    batch_size = batch["images"].shape[0]

    # Concatenate predictions from all detection layers
    all_predictions = []
    for layer_pred in detection_preds:
        # Reshape: (B, num_anchors, H, W, 5+C) -> (B, num_anchors*H*W, 5+C)
        b, na, h, w, nc = layer_pred.shape
        layer_pred_flat = layer_pred.view(b, na * h * w, nc)
        all_predictions.append(layer_pred_flat)

    # Concatenate all layers: (B, total_boxes, 5+C)
    concatenated_preds = torch.cat(all_predictions, dim=1)

    print(f"  Detection predictions shape: {concatenated_preds.shape}")

    nms_results = DetectionHelper.non_max_suppression(
        concatenated_preds,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=100
    )

    print(f"✓ NMS completed")
    print(f"  Input batch size: {batch_size}")
    print(f"  NMS results length: {len(nms_results)}")
    num_detections = sum([r.shape[0] if r is not None else 0 for r in nms_results])
    print(f"  Total detections after NMS: {num_detections}")

    # ========================================================================
    # TEST 2: Detection Metrics (AP, mAP)
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 2: Detection Metrics (AP/mAP)")
    print("-"*60)

    # Use NMS results as detections
    detections_list = nms_results

    # Get ground truths from batch
    # batch["detections"] has shape (num_targets, 6): [batch_idx, class, x, y, w, h]
    ground_truths_list = []
    for i in range(batch_size):
        # Filter ground truths for this image
        mask = batch["detections"][:, 0] == i
        img_targets = batch["detections"][mask]

        if img_targets.shape[0] > 0:
            # Convert from (class, x, y, w, h) to (x1, y1, x2, y2, class)
            boxes_xywh = img_targets[:, 2:6]  # x, y, w, h
            boxes_xyxy = DetectionHelper.xywh2xyxy(boxes_xywh)
            classes = img_targets[:, 1:2]  # class
            gts = torch.cat([boxes_xyxy, classes], dim=1)
            ground_truths_list.append(gts)
        else:
            ground_truths_list.append(None)

    ap_results = DetectionMetrics.calculate_ap(
        detections_list,
        ground_truths_list,
        iou_threshold=0.5,
        num_classes=10
    )

    print(f"✓ AP calculation completed")
    print(f"  mAP: {ap_results['mAP']:.4f}")
    print(f"  AP per class (first 3): ")
    for i in range(3):
        print(f"    Class {i}: {ap_results[f'AP_class_{i}']:.4f}")

    # ========================================================================
    # TEST 3: Detection Confusion Matrix
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 3: Detection Confusion Matrix")
    print("-"*60)

    conf_matrix = DetectionMetrics.confusion_matrix_detection(
        detections_list,
        ground_truths_list,
        num_classes=10,
        iou_threshold=0.5
    )

    print(f"✓ Confusion matrix computed")
    print(f"  Shape: {conf_matrix.shape}")
    print(f"  Total predictions: {conf_matrix.sum()}")

    # ========================================================================
    # TEST 4: Segmentation Metrics
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 4: Segmentation Metrics")
    print("-"*60)

    # Use actual model segmentation predictions
    # drivable_segmentation_predictions has shape (B, num_classes, H, W)
    drivable_preds = outputs.drivable_segmentation_predictions
    num_seg_classes = drivable_preds.shape[1]

    # Get predicted class for each pixel: (B, H, W)
    predictions = torch.argmax(drivable_preds, dim=1)

    # Get ground truth drivable masks from batch
    # Assuming batch has drivable_masks key
    if "drivable_masks" in batch and batch["drivable_masks"] is not None:
        targets = batch["drivable_masks"].long()
    else:
        # Fallback: use predictions (just for testing)
        targets = predictions.clone()

    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Number of segmentation classes: {num_seg_classes}")

    # IoU
    iou_results = SegmentationMetrics.compute_iou(
        predictions, targets, num_classes=num_seg_classes
    )
    print(f"✓ IoU calculated")
    print(f"  mIoU: {iou_results['mIoU']:.4f}")
    for i in range(num_seg_classes):
        print(f"    Class {i} IoU: {iou_results[f'IoU_class_{i}']:.4f}")

    # Dice coefficient
    dice_results = SegmentationMetrics.compute_dice(
        predictions, targets, num_classes=num_seg_classes
    )
    print(f"✓ Dice coefficient calculated")
    print(f"  mDice: {dice_results['mDice']:.4f}")

    # Pixel accuracy
    pixel_acc = SegmentationMetrics.pixel_accuracy(predictions, targets)
    print(f"✓ Pixel accuracy: {pixel_acc:.4f}")

    # Segmentation confusion matrix
    seg_conf_matrix = SegmentationMetrics.confusion_matrix_segmentation(
        predictions, targets, num_classes=num_seg_classes
    )
    print(f"✓ Segmentation confusion matrix computed")
    print(f"  Shape: {seg_conf_matrix.shape}")

    # ========================================================================
    # TEST 5: Visualization Functions
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 5: Visualization Functions")
    print("-"*60)

    # Test colorize mask using actual predictions
    mask = predictions[0].cpu().numpy()
    colored_mask = SegmentationHelper.colorize_mask(mask)
    print(f"✓ Mask colorization works")
    print(f"  Input shape: {mask.shape}, Output shape: {colored_mask.shape}")

    # Test draw bboxes using actual detections from first image
    # Convert batch image from tensor to numpy
    first_image = batch["images"][0].cpu().permute(1, 2, 0).numpy()
    first_image = (first_image * 255).astype(np.uint8)

    if nms_results[0] is not None and nms_results[0].shape[0] > 0:
        # Get boxes, labels, and scores from NMS results
        # NMS output format: [x1, y1, x2, y2, obj_conf, class_conf, class]
        det = nms_results[0].cpu().numpy()
        boxes = det[:, :4]
        scores = det[:, 4] * det[:, 5]  # obj_conf * class_conf
        labels = det[:, 6].astype(int)
    else:
        # Fallback to dummy boxes if no detections
        boxes = np.array([[100, 100, 200, 200]])
        labels = np.array([0])
        scores = np.array([0.5])

    class_names = [f"class_{i}" for i in range(10)]

    image_with_boxes = Visualizer.draw_bboxes(
        first_image, boxes, labels=labels, scores=scores, class_names=class_names
    )
    print(f"✓ Bbox drawing works")
    print(f"  Drew {len(boxes)} boxes on actual batch image")

    # Test plot metrics (without displaying)
    metrics_dict = {
        'loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'mAP': [0.3, 0.4, 0.5, 0.6, 0.65]
    }
    Visualizer.plot_metrics(metrics_dict, title="Test Metrics", save_path="test_metrics.png")
    print(f"✓ Metrics plotting works (saved to test_metrics.png)")

    # Test confusion matrix plot
    Visualizer.plot_confusion_matrix(
        conf_matrix[:5, :5],  # Use subset for visualization
        class_names=class_names[:5],
        title="Test Confusion Matrix",
        save_path="test_confusion_matrix.png"
    )
    print(f"✓ Confusion matrix plotting works (saved to test_confusion_matrix.png)")

    print("\n" + "="*60)
    print("All evaluation helper tests passed! ✓")
    print("="*60)

if __name__ == "__main__":

    print("\n" + "="*60)
    print("Running Integration Tests")
    print("="*60)

    # Test 1: Model integration with dataset
    test_integration()

    # Test 2: Evaluation helpers
    # test_evaluation_helpers()