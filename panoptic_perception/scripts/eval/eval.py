import os
from tqdm import tqdm
from typing import List, Optional
from collections import defaultdict

import torch

from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
from panoptic_perception.dataset.enums import BDD100KClassesReduced

from panoptic_perception.models.models import YOLOP, PanopticModelOutputs
from panoptic_perception.models.utils import WeightsManager
from panoptic_perception.utils.detection_utils import DetectionHelper
from panoptic_perception.utils.segmentation_utils import SegmentationUtils
from panoptic_perception.utils.evaluation_helper import DetectionMetrics

from terminaltables import AsciiTable

# Get class names from enum
CLASS_NAMES = [cls.name for cls in BDD100KClassesReduced]

def create_model(model_kwargs:dict):
    
    cfg_path = model_kwargs["cfg_path"]
    device = model_kwargs["device"]
    model_path = model_kwargs["model_path"]
    
    assert os.path.exists(cfg_path), f'{cfg_path} does not exists'
    model = YOLOP(cfg_path)
    
    if model_path and os.path.exists(model_path):

        missing, unexpected, loaded_keys = WeightsManager().load(model, model_path, strict=True)            
        # self.model.load_state_dict(ckpt["model_state"])
        print("=== Weights Loaded ===")
        print(f"Loaded     : {len(loaded_keys)} keys")
        print(f"Missing    : {len(missing)} keys")
        print(f"Unexpected : {len(unexpected)} keys")
        
    else:
        raise FileNotFoundError(
            f'Model Path: {model_path} not found'
        )
    
    device = torch.device(device) if torch.cuda.is_available() and "cuda" in device else torch.device("cpu")    
    model.to(device)      
    return model, device

def create_dataloader(images_dir:str, detection_annotations_dir:str, 
                    segmentation_annotations_dir:str, drivable_annotations_dir:str, 
                    batch_size:int=1, dataset_type:str="eval"):

    dataset = BDD100KDataset({
        "images_dir":images_dir, 
        "detection_annotations_dir":detection_annotations_dir,
        "segmentation_annotations_dir":segmentation_annotations_dir,
        "drivable_annotations_dir":drivable_annotations_dir
    }, dataset_type=dataset_type)

    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=BDDPreprocessor.collate_fn
    )

def initialize_eval_pipeline(model_kwargs:dict, dataset_kwargs:dict):
    
    def check_dir(path):
        return path is not None and os.path.exists(path)
    
    if not check_dir(dataset_kwargs["images_dir"]):
        raise ValueError(
            "Images must be a valid directory"
        )    

    if not any([check_dir(dataset_kwargs["detection_annotations_dir"]),
                check_dir(dataset_kwargs["segmentation_annotations_dir"]),
                check_dir(dataset_kwargs["drivable_annotations_dir"])]):
        raise ValueError(
            "In eval mode, at least one VALID annotation directory must exist "
            "(detection, segmentation, or drivable)."
        )
    
    dataloader = create_dataloader(dataset_kwargs["images_dir"], 
                                dataset_kwargs["detection_annotations_dir"], 
                                dataset_kwargs["segmentation_annotations_dir"],
                                dataset_kwargs["drivable_annotations_dir"], 
                                dataset_type=dataset_kwargs["dataset_type"],
                                batch_size=dataset_kwargs.get("batch_size", 1))
        

    model, device = create_model(model_kwargs)
    
    return model, device, dataloader

def process_detection_outputs(outputs: PanopticModelOutputs, images: torch.Tensor,
                    target_detections: torch.Tensor, image_paths: List[str],
                    dets_by_image:defaultdict, gt_by_image:defaultdict,
                    global_image_idx:int, output_dir: Optional[str] = None,
                    visualize: bool = True) -> int:
    """
    Process model outputs, convert to proper format, and optionally visualize.

    Args:
        outputs: Model outputs containing detection predictions
        images: Batch of images, shape (B, C, H, W)
        target_detections: Ground truth detections, shape (N, 6) [batch_idx, class, x, y, w, h]
                          where x, y, w, h are normalized [0, 1]
        image_paths: List of image paths for saving visualizations
        output_dir: Directory to save visualizations (if None, uses 'eval_outputs')
        visualize: Whether to create and save visualizations

    Returns:
        List of tuples (predictions, targets) for each image in batch
    """

    if outputs.detection_predictions is None:
        return

    detection_preds = outputs.detection_predictions
    batch_size, _, image_h, image_w = images.shape

    # Concatenate predictions from all detection layers
    batch_predictions = []
    for layer_pred in detection_preds:
        b, na, h, w, nc = layer_pred.shape
        layer_pred_flat = layer_pred.view(b, na * h * w, nc)
        batch_predictions.append(layer_pred_flat)

    concatenated_preds = torch.cat(batch_predictions, dim=1)

    # Apply NMS - returns list of tensors, one per image
    # Each tensor has shape (num_detections, 6) [x1, y1, x2, y2, confidence, class_id]
    nms_results = DetectionHelper.non_max_suppression(
        concatenated_preds,
        conf_threshold=0.001,
        iou_threshold=0.45,
        max_detections=300
    )

    # Process each image in the batch
    for i in range(batch_size):
        # Get predictions for this image (already in xyxy format from NMS)
        img_predictions = nms_results[i]  # Shape: (N, 6) or None
        
        if img_predictions is not None:
            dets_by_image[global_image_idx] = img_predictions
        else:
            dets_by_image[global_image_idx] = None

        # Process ground truth detections for this image
        img_targets = None
        if target_detections is not None:
            mask = target_detections[:, 0] == i
            batch_targets = target_detections[mask]

            if batch_targets.shape[0] > 0:
                # Extract class and normalized xywh coordinates
                classes = batch_targets[:, 1:2]  # Shape: (M, 1)
                boxes_xywh = batch_targets[:, 2:6].clone()  # Shape: (M, 4)

                # Scale normalized coordinates to pixel coordinates
                boxes_xywh[:, [0, 2]] *= image_w  # x, w
                boxes_xywh[:, [1, 3]] *= image_h  # y, h

                # Convert xywh to xyxy format
                boxes_xyxy = DetectionHelper.xywh2xyxy(boxes_xywh)

                # Combine: [x1, y1, x2, y2, class_id]
                img_targets = torch.cat([boxes_xyxy, classes], dim=1)
                gt_by_image[global_image_idx] = img_targets
            else:
                gt_by_image[global_image_idx] = None
        else:
            gt_by_image[global_image_idx] = None

        global_image_idx += 1
        # results.append((img_predictions, img_targets))

        # Visualize if requested
        if visualize:
            if output_dir is None:
                output_dir = "eval_outputs"
            os.makedirs(output_dir, exist_ok=True)

            # Get image filename for saving
            if image_paths and i < len(image_paths):
                img_name = os.path.basename(image_paths[i])
                save_path = os.path.join(output_dir, f"vis_{img_name}")
            else:
                save_path = os.path.join(output_dir, f"vis_batch_{i}.jpg")

            # Visualize detections
            DetectionHelper.visualize_detections(
                image=images[i],
                predictions=img_predictions,
                targets=img_targets,
                class_names=CLASS_NAMES,
                save_path=save_path
            )

    return global_image_idx

def process_drivable_area_outputs(outputs:PanopticModelOutputs, images: torch.Tensor,
                                drivable_area_targets:torch.Tensor, image_paths: List[str],
                                drivable_confusion_matrix:torch.Tensor, num_drivable_classes,
                                output_dir: Optional[str] = None,
                                visualize: bool = True):
    
    """
        images - (bs, 3, h, w)
        drivable_area_targets - (bs, h, w)
    """

    batch_drivable_gts = None
    batch_drivable_preds = None
    
    drivable_preds = torch.argmax(outputs.drivable_segmentation_predictions, dim=1)
    if drivable_area_targets is not None:
        drivable_confusion_matrix += SegmentationUtils._compute_confusion_matrix(
            drivable_preds.cpu(), drivable_area_targets.cpu(), num_drivable_classes
        )

    if not visualize:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if drivable_area_targets is not None:
        batch_drivable_gts = SegmentationUtils.transparent_overlay(
            original_imgs=images,
            masks=drivable_area_targets
        )
        
    batch_drivable_preds = SegmentationUtils.transparent_overlay(
        original_imgs=images,
        masks=drivable_preds
    )    
    
    for image_idx in range(len(batch_drivable_preds)):        
        if image_paths and image_idx < len(image_paths):
            img_name = os.path.basename(image_paths[image_idx])
        else:
            img_name = f'vis_{image_idx}'
        
        if batch_drivable_gts is not None and batch_drivable_gts:
            gt_save_path = os.path.join(output_dir, f'{img_name}_gt.png')
            gt_overlay = batch_drivable_gts[image_idx]
            
            SegmentationUtils.save_overlay_image(
                vis_image=gt_overlay,
                save_path=gt_save_path
            )
            
        pred_save_path = os.path.join(output_dir, f'{img_name}_pred.png')
        pred_overlay = batch_drivable_preds[image_idx]
        
        SegmentationUtils.save_overlay_image(
            vis_image=pred_overlay,
            save_path=pred_save_path
        )

def run_eval_pipeline(model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      device: torch.device,
                      output_dir: str = "eval_outputs",
                      visualize: bool = True,
                      max_visualizations: int = 100):
    """
    Run evaluation pipeline on the dataloader.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run on
        output_dir: Directory to save visualizations
        visualize: Whether to save visualizations
        max_visualizations: Maximum number of images to visualize (to avoid too many files)
    """
    val_iter = tqdm(dataloader, desc='Running Validation')
    model.eval()

    all_results = []
    vis_count = 0
    dets_by_image = defaultdict(None) #image_id -> (num_dets, 6)
    gt_by_image = defaultdict(None) #image_id -> (num_gt, 5)    
    global_image_idx = 0
    drivable_confusion_matrix = None
    num_drivable_classes = 2

    for batch_idx, data_items in enumerate(val_iter):
        # Move data to device
        for k, v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(device)

        with torch.no_grad():
            # Forward pass
            outputs = model(
                data_items["images"],
                targets={
                    "drivable_area_seg": data_items.get("drivable_area_seg"),
                    "lane_seg": data_items.get("segmentation_masks"),
                    "detections": data_items["detections"]
                }
            )

        # Determine if we should visualize this batch
        should_visualize = visualize and vis_count < max_visualizations

        # Process outputs and optionally visualize
        global_image_idx = process_detection_outputs(
            outputs=outputs,
            images=data_items["images"],
            target_detections=data_items.get("detections"),
            image_paths=data_items.get("image_paths", []),
            dets_by_image=dets_by_image,
            gt_by_image=gt_by_image,
            global_image_idx=global_image_idx,
            output_dir=f"{output_dir}/detections",
            visualize=should_visualize
        )

        if outputs.drivable_segmentation_predictions is not None:
            if drivable_confusion_matrix is None:
                drivable_confusion_matrix = torch.zeros(
                    num_drivable_classes, num_drivable_classes, dtype=torch.int64
                )
            
            process_drivable_area_outputs(
                outputs=outputs,
                images=data_items["images"],
                drivable_area_targets=data_items.get("drivable_area_seg"),
                image_paths=data_items.get("image_paths", []),
                drivable_confusion_matrix=drivable_confusion_matrix,
                num_drivable_classes=num_drivable_classes,
                output_dir=f"{output_dir}/drivable_area",
                visualize=should_visualize
            )

        if should_visualize:
            vis_count = global_image_idx

    print(f"\nEvaluation complete. Processed {global_image_idx} images.")
    if visualize:
        print(f"Saved {min(vis_count, max_visualizations)} visualizations to '{output_dir}/'")

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
    print("\nDetection Metrics (AP@0.5):")
    print(ap_table_string)
    
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
    print(f"\nDetection Metrics @{stats_iou_threshold}:")
    print(stats_table_string)
    
    if drivable_confusion_matrix is not None:
        drivable_iou, drivable_dice = SegmentationUtils._compute_metrics_from_confusion_matrix(
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

        print("\nDrivable Segmentation Metrics:")
        print(drivable_table_string)
    
if __name__ == "__main__":

    model_kwargs = {
        "cfg_path": "panoptic_perception/configs/models/yolo-detection.cfg",
        "device": "cuda:0",
        "model_path": "checkpoints/yolop-detection-5060Ti/best_model.pt"
    }

    dataset_kwargs = {
        "images_dir": "panoptic_perception/BDD100k/100k/100k",
        "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
        "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
        "dataset_type": "val",
        "batch_size": 4
    }

    model, device, dataloader = initialize_eval_pipeline(
        model_kwargs, dataset_kwargs
    )

    # Run evaluation with visualization
    run_eval_pipeline(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir="eval_outputs-detections-1",
        visualize=True,
        max_visualizations=50  # Limit to 50 images to avoid too many files
    )
