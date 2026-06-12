"""
End-to-end test for LaneDetect module with real BDD100K data.
Visualizes detections, drivable area, and lane predictions for the full batch.

Usage:
    python -m panoptic_perception.tests.test_lane_detect
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
from panoptic_perception.dataset.enums import BDD100KClassesReduced
from panoptic_perception.dataset.utils import LANE_VIS_COLORS
from panoptic_perception.models.models import YOLOP
from panoptic_perception.utils.detection_utils import DetectionHelper

SAVE_DIR = "visualizations/lane_detect_test"
BATCH_SIZE = 4
CFG_PATH = "panoptic_perception/configs/models/yolo-lane-detect.cfg"

BDD_CLASS_NAMES = [e.name.lower() for e in BDD100KClassesReduced]

DATASET_KWARGS = {
    "images_dir": "panoptic_perception/BDD100k/100k/100k",
    "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
    "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
    "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
    "preprocessor_kwargs": {
        "image_resize": (640, 640),
        "original_image_size": (720, 1280)
    }
}


def draw_detections(img, nms_dets, class_names):
    """Draw NMS detection boxes on image. nms_dets: (N, 6) [x1,y1,x2,y2,conf,cls]"""
    if nms_dets is None or len(nms_dets) == 0:
        return img

    H, W = img.shape[:2]
    for det in nms_dets:
        x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
        # detection coords are normalized [0,1] — scale to pixels
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        cls_id = int(cls_id)
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return img


def draw_drivable(img, drivable_pred, alpha=0.4):
    """Overlay drivable area prediction on image.
    drivable_pred: (2, H, W) softmax probabilities or (H, W) class indices.
    """
    if drivable_pred is None:
        return img

    if drivable_pred.ndim == 3:
        mask = torch.argmax(drivable_pred, dim=0).cpu().numpy()
    else:
        mask = drivable_pred.cpu().numpy()

    overlay = img.copy()
    overlay[mask == 1] = [0, 200, 0]    # direct drivable: green
    overlay[mask == 2] = [0, 200, 200]  # alternative: yellow
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


def draw_lanes(img, lane_preds, conf_threshold=0.3):
    """Draw lane predictions on image.
    lane_preds: (192, 78) — last refinement stage output for one image.
    Format per prior: [cls0, cls1, start_y, start_x, theta, length, x0...x71]
    """
    if lane_preds is None:
        return img

    H, W = img.shape[:2]
    n_offsets = lane_preds.shape[1] - 6  # 72
    prior_ys = np.linspace(1.0, 0.0, n_offsets)

    # cls_logits are [col0=invalid, col1=valid]
    cls_scores = torch.softmax(lane_preds[:, :2], dim=1)
    valid_scores = cls_scores[:, 1]  # probability of being a valid lane

    for i in range(lane_preds.shape[0]):
        if valid_scores[i].item() < conf_threshold:
            continue

        xs = lane_preds[i, 6:].cpu().numpy()  # normalized x-coords
        length = lane_preds[i, 5].item()
        start_y = lane_preds[i, 2].item()

        color = LANE_VIS_COLORS[i % len(LANE_VIS_COLORS)]
        pts = []
        for j in range(n_offsets):
            y_norm = prior_ys[j]
            # only draw within the lane's extent
            if y_norm > start_y + 0.05:
                continue
            if xs[j] < -1e3 or xs[j] > 1e3:
                continue
            x_pix = int(np.clip(xs[j] * (W - 1), 0, W - 1))
            y_pix = int(np.clip(y_norm * (H - 1), 0, H - 1))
            pts.append((x_pix, y_pix))

        for k in range(len(pts) - 1):
            cv2.line(img, pts[k], pts[k + 1], color, 2)

    return img


def draw_lane_targets(img, lane_targets, lane_categories=None):
    """Draw ground truth lane annotations.
    lane_targets: (max_lanes, 78)
    """
    if lane_targets is None:
        return img

    H, W = img.shape[:2]
    n_offsets = lane_targets.shape[1] - 6

    prior_ys = np.linspace(1.0, 0.0, n_offsets)

    for i in range(lane_targets.shape[0]):
        if lane_targets[i, 0].item() < 0.5:  # not valid
            continue

        xs = lane_targets[i, 6:].cpu().numpy()
        cat_idx = int(lane_categories[i].item()) if lane_categories is not None and lane_categories[i].item() >= 0 else 0
        color = LANE_VIS_COLORS[cat_idx % len(LANE_VIS_COLORS)]

        pts = []
        for j in range(n_offsets):
            if xs[j] > -1e4:
                x_pix = int(xs[j] * (W - 1))
                y_pix = int(prior_ys[j] * (H - 1))
                pts.append((x_pix, y_pix))

        for k in range(len(pts) - 1):
            cv2.line(img, pts[k], pts[k + 1], color, 3)

    return img


def visualize_full_batch(batch, model_outputs, nms_results, save_dir, tag=""):
    """Visualize all images in the batch with detections, drivable, and lanes."""
    os.makedirs(save_dir, exist_ok=True)

    images = batch["images"]
    batch_size = images.shape[0]
    H, W = images.shape[2], images.shape[3]

    # Lane predictions: list of (bs, 192, 78) per stage — take last stage
    lane_logits = model_outputs.lane_detection_logits
    lane_preds_last = lane_logits[-1] if lane_logits is not None else None

    # Drivable predictions
    drivable_preds = model_outputs.drivable_segmentation_predictions

    # Ground truth
    gt_detections = batch.get("detections")
    gt_lanes = batch.get("lanes")
    gt_lane_cats = batch.get("lane_categories")
    gt_drivable = batch.get("drivable_area_seg")

    for b in range(batch_size):
        img = images[b].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8).copy()

        # --- Panel 1: Predictions (det + drivable + lanes) ---
        pred_img = img.copy()
        pred_img = draw_drivable(pred_img, drivable_preds[b] if drivable_preds is not None else None)
        pred_img = draw_detections(pred_img, nms_results[b], BDD_CLASS_NAMES)
        pred_img = draw_lanes(pred_img, lane_preds_last[b] if lane_preds_last is not None else None)

        # --- Panel 2: Ground truth (det + drivable + lanes) ---
        gt_img = img.copy()

        # GT drivable overlay
        if gt_drivable is not None:
            gt_img = draw_drivable(gt_img, gt_drivable[b])

        # GT detection boxes
        if gt_detections is not None:
            gt_mask = gt_detections[:, 0] == b
            gt_for_img = gt_detections[gt_mask]
            for row in gt_for_img:
                _, cls, xc, yc, w, h = row.cpu().numpy()
                x1 = int((xc - w / 2) * W)
                y1 = int((yc - h / 2) * H)
                x2 = int((xc + w / 2) * W)
                y2 = int((yc + h / 2) * H)
                label = BDD_CLASS_NAMES[int(cls)] if int(cls) < len(BDD_CLASS_NAMES) else str(int(cls))
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(gt_img, f"GT:{label}", (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # GT lanes
        if gt_lanes is not None:
            gt_lane_cat = gt_lane_cats[b] if gt_lane_cats is not None else None
            gt_img = draw_lane_targets(gt_img, gt_lanes[b], gt_lane_cat)

        # --- Compose side-by-side ---
        cv2.putText(pred_img, "Predictions", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(gt_img, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        combined = np.concatenate([pred_img, gt_img], axis=1)

        fname = f"{save_dir}/{tag}batch_img_{b}.png"
        cv2.imwrite(fname, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {fname}")


def main():
    print("=" * 60)
    print("LaneDetect End-to-End Test (Real Data)")
    print("=" * 60)

    # --- Dataset ---
    dataset = BDD100KDataset(DATASET_KWARGS, dataset_type='val')
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=BDDPreprocessor.collate_fn
    )
    print(f"Dataset: {len(dataset)} images, batch_size={BATCH_SIZE}")

    # --- Model ---
    model = YOLOP(CFG_PATH)
    model.eval()
    print(f"Model: YOLOP ({CFG_PATH})")
    print(f"  Modules: {len(model.module_list)}")
    print(f"  Detection head: {model.detection_head_idx}")
    print(f"  Drivable seg head: {model.segmentation_head_idx}")
    print(f"  Lane seg head: {model.lane_segmentation_head_idx}")

    # --- Forward pass ---
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"  images: {batch['images'].shape}")
    if batch.get('detections') is not None:
        print(f"  detections: {batch['detections'].shape}")
    if batch.get('lanes') is not None:
        print(f"  lanes: {batch['lanes'].shape}")
    if batch.get('drivable_area_seg') is not None:
        print(f"  drivable_area_seg: {batch['drivable_area_seg'].shape}")

    with torch.no_grad():
        outputs = model(batch["images"])

    print(f"\nModel outputs:")
    if outputs.detection_predictions is not None:
        det_pred = outputs.detection_predictions
        if isinstance(det_pred, list):
            print(f"  detection_predictions: list of {len(det_pred)} scales")
        else:
            print(f"  detection_predictions: {det_pred.shape}")
    if outputs.drivable_segmentation_predictions is not None:
        print(f"  drivable_seg_predictions: {outputs.drivable_segmentation_predictions.shape}")
    if outputs.lane_detection_logits is not None:
        print(f"  lane_detection_logits: {len(outputs.lane_detection_logits)} stages, last: {outputs.lane_detection_logits[-1].shape}")

    # --- NMS on detections ---
    det_preds = outputs.detection_predictions
    if isinstance(det_preds, list):
        # anchor-based: concat multi-scale predictions
        all_preds = []
        for layer_pred in det_preds:
            b, na, h, w, nc = layer_pred.shape
            all_preds.append(layer_pred.view(b, na * h * w, nc))
        det_concat = torch.cat(all_preds, dim=1)
    else:
        det_concat = det_preds

    nms_results = DetectionHelper.non_max_suppression(
        det_concat,
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=100
    )

    total_dets = sum(r.shape[0] if r is not None else 0 for r in nms_results)
    print(f"\nNMS: {total_dets} detections across {BATCH_SIZE} images")

    # --- Visualize full batch ---
    print(f"\nVisualizing full batch...")
    visualize_full_batch(batch, outputs, nms_results, SAVE_DIR)

    print(f"\nDone. All visualizations saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
