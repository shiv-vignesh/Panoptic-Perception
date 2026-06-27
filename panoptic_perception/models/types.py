import torch
from typing import Tuple, Optional, Any, Dict

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PanopticModelOutputs:

    detection_logits: List[torch.Tensor] = None
    drivable_segmentation_logits: torch.Tensor = None
    lane_segmentation_logits: torch.Tensor = None #unused currently
    lane_detection_logits: torch.Tensor = None
    lane_seg_logits: torch.Tensor = None

    detection_predictions: List[torch.Tensor] = None
    drivable_segmentation_predictions: torch.Tensor = None
    lane_segmentation_predictions: torch.Tensor = None
    lane_detection_predictions : torch.Tensor = None

    detection_loss: torch.Tensor = None
    drivable_segmentation_loss: torch.Tensor = None
    lane_segmentation_loss: torch.Tensor = None
    lane_detection_loss: torch.Tensor = None
    lane_detection_loss_items: dict = None
    defogging_loss: torch.Tensor = None
    depth_reconstruction_loss: torch.Tensor = None

    # ---- ATSS - Detection Head -----
    # List[torch.Tensor]: pre-computed (x, y) center locations per grid
    anchor_proposals: List[torch.Tensor] = None
    anchor_cxcy: List[torch.Tensor] = None
    anchor_wh: List[torch.Tensor] = None
    anchor_strides: List[torch.Tensor] = None
    proposal_shape: Tuple[Tuple[int, int], ...] = None

    # ---- YoloV8 Anchor-Free Detection Head Outputs ----
    bbox_logits_raw: torch.Tensor = None
    cls_logits_raw: torch.Tensor = None
    anchor_points: torch.Tensor = None
    strides_v8: torch.Tensor = None

    # ---- TeacherFusion auxiliary depth reconstruction ----
    depth_reconstruction: Optional["DepthReconstructionLossItems"] = None

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()


@dataclass
class DetectionLossItems:
    # Common
    detection_logits: Optional[List[torch.Tensor]] = None       # List[Tensor] for YOLO/ATSS, tuple for v8
    targets: Optional[torch.Tensor] = None

    # YOLO-classic specific (Detect head fills these)
    num_layers: Optional[int] = None
    anchors: Optional[torch.Tensor] = None
    stride: Optional[torch.Tensor] = None

    # ATSS specific (Detect head fills when ATSS-aware)
    anchor_proposals: Optional[List[torch.Tensor]] = None
    anchor_cxcy: Optional[List[torch.Tensor]] = None
    anchor_wh: Optional[List[torch.Tensor]] = None
    anchor_strides: Optional[List[torch.Tensor]] = None

    # YOLOv8 specific (DetectV8 head fills these)
    pred_scores_logits: Optional[torch.Tensor] = None
    pred_distri_logits: Optional[torch.Tensor] = None
    anchor_points: Optional[torch.Tensor] = None
    strides_v8: Optional[torch.Tensor] = None

    image_size: Optional[Tuple[int, int]] = None
    batch_size: Optional[int] = None


@dataclass
class DrivableSegmentationLossItems:

    drivable_segmentation_logits: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None


@dataclass
class DepthReconstructionLossItems:
    # predictions: {task_name: {"full_res": (B,1,H,W), "<tap_idx>": (B,1,h,w), ...}}
    # target: (B, 1, H, W) reference depth in [0, 1].
    predictions: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    target: Optional[torch.Tensor] = None


@dataclass
class LaneDetectionLossItems:

    lane_seg_logits: Optional[torch.Tensor] = None
    lane_detection_logits: torch.Tensor = None
    targets_detections: Optional[torch.Tensor] = None
    targets_seg_masks: Optional[torch.Tensor] = None
    image_size: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class ModelArchitectureInfo:
    has_anchor_based_detection: bool = False
    has_anchor_free_detection:  bool = False
    has_drivable_seg:           bool = False
    has_lane_seg:               bool = False
    has_lane_det:               bool = False

    @property
    def has_detection(self) -> bool:
        return self.has_anchor_based_detection or self.has_anchor_free_detection

    @property
    def is_anchor_free(self) -> bool:
        # "anchor_free model" = has v8 detect and NOT classic detect
        return self.has_anchor_free_detection and not self.has_anchor_based_detection