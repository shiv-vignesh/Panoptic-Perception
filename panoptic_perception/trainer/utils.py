from dataclasses import dataclass
from typing import Any, Iterable, Dict, List

from collections import defaultdict

import torch
import numpy as np

from panoptic_perception.models.utils import PanopticModelOutputs

def listify(p: Any):

    if p is None:
        p = []
    if not isinstance(p, Iterable):
        p = [p]

    return p

def to_numpy(x: Any) -> np.ndarray:
    """Convert whatever to numpy array"""
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")


def to_tensor(x: Any, dtype=None) -> torch.Tensor:
    """Convert whatever to torch Tensor"""
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    else:
        raise ValueError("Unsupported input type" + str(type(x)))

@dataclass
class EvalMetrics:

    # ---- Detection Metrics ----
    ap_per_class : Dict[str, float] = None # defaultdict(float)
    stats_per_class : Dict[str, float] = None # defaultdict(lambda : defaultdict(float))

    # ---- Drivable Metrics ----
    drivable_metrics : Dict[str, float] = None

    # ---- Lane Detection Metrics ----
    lane_detection_metrics : Dict[str, float] = None

    metric_prefix : str = None # [val_clean, val_foggy, test]

    def reset(self):
        self.ap_per_class = None
        self.stats_per_class = None
        
@dataclass
class EvalBatchContext:
    
    cur_eval_model_outputs : PanopticModelOutputs = None

    cur_eval_image_h : int = None
    cur_eval_image_w : int = None        
    cur_eval_image_paths : list = None

    cur_eval_images : torch.Tensor = None
    cur_eval_gt_detections : torch.Tensor = None
    cur_eval_gt_drivable_area_seg : torch.Tensor = None
    cur_eval_gt_lane_seg : torch.Tensor = None
    cur_eval_gt_lane_detections : torch.Tensor = None

    ap_table_data : List[List[str]] = None
    stats_table_data : List[List[str]] = None

    stats_iou_threshold : float = None
    wandb_ap_data : list = None

    drivable_table_data : dict = None
    wandb_drivable_data : list = None    
