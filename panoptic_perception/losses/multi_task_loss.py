import torch
from typing import Dict, Any, Union, Tuple

from collections import defaultdict

import warnings
warnings.simplefilter("once", UserWarning)

from panoptic_perception.models.types import DetectionLossItems, DrivableSegmentationLossItems

from panoptic_perception.losses.loss_factory import LossFactory
from panoptic_perception.models.types import LaneDetectionLossItems

class MultiTaskLoss:
    def __init__(self, loss_kwargs:Dict[str, Any]):
        super(MultiTaskLoss, self).__init__()

        self.task_losses = defaultdict()

        if not loss_kwargs or not isinstance(loss_kwargs, dict):
            raise ValueError(
                f"Expected loss_kwargs to be non-empty or of type dict, got: {type(loss_kwargs)}"
            )
        
        if "loss_weights" not in loss_kwargs:
            warnings.warn(
                f"loss_weights missing in loss_kwargs: {loss_kwargs.keys()}, fallback to default weights",
                category=UserWarning,
                stacklevel=2
            )

        self.loss_weights = loss_kwargs.get("loss_weights", {
            "detection": 1.0,
            "drivable_segmentation": 1.0,
            "lane_detection":1.0,
            "lane_segmentation": 0.0
        })

        for task_name, cfg in loss_kwargs.items():
            # Skip the weights block and any doc-only keys (`_description`, `_note`, …).
            # Matches the convention used elsewhere in the configs — leading-underscore
            # keys are documentation/metadata, not task entries.
            if "loss_weights" in task_name or task_name.startswith("_"):
                continue

            loss_func = LossFactory.build(cfg)
            self.task_losses[task_name] = loss_func

        self.task_losses_funcs = set(self.task_losses.keys())

    def __call__(self, loss_items:Dict[str, Union[DetectionLossItems, 
                    DrivableSegmentationLossItems, LaneDetectionLossItems]],
                    device:torch.device) -> Tuple[dict, dict]:

        model_tasks = set(loss_items.keys())

        common = self.task_losses_funcs & model_tasks
        if not common:
            raise ValueError(
                  f"Loss functions registered for {sorted(common)} " 
                  f"but model produced no outputs for these tasks. "
                  f"Available outputs: {sorted(model_tasks)}. "
                  f"Check head wiring in the model or remove these losses from "
                  f"the config. "
                )

        multi_task_loss = defaultdict(lambda : torch.zeros(1, device=device))
        multi_task_loss_items = defaultdict(dict)

        for task_name, loss_item in loss_items.items():

            if task_name not in self.task_losses:
                warnings.warn(
                    f"Model produces output for task '{task_name}' but no loss is "
                    f"registered. This task will not contribute to training. "
                    f"If intentional (e.g., frozen head), ignore this warning. "
                    f"Otherwise add '{task_name}' to your loss config.",
                    UserWarning,
                    stacklevel=3,
                )

                continue
            
            if loss_item is None:
                warnings.warn(
                    f"Model produces NoneType output for task '{task_name}' "
                    f"Cannot compute loss for the '{task_name}'. "
                    f"If intentional, ignore this warning. "
                )
                continue
            
            task_loss, task_loss_dict = self.task_losses[task_name](loss_item)
            multi_task_loss[task_name] = task_loss * self.loss_weights.get(task_name, 1.0)
            multi_task_loss_items[task_name] = task_loss_dict

        return multi_task_loss, multi_task_loss_items