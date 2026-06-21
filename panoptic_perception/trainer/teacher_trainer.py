from typing import Optional

import torch

from panoptic_perception.models.teacher_model import TeacherFusion
from panoptic_perception.models.types import PanopticModelOutputs
from panoptic_perception.models.utils import WeightsManager
from panoptic_perception.trainer.trainer_refactor import Trainer
from panoptic_perception.utils.logger import Logger


def load_yolop_into_backbones(
    teacher_model: TeacherFusion,
    image_backbone_path: str,
    depth_backbone_path: Optional[str] = None,
    logger: Optional[Logger] = None,
) -> None:
    """
    Initialize image_model and depth_model inside a TeacherFusion from one or two
    YOLOP checkpoint files. Uses WeightsManager (same loader as CheckpointCallback)
    so envelope unwrap, anchors_grid→anchors backward-compat, and key remapping
    behavior match the rest of the trainer.

    Always non-strict so:
      - The detection head loads if present (good)
      - Fusion blocks (new) stay at init (expected)
      - First-conv channel mismatches (e.g. depth in_channels=1 case) are skipped
        rather than raised, leaving that conv at init

    If `depth_backbone_path` is None, the image checkpoint is loaded into both
    backbones — the common "warm-start from RGB YOLOP" path.

    Note: WeightsManager.load uses map_location="cpu" internally; the parameters
    are moved back to the target_model's device on assignment.
    """
    wm = WeightsManager(verbose=False)

    def _load(target_model, path: str, tag: str):
        result = wm.load(target_model, path, strict=False, key_prefix=None)
        if result is None:
            # WeightsManager prints/returns None when the path doesn't exist.
            if logger is not None:
                logger.log_message(f"[teacher init] {tag}: checkpoint not found at {path}; skipping")
            return
        missing, unexpected, loaded_keys = result
        if logger is not None:
            logger.log_message(
                f"[teacher init] {tag}: loaded from {path} "
                f"(loaded={len(loaded_keys)}, missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if missing:
                logger.log_message(f"[teacher init] {tag} missing keys (first 10): {missing[:10]}")
            if unexpected:
                logger.log_message(f"[teacher init] {tag} unexpected keys (first 10): {unexpected[:10]}")

    _load(teacher_model.image_model, image_backbone_path, tag="image_backbone")

    depth_path = depth_backbone_path or image_backbone_path
    _load(teacher_model.depth_model, depth_path, tag="depth_backbone")


class TeacherTrainer(Trainer):
    """
    Trainer for the privileged-information teacher: TeacherFusion consumes
    (clean_image, depth) and produces a PanopticModelOutputs via image_model's
    forward_task_head. Shares everything with the base Trainer except the
    forward dispatch (which needs the depth input).

    `targets` drops `clean_images` since the teacher path has no defog branch.
    """

    def _build_targets(self, data_items: dict) -> dict:
        return {
            "drivable_area_seg": data_items.get("drivable_area_seg"),
            "lane_seg":          data_items.get("segmentation_masks"),
            "detections":        data_items["detections"],
            "lanes_detections":  data_items.get("lanes_detections"),
            "lane_seg_masks":    data_items.get("lane_seg_masks"),
        }

    def _forward_model(self, data_items: dict) -> PanopticModelOutputs:
        depth = data_items["depth_maps"]

        # depth_maps from FoggyBDDPreprocessor is (B, H, W) single-channel.
        # The depth_backbone is currently built from the same 3-channel cfg as
        # the image_backbone — replicate to (B, 3, H, W) until the cfg parser
        # honors depth_in_channels=1 at construction time.
        if depth.dim() == 3:
            depth = depth.unsqueeze(1).repeat(1, 3, 1, 1).float()
        elif depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth.repeat(1, 3, 1, 1).float()

        return self.model(
            data_items["images"],
            depth,
            targets=self._build_targets(data_items),
        )
