import torch
import torch.nn.functional as F

from panoptic_perception.losses.loss_factory import LossFactory
from panoptic_perception.models.types import DepthReconstructionLossItems


@LossFactory.register_loss_function("depth-reconstruction")
class DepthReconstructionLoss:
    def __init__(self, loss_type="smooth_l1", aux_weights=None, full_res_weight=1.0):
        self.loss_type = loss_type
        self.full_res_weight = float(full_res_weight)
        self.aux_weights = {int(k): float(v) for k, v in (aux_weights or {}).items()}
        self._loss_fn = F.smooth_l1_loss if loss_type == "smooth_l1" else F.l1_loss

    def _weight_for(self, scale_key: str) -> float:
        if scale_key == "full_res":
            return self.full_res_weight
        # scale_key format: "tap_<int>"
        tap_id = int(scale_key.split("_", 1)[1])
        return self.aux_weights.get(tap_id, 0.3)

    def __call__(self, loss_items: DepthReconstructionLossItems):
        target = loss_items.target
        total = None
        per_scale = {}

        for task_name, preds in loss_items.predictions.items():
            for scale_key, pred in preds.items():
                tgt = target
                if pred.shape[-2:] != target.shape[-2:]:
                    tgt = F.interpolate(target, size=pred.shape[-2:], mode="area")

                scale_loss = self._loss_fn(pred, tgt)
                weighted = self._weight_for(scale_key) * scale_loss

                total = weighted if total is None else total + weighted
                per_scale[f"{task_name}/{scale_key}"] = scale_loss.detach()

        if total is None:
            raise ValueError("DepthReconstructionLoss got empty predictions dict")

        return total, {"depth_recon_total": total.detach(), **per_scale}
