import torch

from panoptic_perception.losses.loss_factory import LossFactory
from panoptic_perception.models.types import DrivableSegmentationLossItems

@LossFactory.register_loss_function("segmentation-loss")
class SegmentationLoss:

    def __init__(self):

        self._ce_weights = torch.tensor([1.00, 4.91])
        self.segmentation_loss_weights = 1.0
    
    def dice_loss(self, pred_softmax:torch.Tensor, target_onehot:torch.Tensor, smooth=1.0):
        dims = (0, 2, 3)
        intersection = (pred_softmax * target_onehot).sum(dim=dims)
        union = pred_softmax.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
        
    
    def tversky_loss(self, pred_softmax:torch.Tensor, target_onehot:torch.Tensor, alpha=0.2, beta=0.8, 
                    smooth=1.0, gamma=0.75):
        dims = (0, 2, 3)                                                                 
        tp = (pred_softmax * target_onehot).sum(dim=dims)                                
        fp = (pred_softmax * (1 - target_onehot)).sum(dim=dims)                          
        fn = ((1 - pred_softmax) * target_onehot).sum(dim=dims)                          
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)                 

        return (1.0 - tversky.mean()) ** gamma 
    
    def compute_segmentation_loss(self, predictions, targets, dice_weight=0.7, ce_weight=0.3):
        """
        Compute combined CE + Dice segmentation loss.

        Args:
            predictions (torch.Tensor): Raw logits (B, C, H, W)
            targets (torch.Tensor): Class indices (B, H, W)
            dice_weight: Weight for Dice loss
            ce_weight: Weight for CE loss

        Returns:
            torch.Tensor: Combined loss
        """
        bs, c, h, w = predictions.shape
        bs, h_t, w_t = targets.shape

        assert h == h_t and w == w_t, "Prediction and target spatial dimensions must match."

        ce_loss = torch.nn.functional.cross_entropy(
            input=predictions.view(bs, c, -1),
            target=targets.view(bs, -1),
            reduction="mean",
            weight=self._ce_weights.to(predictions.device)
        )

        # Dice loss (expects softmax probabilities + one-hot targets)
        pred_softmax = torch.softmax(predictions, dim=1)
        target_onehot = torch.nn.functional.one_hot(targets, num_classes=c)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        t_loss = self.tversky_loss(pred_softmax, target_onehot)

        loss = ce_weight * ce_loss + dice_weight * t_loss
        loss = self.segmentation_loss_weights * loss
        
        loss_items = {
            "ce_loss": ce_loss,
            "t_loss": t_loss
        }

        return loss, loss_items
    
    def __call__(self, loss_items: DrivableSegmentationLossItems):
        
        return self.compute_segmentation_loss(
            loss_items.drivable_segmentation_logits,
            loss_items.targets
        )