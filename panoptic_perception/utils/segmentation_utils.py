import torch
import numpy as np
import cv2

from enum import Enum
from collections import defaultdict

class DriveableAreaColor(Enum):
    BACKGROUND = (0, (0, 0, 0))
    DRIVABLE = (1, (0, 255, 0))
    ALTERNATIVE = (2, (255, 255, 0))

    def __init__(self, class_id:int, color):
        self.class_id = class_id
        self.color = color

    @classmethod
    def from_id(cls, class_id: int):
        for member in cls:
            if member.class_id == class_id:
                return member
        return None

    @classmethod
    def from_label(cls, label: str):
        try:
            return cls[label.upper()]
        except KeyError:
            return None

class SegmentationUtils:
    @staticmethod
    def create_segmentation_mask(predictions, threshold=0.5):
        """
        Create a binary segmentation mask from model predictions.

        Args:
            predictions (torch.Tensor): The raw output from the segmentation head of the model.
            threshold (float): Threshold to convert probabilities to binary mask.

        Returns:
            torch.Tensor: Binary segmentation mask.
        """
        # Assuming predictions are logits; apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)

        # Create binary mask based on threshold
        mask = (probs > threshold).float()

        return mask
    
    @staticmethod
    def _compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Compute confusion matrix for a batch of predictions and targets."""
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        # Filter valid indices (in case of ignore labels)
        mask = (targets_flat >= 0) & (targets_flat < num_classes)
        preds_flat = preds_flat[mask]
        targets_flat = targets_flat[mask]

        # Compute confusion matrix using bincount
        indices = targets_flat * num_classes + preds_flat
        conf_matrix = torch.bincount(indices, minlength=num_classes * num_classes)
        return conf_matrix.reshape(num_classes, num_classes)    
    
    @staticmethod
    def _compute_metrics_from_confusion_matrix(conf_matrix: torch.Tensor, num_classes: int) -> tuple:
        """Compute IoU and Dice metrics from confusion matrix."""
        iou_dict = {}
        dice_dict = {}

        iou_per_class = []
        dice_per_class = []

        for cls in range(num_classes):
            tp = conf_matrix[cls, cls].float()
            fp = conf_matrix[:, cls].sum().float() - tp
            fn = conf_matrix[cls, :].sum().float() - tp

            # IoU = TP / (TP + FP + FN)
            iou = tp / (tp + fp + fn + 1e-10)
            iou_dict[f'IoU_class_{cls}'] = iou.item()
            iou_per_class.append(iou.item())

            # Dice = 2*TP / (2*TP + FP + FN)
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-10)
            dice_dict[f'Dice_class_{cls}'] = dice.item()
            dice_per_class.append(dice.item())

        iou_dict['mIoU'] = np.mean(iou_per_class)
        dice_dict['mDice'] = np.mean(dice_per_class)

        return iou_dict, dice_dict    
    
    @staticmethod
    def transparent_overlay(original_imgs:torch.Tensor, masks:torch.Tensor, alpha:float=0.3):        
        """
        Create a transparent overlay of segmentation mask on an image
            Args:
            original_img: Input image in RGB format (bs, 3, resize_h, resize_w)
            mask: binary mask (0 or 1) or discrete class (bs, resize_h, resize_w)
            alpha: Transparency level (0-1)
        Returns:         
            Image with transparent overlay     
            
        torch.Size([16, 640, 640]) torch.Size([16, 640, 640]) torch.Size([16, 3, 640, 640])                  
        """
        
        assert len(masks.shape) == 3, f'Expected 3 dimension shape mask, got: {masks.shape}'
        assert len(original_imgs.shape) == 4, f'Expected 4 dimension shape image, got: {original_imgs.shape}'
        
        # Convert to numpy, (bs, H, W, 3)
        if original_imgs.max() <= 1.0:
            original_imgs = (original_imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        else:
            original_imgs = (original_imgs.permute(0, 2, 3, 1).cpu().numpy()).astype(np.uint8)
        
        masks = masks.cpu().numpy()        
        batch_results = []
        for b in range(original_imgs.shape[0]):
            original_img = original_imgs[b] # (H, W, 3)
            mask = masks[b] # (H, W)
            
            overlayed_image = original_img.copy()

            for cls in DriveableAreaColor:
                if cls == DriveableAreaColor.BACKGROUND:
                    continue
            
                class_id = cls.class_id
                color = cls.color
                class_mask = (mask == class_id).astype(np.uint8)
            
                #convert to binary mask with 3 channels
                if np.any(class_mask):
                    class_mask = class_mask * 255
                    class_mask_rgb = cv2.cvtColor(class_mask, cv2.COLOR_GRAY2RGB)                
                    colored_mask = cv2.bitwise_and(class_mask_rgb, color)
                    
                    overlayed_image = cv2.addWeighted(
                        overlayed_image, 1 - alpha, colored_mask, alpha, 0
                    )
            batch_results.append(overlayed_image)

        return batch_results

    @staticmethod
    def save_overlay_image(vis_image:np.array, save_path:str):
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

class SegmentationLossCalculator:

    BCESeg = torch.nn.BCEWithLogitsLoss()
    CESeg = torch.nn.CrossEntropyLoss()
    
    _ce_weights = torch.tensor([1.00, 4.91])
    
    @staticmethod
    def dice_loss(pred_softmax:torch.Tensor, target_onehot:torch.Tensor, smooth=1.0):
        dims = (0, 2, 3)
        intersection = (pred_softmax * target_onehot).sum(dim=dims)
        union = pred_softmax.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
        
    @staticmethod
    def tversky_loss(pred_softmax:torch.Tensor, target_onehot:torch.Tensor, alpha=0.2, beta=0.8, 
                    smooth=1.0, gamma=0.75):
        dims = (0, 2, 3)                                                                 
        tp = (pred_softmax * target_onehot).sum(dim=dims)                                
        fp = (pred_softmax * (1 - target_onehot)).sum(dim=dims)                          
        fn = ((1 - pred_softmax) * target_onehot).sum(dim=dims)                          
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)                 

        return (1.0 - tversky.mean()) ** gamma 
    
    @staticmethod
    def focal_loss():
        pass

    @staticmethod
    def compute_segmentation_loss_2(predictions, targets, dice_weight=0.0, ce_weight=1.0):                                         
        bs, c, h, w = predictions.shape

        # Targets are class indices (B, H, W) — convert to one-hot (B, C, H, W)
        target_onehot = torch.nn.functional.one_hot(targets, num_classes=c)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        # BCEWithLogitsLoss applies sigmoid internally (second sigmoid = double sigmoid)
        # Flatten spatial dims: (B, C, H*W)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions.view(bs, c, -1),
            target_onehot.view(bs, c, -1),
            reduction="mean"
        )

        return bce_loss

    @staticmethod
    def compute_segmentation_loss(predictions, targets, dice_weight=0.7, ce_weight=0.3):
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

        # Cross-Entropy loss (expects raw logits)
        # ce_loss = SegmentationLossCalculator.CESeg(
        #     predictions.view(bs, c, -1), targets.view(bs, -1)
        # )
        
        ce_loss = torch.nn.functional.cross_entropy(
            input=predictions.view(bs, c, -1),
            target=targets.view(bs, -1),
            reduction="mean",
            weight=SegmentationLossCalculator._ce_weights.to(predictions.device)
        )

        # Dice loss (expects softmax probabilities + one-hot targets)
        pred_softmax = torch.softmax(predictions, dim=1)
        target_onehot = torch.nn.functional.one_hot(targets, num_classes=c)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        t_loss = SegmentationLossCalculator.tversky_loss(pred_softmax, target_onehot)

        return ce_weight * ce_loss + dice_weight * t_loss