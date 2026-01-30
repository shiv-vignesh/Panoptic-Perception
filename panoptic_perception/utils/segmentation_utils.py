import torch

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
    

class SegmentationLossCalculator:

    BCESeg = torch.nn.BCEWithLogitsLoss()
    CESeg = torch.nn.CrossEntropyLoss()
    
    @staticmethod
    def dice_loss(pred_softmax:torch.Tensor, target_onehot:torch.Tensor, smooth=1.0):
        dims = (0, 2, 3)
        intersection = (pred_softmax * target_onehot).sum(dim=dims)
        union = pred_softmax.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
        
    @staticmethod
    def tversky_loss(pred_softmax:torch.Tensor, target_onehot:torch.Tensor, alpha=0.3, beta=0.7, smooth=1.0):
        dims = (0, 2, 3)                                                                 
        tp = (pred_softmax * target_onehot).sum(dim=dims)                                
        fp = (pred_softmax * (1 - target_onehot)).sum(dim=dims)                          
        fn = ((1 - pred_softmax) * target_onehot).sum(dim=dims)                          
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)                 
        return 1.0 - tversky.mean()
    
    @staticmethod
    def focal_loss():
        pass

    @staticmethod
    def compute_segmentation_loss(predictions, targets, dice_weight=0.5, ce_weight=0.5):
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
        ce_loss = SegmentationLossCalculator.CESeg(
            predictions.view(bs, c, -1), targets.view(bs, -1)
        )

        # Dice loss (expects softmax probabilities + one-hot targets)
        pred_softmax = torch.softmax(predictions, dim=1)
        target_onehot = torch.nn.functional.one_hot(targets, num_classes=c)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        d_loss = SegmentationLossCalculator.dice_loss(pred_softmax, target_onehot)

        return ce_weight * ce_loss + dice_weight * d_loss