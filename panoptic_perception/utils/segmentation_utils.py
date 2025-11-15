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
    def focal_loss():
        pass

    @staticmethod
    def compute_segmentation_loss(predictions, targets):
        """
        Compute the segmentation loss.

        Args:
            predictions (torch.Tensor): The raw output from the segmentation head of the model.
            targets (torch.Tensor): The ground truth segmentation masks.
            criterion: Loss function to use (e.g., nn.BCEWithLogitsLoss).

        Returns:
            torch.Tensor: Computed loss value.
        """
        
        bs, c, h, w = predictions.shape
        bs, h_t, w_t = targets.shape
        
        assert h == h_t and w == w_t, "Prediction and target spatial dimensions must match."
        
        predictions = predictions.view(bs, c, -1)
        targets = targets.view(bs, -1)

        if c == 1:
            loss = SegmentationLossCalculator.BCESeg(predictions, targets)
        else:
            loss = SegmentationLossCalculator.CESeg(predictions, targets)
        
        return loss