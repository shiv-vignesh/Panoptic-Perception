from typing import List

import torch

from panoptic_perception.losses.loss_factory import LossFactory

@LossFactory.register_loss_function("image-classifier")
class ImageClassifierLoss:

    _LOSS_FUNCTIONS = {
        "cross_entropy":torch.nn.CrossEntropyLoss
    }

    def __init__(self, criterion: str, criterion_kwargs: dict = None):

        if criterion not in self._LOSS_FUNCTIONS:
            raise ValueError(f'Unsupported Loss function: {criterion}')
        
        criterion_kwargs = criterion_kwargs or {}
        try:
            self.loss_function = self._LOSS_FUNCTIONS[criterion](**criterion_kwargs)
        except TypeError as e:
            # Catches invalid keyword arguments passed to the PyTorch loss constructor
            raise ValueError(
                f"Invalid arguments provided for {criterion}: {e}"
            ) from e

    def __call__(self, logits:torch.Tensor, targets:torch.Tensor):

        return self.loss_function(
            logits, targets
        )