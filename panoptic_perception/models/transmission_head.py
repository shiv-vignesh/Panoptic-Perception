"""Auxiliary transmission-map head for depth-guided fog training.

Predicts a single-channel transmission map t_hat(x) in [0, 1] from a
mid-level backbone feature map. Supervised by the physical transmission
t*(x) = exp(-beta * d(x) * max_depth) provided by FoggyBDD100KDataset.

The head exists only during training as an inductive bias: it forces the
encoder to internalize depth-like spatial structure (where fog lives)
without requiring depth at inference. At inference, callers can ignore
the head's output entirely.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransmissionHead(nn.Module):
    """Small conv head: feature map -> single-channel transmission in [0, 1].

    Designed to attach to a mid/deep backbone feature map (stride 16 or 32
    typical for YOLOP/YOLOv8). Upsamples to the target spatial size at the
    end so the loss can be computed against the full-resolution target.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, feature: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """Returns transmission map of shape (B, 1, H, W) in [0, 1]."""
        logits = self.net(feature)
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


def transmission_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fog_applied: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Masked L1 between predicted and target transmission.

    Args:
        pred:        (B, 1, H, W), in [0, 1]
        target:      (B, 1, H, W), in [0, 1]
        fog_applied: (B,) float, 1.0 where fog was applied, 0.0 for clean samples
        reduction:   'mean' or 'sum' over fog-applied samples only.

    L1 is more robust than MSE here — transmission has long-tailed values
    when beta is large (small t near 0 for distant pixels), and we don't
    want a few far-away pixels to dominate the gradient.
    """
    if fog_applied.sum() == 0:
        return pred.new_zeros(())

    per_pixel = (pred - target).abs()                # (B, 1, H, W)
    per_sample = per_pixel.flatten(1).mean(dim=1)    # (B,)
    masked = per_sample * fog_applied                # (B,)

    if reduction == "sum":
        return masked.sum()
    return masked.sum() / fog_applied.sum().clamp(min=1.0)
