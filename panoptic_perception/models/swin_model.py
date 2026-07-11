from typing import Optional

import ast

import torch
import torch.nn as nn

from panoptic_perception.models.model_factory import ModelFactory
from panoptic_perception.models.utils import parse_model_config
from panoptic_perception.models.types import ImageClassifierOutputs
from panoptic_perception.models.common import (
    PatchEmbed, SwinLayer, PatchMerge
)

from panoptic_perception.losses.classifier import ImageClassifierLoss


class SwinBackbone(nn.Module):
    def __init__(self, cfg: str):
        super().__init__()
        self.patch_embed = None
        self.swin_layers = nn.ModuleList()
        self.final_channels = -1
        self._build(cfg)

    def _build(self, cfg):
        module_defs = parse_model_config(cfg)
        H = W = -1
        embed_dim = -1
        num_layers = 0

        for m in module_defs:
            t = m["type"]

            if t == "PatchEmbed":
                image_size = ast.literal_eval(m.get("image_size", "(224, 224)"))
                embed_dim = int(m.get("embed_dim", 96))
                patch_size = int(m.get("patch_size", 4))
                self.patch_embed = PatchEmbed(
                    image_size=image_size,
                    in_channels=int(m.get("num_channels", 3)),
                    embed_dim=embed_dim,
                    patch_size=patch_size,
                    stride=int(m.get("stride", 4)),
                    apply_ape=bool(m.get("apply_ape", False)),
                )
                H, W = image_size[0] // patch_size, image_size[1] // patch_size

            elif t == "SwinStack":
                depths = ast.literal_eval(m["depths"])
                num_heads = ast.literal_eval(m["num_heads"])
                window_size = int(m.get("window_size", 7))
                mlp_ratio = float(m.get("mlp_ratio", 4.))
                qkv_bias = bool(m.get("qkv_bias", True))
                num_layers = len(depths)

                assert H != -1 and W != -1, "PatchEmbed must precede SwinStack in cfg"
                assert embed_dim != -1, "embed_dim not set; PatchEmbed must come first"

                for i in range(num_layers):
                    self.swin_layers.append(SwinLayer(
                        embed_dim=int(embed_dim * 2 ** i),
                        input_res=(H // (2 ** i), W // (2 ** i)),
                        depth=depths[i],
                        num_heads=num_heads[i],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        downsample=PatchMerge if i < num_layers - 1 else None,
                    ))

        assert num_layers > 0, "no SwinStack in cfg"
        self.final_channels = int(embed_dim * 2 ** (num_layers - 1))

    def forward(self, x: torch.Tensor, intercept_layers=None):

        x = self.patch_embed(x)
        taps = {}
        for i, layer in enumerate(self.swin_layers):
            x = layer(x)
            if intercept_layers is not None and i in intercept_layers:
                taps[i] = x

        return x, taps


@ModelFactory.register_task_model("swin-classifier")
class SwinClassifier(nn.Module):
    def __init__(self, backbone: SwinBackbone, num_classes: int, loss_function:Optional[ImageClassifierLoss]=None):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(backbone.final_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(backbone.final_channels, num_classes)

        self.num_classes = num_classes

        self._loss_function = loss_function

    def forward(self, x: torch.Tensor, targets:torch.Tensor=None) -> ImageClassifierOutputs:
        x, _ = self.backbone(x)
        x = self.norm(x)
        x = self.pool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        logits = self.head(x)

        model_outputs = ImageClassifierOutputs()
        model_outputs.logits = logits

        if targets is not None:
            loss = self.loss_function(logits, targets)
            model_outputs.loss = loss
            model_outputs.targets = targets

        return model_outputs

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_fn:ImageClassifierLoss):
        if loss_fn is not None and not isinstance(loss_fn, ImageClassifierLoss):
            raise ValueError(
                f"Currently supported loss function: {ImageClassifierLoss}, got {type(loss_fn.__name__)}"
            )
        self._loss_function = loss_fn

    @classmethod
    def from_config(cls, cfg: str):
        backbone = SwinBackbone(cfg)
        num_classes = None
        for m in parse_model_config(cfg):
            if m["type"] == "cls_head":
                num_classes = int(m["num_classes"])
                break
        if num_classes is None:
            raise ValueError("SwinClassifier.from_cfg requires a [cls_head] block with num_classes")
        return cls(backbone, num_classes)
