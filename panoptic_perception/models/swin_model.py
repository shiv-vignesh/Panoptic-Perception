from typing import Optional, Dict

import ast

import torch
import torch.nn as nn

from panoptic_perception.models.model_factory import ModelFactory
from panoptic_perception.models.utils import parse_model_config, initialize_weights
from panoptic_perception.models.types import ImageClassifierOutputs, PanopticModelOutputs
from panoptic_perception.models.common import (
    PatchEmbed, SwinLayer, PatchMerge
)

from panoptic_perception.models.models import create_modules, BaseTaskModel

from panoptic_perception.losses.multi_task_loss import MultiTaskLoss
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
                    apply_ape=str(m.get("apply_ape", "False")).lower() == "true"
                )
                H, W = image_size[0] // patch_size, image_size[1] // patch_size

            elif t == "SwinStack":
                depths = ast.literal_eval(m["depths"])
                num_heads = ast.literal_eval(m["num_heads"])
                window_size = int(m.get("window_size", 7))
                mlp_ratio = float(m.get("mlp_ratio", 4.))
                qkv_bias=str(m.get("qkv_bias", "True")).lower() == "true"
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

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def get_param_groups(self, optimizer_kwargs: dict) -> list:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "name": "decay", "lr_scale": 1.0, "trainable": True},
            {"params": no_decay, "name": "no_decay", "lr_scale": 1.0, "trainable": True,
             "weight_decay": 0.0},
        ]

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

@ModelFactory.register_task_model("swin-yolov5-fpn")
class SwinObjectDetection(BaseTaskModel):

    _BACKBONE_BLOCK_TYPES = ("PatchEmbed", "SwinStack")

    def __init__(self, backbone:SwinBackbone, cfg:str,
                loss_function:Optional[MultiTaskLoss]=None):
        
        super().__init__(loss_function=loss_function)
        self.backbone = backbone
        self._build(cfg)

    def _build(self, cfg:str):

        module_defs = parse_model_config(cfg)

        if module_defs[0]["type"] == "heads":
            self.detection_head_idx = int(module_defs[0].get("detection_head_idx", -1))
            self.segmentation_head_idx = int(module_defs[0].get("segmentation_head_idx", -1))
            self.lane_segmentation_head_idx = int(module_defs[0].get("lane_segmentation_head_idx", -1))
            module_defs = module_defs[1:]

        neck_defs = [m for m in module_defs if m["type"] not in self._BACKBONE_BLOCK_TYPES]

        self.module_list, self.routes, self.module_names, self._cache_layer_idx = create_modules(
            module_defs=neck_defs,
            segmentation_head_idx=self.segmentation_head_idx,
            lane_segmentation_head_idx=self.lane_segmentation_head_idx
        )

        self._tap_indices: Dict[int, int] = {
            i: int(m["tap_idx"])
            for i, m in enumerate(neck_defs)
            if m["type"] == "SwinFeatureReshape"
        }
        self._num_taps = len(self._tap_indices)

        initialize_weights(self.module_list)

    def get_param_groups(self, optimizer_kwargs: dict = None) -> list:
        optimizer_kwargs = optimizer_kwargs or {}
        initial_lr = float(optimizer_kwargs.get("initial_lr", 1e-4))
        backbone_lr_scale = float(optimizer_kwargs.get("backbone_lr_scale", 0.1))
        neck_head_lr_scale = float(optimizer_kwargs.get("neck_head_lr_scale", 1.0))

        return [
            {
                "params": list(self.backbone.parameters()),
                "name": "backbone",
                "lr": initial_lr * backbone_lr_scale,
                "lr_scale": backbone_lr_scale,
                "trainable": True,
            },
            {
                "params": list(self.module_list.parameters()),
                "name": "neck_head",
                "lr": initial_lr * neck_head_lr_scale,
                "lr_scale": neck_head_lr_scale,
                "trainable": True,
            },
        ]

    def forward(self, x: torch.Tensor, targets:torch.Tensor=None) -> ImageClassifierOutputs:

        batch_size, _, height, width = x.shape
        device = x.device
        cache = {} # Cache for layer outputs
        model_outputs = PanopticModelOutputs()

        x, _intercepts = self.backbone(x, intercept_layers=sorted(set(self._tap_indices.values())))

        for i, (module, route) in enumerate(zip(self.module_list, self.routes)):
            if self.module_names[i] == "SwinFeatureReshape":
                tap_idx = self._tap_indices[i]
                swin_tokens = _intercepts[tap_idx]
                cache[i] = module(swin_tokens, height, width, tap_idx)
                x = cache[i]

            elif len(route) == 1:
                if route[0] == -1:
                    x = module(x)
                else:
                    assert route[0] in cache, f"Output for layer {route[0]} not found in cache."
                    x = module(cache[route[0]])

            elif len(route) > 1:
                if self.module_names[i] == "Concat":
                    for r in route:
                        if r == -1:
                            continue
                        assert r in cache, f"Output for layer {r} not found in cache."
                        x = torch.cat([x, cache[r]], dim=1)

                elif self.module_names[i] == "ResidualAdd":
                    for r in route:
                        if r == -1:
                            continue
                        assert r in cache, f"Output for layer {r} not found in cache."
                        assert x.shape == cache[r].shape, f"Residual Add Expects Tensors of Same Size, Found: {x.shape} and {cache[r].shape}"

                        x = x + cache[r]

                elif self.module_names[i] == "Detect":
                    inputs = []
                    for r in route:
                        if r == -1:
                            continue
                        assert r in cache, f"Output for layer {r} not found in cache."
                        inputs.append(cache[r])

                    detection_outputs  = module(inputs, image_size=(height, width))

                    model_outputs.detection_logits = detection_outputs
                    model_outputs.anchor_proposals = module._anchor_proposals
                    model_outputs.proposal_shape = module._proposal_shape
                    model_outputs.anchor_cxcy = module._anchor_cxcy
                    model_outputs.anchor_wh = module._anchor_wh
                    model_outputs.anchor_strides = module._anchor_strides

                    if not self.training:
                        model_outputs.detection_predictions = module.activation(detection_outputs)

            # Capture segmentation outputs 
            if self.module_names[i] == "DrivableAreaSegmentation":
                model_outputs.drivable_segmentation_logits = x
                if not self.training:
                    model_outputs.drivable_segmentation_predictions = torch.softmax(x, dim=1)

            elif self.module_names[i] == "LaneSegmentation":
                model_outputs.lane_segmentation_logits = x
                if not self.training:
                    model_outputs.lane_segmentation_predictions = torch.softmax(x, dim=1)

            if i in self._cache_layer_idx:
                cache[i] = x                        

            # print(f'Layer_idx: {i} - {self.module_names[i]} Route - {route} Tensor - {x.shape}')

        if targets is not None:
            self._compute_loss(
                model_outputs, 
                targets, 
                height, width,
                batch_size,
                device                
            )

        return model_outputs

    @classmethod
    def from_config(cls, cfg: str):
        backbone = SwinBackbone(cfg)

        return cls(
            backbone, cfg
        )