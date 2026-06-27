import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from panoptic_perception.models.models import BaseTaskModel, create_modules
from panoptic_perception.models.types import DepthReconstructionLossItems
from panoptic_perception.models.utils import parse_model_config


class SimpleCrossAdaptiveFusion(nn.Module):
    """Conv-only fusion: image_features, depth_features → (refined_image, refined_depth)."""

    def __init__(self, image_channels: int, depth_channels: int, out_channels: int):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(image_channels + depth_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

        self.image_transform = nn.Sequential(
            nn.Conv2d(image_channels + out_channels, image_channels, kernel_size=1),
            nn.BatchNorm2d(image_channels),
            nn.ReLU6(inplace=True),
        )

        self.depth_transform = nn.Sequential(
            nn.Conv2d(depth_channels + out_channels, depth_channels, kernel_size=1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, image_features: torch.Tensor, depth_features: torch.Tensor):
        deeper_features = self.conv_layer(torch.cat([image_features, depth_features], dim=1))

        image_out = self.image_transform(torch.cat([image_features, deeper_features], dim=1))
        depth_out = self.depth_transform(torch.cat([depth_features, deeper_features], dim=1))

        return image_out, depth_out


class AttentionBasedFusion(nn.Module):
    """
    Cross-attention fusion using PyTorch's scaled_dot_product_attention (SDPA),
    which is O(N) memory instead of O(N²). Safe at 768x1280 and similar resolutions.
    """

    def __init__(self, image_grid_channels: int, depth_grid_channels: int,
                 weighted_fusion: bool, dropout_p: float = 0.1):
        super().__init__()

        if image_grid_channels % 2 != 0:
            raise ValueError(
                f"image_grid_channels must be even (we project to channels // 2); got {image_grid_channels}"
            )

        if weighted_fusion:
            # The gated path mixes the image-indexed fusion_features into the
            # depth-indexed stream — same coordinate-system bug we're avoiding
            # by passing depth through unchanged in the non-weighted path.
            # Gate symmetrically (Fix B) before re-enabling.
            raise NotImplementedError(
                "AttentionBasedFusion.weighted_fusion is disabled — the gated path still "
                "writes image-indexed fusion into the depth stream. Use weighted_fusion=False "
                "until symmetric (Fix-B) cross-attention is implemented."
            )

        self.proj_channel = image_grid_channels // 2

        self.image_dim_reduce = nn.Conv2d(image_grid_channels, self.proj_channel, kernel_size=1)
        self.depth_dim_reduce = nn.Conv2d(depth_grid_channels, self.proj_channel, kernel_size=1)

        self.query_conv = nn.Conv2d(self.proj_channel, self.proj_channel, kernel_size=1)
        self.key_conv = nn.Conv2d(self.proj_channel, self.proj_channel, kernel_size=1)
        self.value_conv = nn.Conv2d(self.proj_channel, self.proj_channel, kernel_size=1)

        self.output_proj = nn.Conv2d(self.proj_channel, image_grid_channels, kernel_size=1)

        self.norm1 = nn.LayerNorm(self.proj_channel)
        self.norm2 = nn.LayerNorm(image_grid_channels)

        self.dropout_p = dropout_p
        self.image_grid_channels = image_grid_channels
        self.weighted_fusion = weighted_fusion

        if self.weighted_fusion:
            # 4-way gating: img-add, img-mul, depth-add, depth-mul.
            # Original "weighted" path used the same weight twice on both
            # add and mul, collapsing to a uniform pre-multiplier. This
            # produces a real 4-branch convex combination instead.
            self.gate_network = nn.Sequential(
                nn.Conv2d(image_grid_channels * 2, 512, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(512, 4, kernel_size=1),
                nn.Softmax(dim=1),
            )

        self.apply(self._weights_init_kaiming)
        self._pe_cache : Dict[Tuple[int, int], torch.Tensor] = {}

    @staticmethod
    def _weights_init_kaiming(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

    @staticmethod
    def _ln_chw(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        # LayerNorm wants channels last; conv tensors are channels-first.
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _get_pe(self, h:int, w:int, device:torch.device, dtype:torch.dtype):

        if (h, w) not in self._pe_cache or self._pe_cache[(h, w)].device != device:

            # Create 2D grid
            y_embed = torch.arange(h, dtype=torch.float32, device=device)
            x_embed = torch.arange(w, dtype=torch.float32, device=device)
            
            # Calculate frequencies (using half of channels for each dimension)
            dim_t = torch.arange(self.proj_channel // 2, dtype=torch.float32, device=device)
            dim_t = 10000 ** (2 * (dim_t // 2) / (self.proj_channel // 2))
            
            pos_y = y_embed[:, None] / dim_t
            pos_x = x_embed[:, None] / dim_t
            
            # Interleave sine and cosine
            pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
            pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
            
            # Combine and reshape to match query/key dimensions: (1, C, H, W)
            pos = torch.cat((
                pos_y.unsqueeze(1).expand(h, w, -1),
                pos_x.unsqueeze(0).expand(h, w, -1)
            ), dim=-1).permute(2, 0, 1).unsqueeze(0)

            self._pe_cache[(h, w)] = pos

        return self._pe_cache[(h, w)].to(dtype)

    def forward(self, image_grid_features: torch.Tensor, depth_grid_features: torch.Tensor):
        bs, _, h, w = image_grid_features.shape

        if h != w:
            raise NotImplementedError(
                f"AttentionBasedFusion currently supports square feature maps only; "
                f"got (H={h}, W={w}). Use fusion_type='simple' for non-square inputs."
            )

        image_feat = self.image_dim_reduce(image_grid_features)
        depth_feat = self.depth_dim_reduce(depth_grid_features)

        image_feat = self._ln_chw(self.norm1, image_feat)
        depth_feat = self._ln_chw(self.norm1, depth_feat)

        pe = self._get_pe(h, w, image_feat.device, image_feat.dtype)   # (1, C, h, w)
        image_feat_pe = image_feat + pe
        depth_feat_pe = depth_feat + pe

        queries = self.query_conv(image_feat_pe)
        keys = self.key_conv(depth_feat_pe)
        values = self.value_conv(depth_feat)

        # SDPA expects (B, n_heads, seq, head_dim). n_heads=1 keeps math
        # identical to the original single-head attention but with O(N) memory.
        q = queries.view(bs, -1, h * w).transpose(1, 2).unsqueeze(1)
        k = keys.view(bs, -1, h * w).transpose(1, 2).unsqueeze(1)
        v = values.view(bs, -1, h * w).transpose(1, 2).unsqueeze(1)

        fusion_features = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        fusion_features = fusion_features.squeeze(1).transpose(1, 2).view(bs, -1, h, w)

        fusion_features = self.output_proj(fusion_features)
        fusion_features = self._ln_chw(self.norm2, fusion_features)

        if self.weighted_fusion:
            raise NotImplementedError("weighted_fusion is incompatible with the asymmetric depth-queries-image branch")
        
        #TODO, symmetric depth-queries-image branch
        return image_grid_features + fusion_features, depth_grid_features


class CrossAdaptiveFusionModule(nn.Module):
    """
    Stack of N fusion blocks applied sequentially. Each block takes
    (image, depth) and returns refined (image, depth) at the same shape.
    """

    def __init__(self, image_channels: int, depth_channels: int,
                 out_channels: int, fusion_type: str,
                 weighted_fusion: bool, num_fusion_blocks: int = 3):
        super().__init__()

        blocks = []
        for _ in range(num_fusion_blocks):
            if fusion_type == "simple":
                blocks.append(SimpleCrossAdaptiveFusion(
                    image_channels=image_channels,
                    depth_channels=depth_channels,
                    out_channels=out_channels,
                ))
            elif fusion_type == "attention":
                blocks.append(AttentionBasedFusion(
                    image_grid_channels=image_channels,
                    depth_grid_channels=depth_channels,
                    weighted_fusion=weighted_fusion,
                ))
            else:
                raise ValueError(
                    f"fusion_type must be 'simple' or 'attention'; got {fusion_type!r}"
                )

        # ModuleList — Sequential cannot dispatch (image, depth) → (image, depth).
        self.cross_fusion_module = nn.ModuleList(blocks)

    def forward(self, image_features: torch.Tensor, depth_features: torch.Tensor):
        for block in self.cross_fusion_module:
            image_features, depth_features = block(image_features, depth_features)
        return image_features, depth_features

class DepthReconstructionDecoder(nn.Module):
    # Cfg-driven U-Net decoder over fused taps. Routes inside the cfg reference
    # backbone tap layer_idxs; create_modules resolves them via external_channels.
    # forward pre-seeds a cache with the fused tap tensors, then dispatches each
    # module by route exactly like YOLOP.forward.

    def __init__(self, cfg_path: str, tap_channels: Dict[int, int]):
        super().__init__()

        module_defs = parse_model_config(cfg_path)
        if module_defs[0]["type"] == "heads":
            module_defs = module_defs[1:]

        self._module_defs = module_defs
        self.module_list, self.routes, self.module_names, _ = create_modules(
            module_defs, external_channels=tap_channels,
        )

        last_c = self._last_out_channels(module_defs)
        self.depth_pred = nn.Conv2d(last_c, 1, kernel_size=1)
        self.aux_heads = nn.ModuleDict({
            str(tap): nn.Conv2d(c, 1, kernel_size=1) for tap, c in tap_channels.items()
        })

    @staticmethod
    def _last_out_channels(module_defs: list) -> int:
        for d in reversed(module_defs):
            if "out_channels" in d:
                return int(d["out_channels"])
        raise ValueError("depth_recon cfg has no out_channels declaration")

    def forward(self, fused_by_tap: Dict[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cache = dict(fused_by_tap)

        aux = {
            f"tap_{tap}": torch.sigmoid(head(fused_by_tap[int(tap)]))
            for tap, head in self.aux_heads.items()
        }

        x = None
        for module, route, mdef in zip(self.module_list, self.routes, self._module_defs):
            if len(route) == 1:
                x = module(x if route[0] == -1 else cache[route[0]])
            else:
                tensors = [x if r == -1 else cache[r] for r in route]
                x = torch.cat(tensors, dim=1)
            cache[int(mdef["layer_idx"])] = x

        full_res = torch.sigmoid(self.depth_pred(x))
        return {"full_res": full_res, **aux}


class TeacherFusion(nn.Module):
    def __init__(self, image_model: BaseTaskModel, depth_model: BaseTaskModel,
                 fusion_kwargs: dict):
        super().__init__()

        self.image_model = image_model
        self.depth_model = depth_model

        if not fusion_kwargs.get("backbone_intercepts", {}):
            raise ValueError(
                f"Expected non-empty backbone_intercepts, got "
                f"{fusion_kwargs.get('backbone_intercepts', {})}"
            )

        self._initialize_fusion_block(fusion_kwargs)
        self._initialize_depth_decoders(fusion_kwargs)

    def _initialize_depth_decoders(self, fusion_kwargs: dict):
        aux_cfg = fusion_kwargs.get("aux_depth_recon_cfg")
        if not aux_cfg:
            self.depth_decoders = None
            return

        self.depth_decoders = nn.ModuleDict({
            task: DepthReconstructionDecoder(
                cfg_path=aux_cfg,
                tap_channels={tap: ch for tap, ch in layer_specs},
            )
            for task, layer_specs in self._backbone_intercepts.items()
        })

    def _initialize_fusion_block(self, fusion_kwargs: dict):
        backbone_intercepts = fusion_kwargs.get("backbone_intercepts", {})
        if not backbone_intercepts:
            raise ValueError(f"Expected non-empty dict of backbone_intercepts, got {backbone_intercepts}")

        self._backbone_intercepts: Dict[str, List[List[int]]] = backbone_intercepts
        self._tap_layers:   List[int] = []
        self._tap_channels: List[int] = []

        for task, _layers_info in backbone_intercepts.items():
            if not isinstance(_layers_info, (list, tuple)) or not all(
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and all(isinstance(x, int) for x in pair)
                for pair in _layers_info
            ):
                raise ValueError(
                    f"backbone_intercepts[{task!r}] must be a sequence of "
                    f"[layer_idx, channels] pairs; got {_layers_info!r}"
                )

            for tap_layer, channels in _layers_info:
                self._tap_layers.append(tap_layer)
                self._tap_channels.append(channels)

        fusion_type       = fusion_kwargs.get("fusion_type", "simple")
        weighted_fusion   = fusion_kwargs.get("weighted_fusion", False)
        num_fusion_blocks = fusion_kwargs.get("num_fusion_blocks", 3)
        out_channels_arg  = fusion_kwargs.get("simple_out_channels")  # only used by simple

        # One fusion stack per (task, tap_layer). nn.ModuleDict requires
        # string keys, so layer indices are stringified at the dict layer.
        self.fusion_blocks = nn.ModuleDict({
            task: nn.ModuleDict({
                str(tap_layer): CrossAdaptiveFusionModule(
                    image_channels=channels,
                    depth_channels=channels,
                    out_channels=out_channels_arg if out_channels_arg is not None else channels,
                    fusion_type=fusion_type,
                    weighted_fusion=weighted_fusion,
                    num_fusion_blocks=num_fusion_blocks,
                )
                for tap_layer, channels in layer_specs
            })
            for task, layer_specs in backbone_intercepts.items()
        })

    def forward(self, image: torch.Tensor, depth: torch.Tensor,
                targets: Dict[str, torch.Tensor] = None):
        """
        Returns PanopticModelOutputs from image_model.forward_task_head — same
        contract as YOLOP.forward, so the existing trainer doesn't need changes.

        Fusion produces (fused_image, fused_depth) at each tap. The image-side
        fused features are routed into the image_model's task head; the depth-side
        is currently discarded but available here for symmetric losses later.
        """
        unique_taps = sorted(set(self._tap_layers))

        image_intercepts, image_cache = self.image_model.forward_backbone(image, unique_taps)
        depth_intercepts, _depth_cache = self.depth_model.forward_backbone(depth, unique_taps)

        fused_for_head: Dict[int, torch.Tensor] = {}
        for task, layer_specs in self._backbone_intercepts.items():
            for tap_layer, _channels in layer_specs:
                fused_image, _fused_depth = self.fusion_blocks[task][str(tap_layer)](
                    image_intercepts[tap_layer],
                    depth_intercepts[tap_layer],
                )
                fused_for_head[tap_layer] = fused_image

        outputs = self.image_model.forward_task_head(
            fused_intercepts=fused_for_head,
            cache=image_cache,
            image_shape=image.shape,
            targets=targets,
        )

        if self.depth_decoders is not None:
            predictions: Dict[str, Dict[str, torch.Tensor]] = {}
            for task, layer_specs in self._backbone_intercepts.items():
                fused_by_tap = {tap: fused_for_head[tap] for tap, _ in layer_specs}
                predictions[task] = self.depth_decoders[task](fused_by_tap)

            outputs.depth_reconstruction = DepthReconstructionLossItems(
                predictions=predictions,
                target=depth.detach()[:, :1],
            )

            loss_fn = self.image_model.loss_function
            if loss_fn is not None and "depth_reconstruction" in getattr(loss_fn, "task_losses", {}):
                weighted, _ = loss_fn(
                    {"depth_reconstruction": outputs.depth_reconstruction},
                    device=depth.device,
                )
                outputs.depth_reconstruction_loss = weighted["depth_reconstruction"]

        return outputs

    def get_active_tasks(self) -> str:
        """Delegate to image_model — the teacher's task structure mirrors it."""
        return self.image_model.get_active_tasks()

    def get_param_groups(self, optimizer_kwargs: dict) -> List[dict]:
        """
        Three sources merged into the optimizer's param_groups list:
          - image_backbone: delegated to image_model.get_param_groups
          - depth_backbone: delegated to depth_model.get_param_groups
          - fusion_blocks:  single group (always trainable), with own lr_scale knob

        Per-backbone freeze ranges go under:
            optimizer_kwargs["image_backbone_kwargs"]   # {groups: {...}, dcn_lr_mult: ...}
            optimizer_kwargs["depth_backbone_kwargs"]
        If absent, the top-level optimizer_kwargs is forwarded as-is (so existing
        single-backbone configs keep working when used unchanged).

        Group names are namespaced ("image.backbone", "depth.fpn", "fusion_blocks")
        so the trainer log / wandb panel can distinguish them at a glance.
        """
        image_kwargs = optimizer_kwargs.get("image_backbone_kwargs", optimizer_kwargs)
        depth_kwargs = optimizer_kwargs.get("depth_backbone_kwargs", optimizer_kwargs)

        image_groups = self.image_model.get_param_groups(image_kwargs)
        depth_groups = self.depth_model.get_param_groups(depth_kwargs)

        for g in image_groups:
            g["name"] = f"image.{g.get('name', '?')}"
        for g in depth_groups:
            g["name"] = f"depth.{g.get('name', '?')}"

        fusion_group = {
            "params": list(self.fusion_blocks.parameters()),
            "name": "fusion_blocks",
            "lr_scale": optimizer_kwargs.get("fusion_lr_scale", 1.0),
            "trainable": True,
        }

        groups = image_groups + depth_groups + [fusion_group]

        if self.depth_decoders is not None:
            groups.append({
                "params": list(self.depth_decoders.parameters()),
                "name": "depth_decoders",
                "lr_scale": optimizer_kwargs.get("depth_decoder_lr_scale", 1.0),
                "trainable": True,
            })

        return groups
