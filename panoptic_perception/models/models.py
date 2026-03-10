import os, ast
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn

from panoptic_perception.models.common import (
    ConvBlock, Focus, BottleneckCSP, SPP, Upsample, Detect, C2F, SPPF, DetectV8
)
from panoptic_perception.models.utils import parse_model_config, initialize_weights, PanopticModelOutputs
from panoptic_perception.models.gdip import GDIP, MultiLevelGDIP, build_vision_encoder
from panoptic_perception.utils.detection_utils import DetectionLossCalculator
from panoptic_perception.utils.segmentation_utils import SegmentationLossCalculator

def create_modules(module_defs: list,
                   segmentation_head_idx: int = -1,
                   lane_segmentation_head_idx: int = -1) -> Tuple[nn.ModuleList, list, list, list]:
    """
    Create modules from config definitions.

    Args:
        module_defs: List of module definition dicts
        num_classes: Number of detection classes
        segmentation_head_idx: Index of drivable area segmentation head (default: -1)
        lane_segmentation_head_idx: Index of lane segmentation head (default: -1)

    Returns:
        Tuple of (module_list, routes, module_names, cache_layer_idx)
    """
    module_list = []
    routes = []
    module_names = []
    output_channels = []  # Track the output channels for each layer

    for i, module_def in enumerate(module_defs):
        mtype = module_def["type"]
        module_names.append(mtype)

        # Parse route info
        route = module_def.get("route", -1)
        route = [-1] if route == -1 else [int(x) for x in str(route).split(",")]

        # -- FOCUS --------------------------------------------------------------
        if mtype == "Focus":
            in_ch = int(module_def.get("in_channels", 3))
            out_ch = int(module_def["out_channels"])
            k = int(module_def["kernel_size"])
            module = Focus(in_ch, out_ch, k)
            output_channels.append(out_ch)

        # -- CONVBLOCK ----------------------------------------------------------
        elif mtype == "ConvBlock":
            if route[0] != -1:
                in_ch = int(module_def["in_channels"])
            else:
                in_ch = output_channels[-1] if len(output_channels) > 0 else int(module_def["in_channels"])

            out_ch = int(module_def["out_channels"])
            k = int(module_def["kernel_size"])
            s = int(module_def["stride"])
            p = int(module_def["padding"]) if "padding" in module_def else None
            activation_func = str(module_def.get("activation_func", "hardswish"))
            
            module = ConvBlock(in_ch, out_ch, k, s, p, activation_func=activation_func)
            output_channels.append(out_ch)

            # Check if this ConvBlock is a segmentation head
            layer_idx = int(module_def.get("layer_idx", i))
            if layer_idx == segmentation_head_idx:
                module_names[-1] = "DrivableAreaSegmentation"  # Mark as segmentation head
            elif layer_idx == lane_segmentation_head_idx:
                module_names[-1] = "LaneSegmentation"  # Mark as lane segmentation head

        # -- BOTTLENECKCSP -----------------------------------------------------
        elif mtype == "BottleneckCSP":
            in_ch = output_channels[-1] if len(output_channels) > 0 else int(module_def["in_channels"])
            out_ch = int(module_def["out_channels"])
            n = int(module_def["num_bottlenecks"])
            residual = str(module_def.get("residual", "True")) == "True"
            use_deform = str(module_def.get("use_deform", "False")) == "True"
            
            module = BottleneckCSP(in_ch, out_ch, n, residual, use_deform)
            output_channels.append(out_ch)

        elif mtype == "C2F":
            in_ch = output_channels[-1] if len(output_channels) > 0 else int(module_def["in_channels"])
            out_ch = int(module_def["out_channels"])
            n = int(module_def["n"])
            shortcut = bool(module_def["shortcut"])
            
            module = C2F(in_ch, out_ch, n, shortcut)
            output_channels.append(out_ch)
        
        # -- SPP ---------------------------------------------------------------
        elif mtype == "SPP":
            in_ch = output_channels[-1] if len(output_channels) > 0 else int(module_def["in_channels"])
            out_ch = int(module_def["out_channels"])
            
            ks = [int(x) for x in module_def.get("pool_sizes", "5,9,13").split(",")]
            module = SPP(in_ch, out_ch, ks)
            output_channels.append(out_ch)

        elif mtype == "SPPF":
            in_ch = output_channels[-1] if len(output_channels) > 0 else int(module_def["in_channels"])
            out_ch = int(module_def["out_channels"])

            module = SPPF(in_ch, out_ch)
            output_channels.append(out_ch)

        # -- UPSAMPLE ----------------------------------------------------------
        elif mtype == "Upsample":
            sf = int(module_def.get("scale_factor", 2))
            mode = module_def.get("mode", "nearest")
            module = Upsample(sf, mode)
            # channel count stays same as input
            output_channels.append(output_channels[-1])

        # -- CONCAT -------------------------------------------------------------
        elif mtype == "Concat":
            module = nn.Identity()
            # Compute total channels = sum of channels from each route layer
            concat_ch = sum(output_channels[r] if r != -1 else output_channels[-1] for r in route)
            output_channels.append(concat_ch)

        elif mtype == "ResidualAdd":
            module = nn.Identity()
            output_channels.append(output_channels[-1])

        # -- HEADS (skip building) ---------------------------------------------
        elif mtype == "Detect":
            num_classes = int(module_def.get("num_classes"))
            anchors, _ = ast.literal_eval(module_def["anchors"])
            channels = [output_channels[r] if r != -1 else output_channels[-1] for r in route]

            module = Detect(anchors, num_classes, channels)
        
        elif mtype == "DetectV8":
            num_classes = int(module_def.get("num_classes"))
            channels = [output_channels[r] if r != -1 else output_channels[-1] for r in route]
            reg_max = int(module_def.get("reg_max", 16))

            module = DetectV8(num_classes, channels, reg_max=reg_max)

        module_list.append(module)
        routes.append(route)

    # Cache layers needed later
    _cache_layer_idx = []
    for r in routes:
        if isinstance(r, list):
            for ri in r:
                if ri != -1:
                    _cache_layer_idx.append(ri)

    return nn.ModuleList(module_list), routes, module_names, _cache_layer_idx

class YOLOP(nn.Module):
    def __init__(self, cfg: str, img_size: int = 416, num_classes: int = 80,
                 loss_weights: dict = None):
        super(YOLOP, self).__init__()

        module_defs = parse_model_config(cfg)

        #TODO, unused vars: remove
        self.img_size = img_size
        self.num_classes = num_classes

        # Multi-task loss weights (detection prioritized by default)
        self.loss_weights = loss_weights or {
            "detection": 1.0,
            "drivable_segmentation": 1.0,
            "lane_segmentation": 0.0
        }

        if module_defs[0]['type'] == 'heads':
            self.module_defs = module_defs[1:]

            self.segmentation_head_idx = int(module_defs[0].get('segmentation_head_idx', -1))
            self.lane_segmentation_head_idx = int(module_defs[0].get('lane_segmentation_head_idx', -1))
            self.detection_head_idx = int(module_defs[0].get('detection_head_idx', -1))

        self.in_channels = self.module_defs[0].get('in_channels', 3)
        self.module_list, self.routes, self.module_names, self._cache_layer_idx = create_modules(
            self.module_defs,
            segmentation_head_idx=self.segmentation_head_idx,
            lane_segmentation_head_idx=self.lane_segmentation_head_idx
        )

        # Initialize model weights
        initialize_weights(self)
        
    def forward(self, x, targets=None) -> PanopticModelOutputs:
        
        cache = {} # Cache for layer outputs
        _, _, height, width = x.shape
        
        model_outputs = PanopticModelOutputs()

        for i, (module, route) in enumerate(zip(self.module_list, self.routes)):
            if route == -1:
                x = module(x)  # single previous layer route
            
            elif len(route) == 1:
                # single previous layer route
                if route[0] == -1:
                    x = module(x)
                else:
                    x = module(cache[route[0]]) 

            elif len(route) > 1:
                # multiple previous layer routes
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
                    
                    if not self.training:
                        model_outputs.detection_predictions = module.activation(detection_outputs)

                    # for detection in detection_outputs:
                    #     print(f'Layer: {i} - Module Name: {self.module_names[i]} - prediction shape: {detection.shape}')
                
                else: #TODO, future modules
                    pass # concatenate multiple previous layers

            # Capture segmentation outputs
            if self.module_names[i] == "DrivableAreaSegmentation":
                model_outputs.drivable_segmentation_logits = x
                if not self.training:
                    # predictions["drivable_area_seg"] = x
                    model_outputs.drivable_segmentation_predictions = torch.softmax(x, dim=1)
                    # model_outputs.drivable_segmentation_predictions = torch.sigmoid(x)
                # print(f'Layer: {i} - Module Name: {self.module_names[i]} (Drivable Area Seg Head) - output shape: {x.shape}')

            elif self.module_names[i] == "LaneSegmentation":
                model_outputs.lane_segmentation_logits = x
                if not self.training:
                    model_outputs.lane_segmentation_predictions = torch.softmax(x, dim=1)

                # print(f'Layer: {i} - Module Name: {self.module_names[i]} (Lane Seg Head) - output shape: {x.shape}')

            # else:
            #     # print(f'Layer: {i} - Module Name: {self.module_names[i]} - output shape: {x.shape}')

            if i in self._cache_layer_idx:
                cache[i] = x
                
            # print(f'Layer_idx: {i} - ModuleName: {self.module_names[i]} - TensorShape: {x.shape}')

        
        if targets is not None:

            if model_outputs.detection_logits is not None and targets["detections"] is not None:
                output_name = "detections"
                assert output_name in targets, f"Target for {output_name} not provided."

                det_loss, det_loss_items = DetectionLossCalculator.compute_detection_loss_2(
                    model_outputs.detection_logits,
                    targets["detections"],
                    self.module_list[self.detection_head_idx].num_layers,
                    self.module_list[self.detection_head_idx].anchors,
                    self.module_list[self.detection_head_idx].stride,
                    cls_loss_type='focal'  # Use focal loss for better class imbalance handling
                )

                # Apply multi-task loss weight
                model_outputs.detection_loss = det_loss * self.loss_weights.get("detection", 1.0)
                # print(f'Detection Loss: {det_loss}')

            if model_outputs.drivable_segmentation_logits is not None and targets["drivable_area_seg"] is not None:
                output_name = "drivable_area_seg"
                assert output_name in targets, f"Target for {output_name} not provided."
                drivable_seg_loss = SegmentationLossCalculator.compute_segmentation_loss(
                    model_outputs.drivable_segmentation_logits,
                    targets["drivable_area_seg"]
                )
                # Apply multi-task loss weight
                model_outputs.drivable_segmentation_loss = drivable_seg_loss * self.loss_weights.get("drivable_segmentation", 0.2)
                # print(f'Drivable Area Segmentation Loss: {drivable_seg_loss}')
                
            if model_outputs.lane_segmentation_logits is not None:
                output_name = "lane_seg"
                assert output_name in targets, f"Target for {output_name} not provided."

                lane_seg_loss = SegmentationLossCalculator.compute_segmentation_loss(
                    model_outputs.lane_segmentation_logits,
                    targets["lane_seg"]
                )
                # Apply multi-task loss weight
                model_outputs.lane_segmentation_loss = lane_seg_loss * self.loss_weights.get("lane_segmentation", 0.0)
                # print(f'Lane Segmentation Loss: {lane_seg_loss}')

        return model_outputs

class YOLOv8P(nn.Module):
    def __init__(self, cfg:str, loss_weights: dict = None):
        super(YOLOv8P, self).__init__()
        
        module_defs = parse_model_config(cfg)
        # Multi-task loss weights (detection prioritized by default)
        self.loss_weights = loss_weights or {
            "detection": 1.0,
            "drivable_segmentation": 1.0,
            "lane_segmentation": 0.0
        }
        
        if module_defs[0]["type"] == "heads":
            self.module_defs = module_defs[1:]
            
            self.detection_head_idx = int(module_defs[0].get('detection_head_idx', -1))
            self.segmentation_head_idx = int(module_defs[0].get('segmentation_head_idx', -1))
            self.lane_segmentation_head_idx = int(module_defs[0].get('lane_segmentation_head_idx', -1))
            
            
        self.in_channels = int(self.module_defs[0].get("in_channels", 3))
        self.module_list, self.routes, self.module_names, self._cache_layer_idx = create_modules(
            self.module_defs,
            segmentation_head_idx=self.segmentation_head_idx,
            lane_segmentation_head_idx=self.lane_segmentation_head_idx
        )
        
        self.anchor_free = False
        if "DetectV8" in self.module_names:
            self.anchor_free = True

        # Initialize model weights
        initialize_weights(self)
        
    def forward(self, x, targets=None):

        cache = {} # Cache for layer outputs
        _, _, height, width = x.shape
        
        model_outputs = PanopticModelOutputs()
        
        for i, (module, route) in enumerate(zip(self.module_list, self.routes)):
            if route == -1:
                x = module(x)
                
            elif len(route) == 1:
                # single previous layer route
                if route[0] == -1:
                    x = module(x)
                else:
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
                    model_outputs.detection_logits = detection_outputs #list

                    if not self.training:
                        model_outputs.detection_predictions = module.activation(detection_outputs) #list

                elif self.module_names[i] == "DetectV8":
                    inputs = []
                    for r in route:
                        if r == -1:
                            continue
                        assert r in cache, f"Output for layer {r} not found in cache."
                        inputs.append(cache[r])

                    bbox_outputs, cls_outputs, anchor_points, strides = module(inputs, image_size=(height, width))
                    
                    model_outputs.bbox_logits_raw = bbox_outputs
                    model_outputs.cls_logits_raw = cls_outputs
                    model_outputs.anchor_points = anchor_points
                    model_outputs.strides = strides
                    
                    if not self.training:
                        model_outputs.detection_predictions = module.activation(
                                model_outputs.bbox_logits_raw,
                                model_outputs.cls_logits_raw,
                                model_outputs.anchor_points,
                                model_outputs.strides
                            ) #torch.Tensor (B, 8400, 4+C)
                
                else: #TODO, future modules
                    pass # concatenate multiple previous layers                
            
            # Capture segmentation outputs
            if self.module_names[i] == "DrivableAreaSegmentation":
                model_outputs.drivable_segmentation_logits = x
                if not self.training:
                    model_outputs.drivable_segmentation_predictions = torch.softmax(x, dim=1)
            
            # Capture segmentation outputs
            if self.module_names[i] == "LaneSegmentation":
                model_outputs.drivable_segmentation_logits = x
                if not self.training:
                    model_outputs.drivable_segmentation_predictions = torch.softmax(x, dim=1)

            if i in self._cache_layer_idx:
                cache[i] = x            

        if targets is not None:

            has_detection = (model_outputs.detection_logits is not None or
                            model_outputs.bbox_logits_raw is not None)

            if has_detection and targets["detections"] is not None:
                output_name = "detections"
                assert output_name in targets, f"Target for {output_name} not provided."

                if self.anchor_free:
                    det_loss, det_loss_items = DetectionLossCalculator.compute_detection_loss_v8(
                        model_outputs.cls_logits_raw,
                        model_outputs.bbox_logits_raw,
                        model_outputs.anchor_points,
                        model_outputs.strides,
                        targets["detections"],
                        image_size=(height, width)
                    )

                else:
                    det_loss, det_loss_items = DetectionLossCalculator.compute_detection_loss_2(
                        model_outputs.detection_logits,
                        targets["detections"],
                        self.module_list[self.detection_head_idx].num_layers,
                        self.module_list[self.detection_head_idx].anchors,
                        self.module_list[self.detection_head_idx].stride,
                        cls_loss_type='focal'  # Use focal loss for better class imbalance handling
                    )

                # Apply multi-task loss weight
                model_outputs.detection_loss = det_loss * self.loss_weights.get("detection", 1.0)
                # print(f'Detection Loss: {det_loss}')

            if model_outputs.drivable_segmentation_logits is not None and targets["drivable_area_seg"] is not None:
                output_name = "drivable_area_seg"
                assert output_name in targets, f"Target for {output_name} not provided."
                drivable_seg_loss = SegmentationLossCalculator.compute_segmentation_loss(
                    model_outputs.drivable_segmentation_logits,
                    targets["drivable_area_seg"]
                )
                # Apply multi-task loss weight
                model_outputs.drivable_segmentation_loss = drivable_seg_loss * self.loss_weights.get("drivable_segmentation", 0.2)
                # print(f'Drivable Area Segmentation Loss: {drivable_seg_loss}')
                
            if model_outputs.lane_segmentation_logits is not None:
                output_name = "lane_seg"
                assert output_name in targets, f"Target for {output_name} not provided."

                lane_seg_loss = SegmentationLossCalculator.compute_segmentation_loss(
                    model_outputs.lane_segmentation_logits,
                    targets["lane_seg"]
                )
                # Apply multi-task loss weight
                model_outputs.lane_segmentation_loss = lane_seg_loss * self.loss_weights.get("lane_segmentation", 0.0)
                # print(f'Lane Segmentation Loss: {lane_seg_loss}')

        return model_outputs

class GDIPYolo(nn.Module):
    def __init__(self, model_type:str, yolo_cfg:str, gdip_kwargs:dict, loss_weights:dict = None):
        super(GDIPYolo, self).__init__()
        
        self.model_type = model_type        
        if model_type == "yolop":
            self.model = YOLOP(yolo_cfg, loss_weights)
        elif model_type == "yolov8p":
            self.model = YOLOv8P(yolo_cfg, loss_weights)

        self.gdip_mode = gdip_kwargs["mode"]
        
        assert "encoder" in gdip_kwargs and bool(gdip_kwargs["encoder"]), f'Invalid Vision Encoder Kwargs: {gdip_kwargs["encoder"]}'
        
        vision_encoder_kwargs = gdip_kwargs["encoder"]
        if self.gdip_mode == "gdip":            
            self.vision_encoder = build_vision_encoder({
                "type":vision_encoder_kwargs["type"],
                "latent_dim":vision_encoder_kwargs.get("latent_dim", 256),
                "pretrained":vision_encoder_kwargs.get("pretrained", True)
            })

            self.gdip_module = GDIP(
                vision_encoder_kwargs.get("latent_dim", 256)
            )
        
        elif self.gdip_mode == "mgdip":
            assert bool(gdip_kwargs["encoder"]["tap_layers"]), f"For GDIP Mode: {self.gdip_mode} -- Tap Layers must not be empty: {len(gdip_kwargs['encoder']['tap_layers'])}"
            self.tap_layers = gdip_kwargs["encoder"]["tap_layers"]
            self.vision_encoder = build_vision_encoder({
                "type":vision_encoder_kwargs["type"],
                "latent_dim":vision_encoder_kwargs.get("latent_dim", 256),
                "pretrained":vision_encoder_kwargs.get("pretrained", True),
                "tap_layers":gdip_kwargs["encoder"]["tap_layers"]
            })
            
            self.gdip_module = MultiLevelGDIP(
                num_gdip_blocks=len(self.tap_layers),
                latent_dim=vision_encoder_kwargs.get("latent_dim", 256)
            )
            
        self.vis_intermediate = gdip_kwargs.get("visualize_intermediate", True)
            
    def forward(self, x:torch.Tensor, targets=None):
        
        if self.gdip_mode == "gdip":
            latent = self.vision_encoder(x)
            enhanced_image, gates = self.gdip_module(x, latent)
            
            #TODO, plot enhanced image (wandb)
            
        elif self.gdip_mode == "mgdip":
            latent, features = self.vision_encoder(x)
            enhanced_image, gates, intermediate_images = self.gdip_module(x, features, self.vis_intermediate)
        
            #TODO, plot intermediate_images images (wandb)

        return self.model(enhanced_image, targets)
    
def get_model_param_groups(model:Optional[Union[YOLOP, YOLOv8P]], groups:dict, dcn_lr_mult:float=0.1):
    """
    Configure parameter groups for training with selective freezing.

    Supports differential learning rates for DCN layers:
    - offset_conv parameters: base_lr * dcn_lr_mult (slowest, for stability)
    - deform_conv parameters: base_lr * dcn_lr_mult * 5 (slower than normal)
    - other parameters: base_lr (normal)

    For frozen layers:
    - Sets requires_grad = False for all parameters
    - Sets BatchNorm layers to eval mode to prevent running stats from updating

    For trainable layers:
    - Sets requires_grad = True
    - Returns parameters in param_groups for optimizer
    """

    param_groups = []
    frozen_layers = []

    # Collect DCN parameters separately
    dcn_offset_params = []
    dcn_conv_params = []

    for group_name in groups:
        group = groups[group_name]["group"]
        trainable = groups[group_name]["trainable"]
        lr_scale = groups[group_name].get("lr_scale", 1.0)

        last_layer_index = len(model.module_list) - 1

        if len(group) == 2:
            start_idx, end_idx = group

            if start_idx > last_layer_index and end_idx > last_layer_index:
                continue

            layer_indices = list(range(start_idx, end_idx+1))
        elif len(group) == 1:
            idx = group[0]

            if idx > last_layer_index:
                continue

            layer_indices = [idx]
        else:
            layer_indices = []

        for idx in layer_indices:
            module = model.module_list[idx]

            # Set requires_grad for all parameters
            for p in module.parameters():
                p.requires_grad = trainable

            # Track frozen layers for BatchNorm handling
            if not trainable:
                frozen_layers.append(idx)

        if trainable:
            regular_params = []
            for idx in layer_indices:
                for name, param in model.module_list[idx].named_parameters():
                    if 'offset_conv' in name:
                        dcn_offset_params.append(param)
                    elif 'deform_conv' in name:
                        dcn_conv_params.append(param)
                    else:
                        regular_params.append(param)

            if regular_params:
                param_groups.append({"params": regular_params, "name": group_name, "lr_scale": lr_scale})

    # Add DCN parameter groups with lower learning rates
    if dcn_offset_params:
        param_groups.append({
            "params": dcn_offset_params,
            "name": "dcn_offset",
            "lr_scale": dcn_lr_mult  # e.g., 0.1 = 10x lower LR
        })

    if dcn_conv_params:
        param_groups.append({
            "params": dcn_conv_params,
            "name": "dcn_conv",
            "lr_scale": dcn_lr_mult * 5  # e.g., 0.5 = 2x lower LR
        })

    # Freeze BatchNorm layers in frozen modules by setting them to eval mode
    # This prevents running_mean and running_var from updating
    def freeze_bn(module):
        for child in module.modules():
            if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                child.eval()
                # Also ensure BN params don't get gradients
                child.weight.requires_grad = False
                child.bias.requires_grad = False

    for idx in frozen_layers:
        freeze_bn(model.module_list[idx])

    # Register a hook to keep frozen BN layers in eval mode during training
    # This is needed because model.train() would otherwise set them back to train mode
    def freeze_bn_hook(module, input):
        for idx in frozen_layers:
            for child in model.module_list[idx].modules():
                if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    child.eval()

    # Store frozen layers info on model for the hook
    model._frozen_layer_indices = frozen_layers
    model.register_forward_pre_hook(freeze_bn_hook)

    # Safety: if nothing selected for training, raise
    if len(param_groups) == 0:
        raise ValueError("No trainable parameter groups selected by trainable_cfg!")

    return param_groups    