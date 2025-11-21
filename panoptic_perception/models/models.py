import os, ast
from typing import Tuple

import torch
import torch.nn as nn

from panoptic_perception.models.common import ConvBlock, Focus, BottleneckCSP, SPP, Upsample, Detect
from panoptic_perception.models.utils import parse_model_config, initialize_weights, PanopticModelOutputs
from panoptic_perception.utils.detection_utils import DetectionLossCalculator
from panoptic_perception.utils.segmentation_utils import SegmentationLossCalculator

def create_modules(module_defs: list, num_classes: int = 80,
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
            module = ConvBlock(in_ch, out_ch, k, s)
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
            module = BottleneckCSP(in_ch, out_ch, n, residual)
            output_channels.append(out_ch)

        # -- SPP ---------------------------------------------------------------
        elif mtype == "SPP":
            in_ch = output_channels[-1] if len(output_channels) > 0 else int(module_def["in_channels"])
            out_ch = int(module_def["out_channels"])
            
            ks = [int(x) for x in module_def.get("pool_sizes", "5,9,13").split(",")]
            module = SPP(in_ch, out_ch, ks)
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

        # -- HEADS (skip building) ---------------------------------------------
        elif mtype == "Detect":
            num_classes = int(module_def.get("num_classes"))
            anchors, _ = ast.literal_eval(module_def["anchors"])
            channels = [output_channels[r] if r != -1 else output_channels[-1] for r in route]

            module = Detect(anchors, num_classes, channels)
            

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
    def __init__(self, cfg: str, img_size: int = 416, num_classes: int = 80):
        super(YOLOP, self).__init__()

        module_defs = parse_model_config(cfg)            
        
        self.img_size = img_size
        self.num_classes = num_classes

        if module_defs[0]['type'] == 'heads':
            self.module_defs = module_defs[1:]

            self.segmentation_head_idx = int(module_defs[0].get('segmentation_head_idx', -1))
            self.lane_segmentation_head_idx = int(module_defs[0].get('lane_segmentation_head_idx', -1))
            self.detection_head_idx = int(module_defs[0].get('detection_head_idx', -1))

        self.in_channels = self.module_defs[0].get('channels', 3)
        self.module_list, self.routes, self.module_names, self._cache_layer_idx = create_modules(
            self.module_defs,
            num_classes=num_classes,
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
                    model_outputs.drivable_segmentation_predictions = torch.sigmoid(x)
                # print(f'Layer: {i} - Module Name: {self.module_names[i]} (Drivable Area Seg Head) - output shape: {x.shape}')

            elif self.module_names[i] == "LaneSegmentation":
                model_outputs.lane_segmentation_logits = x
                if not self.training:
                    model_outputs.lane_segmentation_predictions = torch.sigmoid(x)

                # print(f'Layer: {i} - Module Name: {self.module_names[i]} (Lane Seg Head) - output shape: {x.shape}')

            # else:
            #     # print(f'Layer: {i} - Module Name: {self.module_names[i]} - output shape: {x.shape}')

            if i in self._cache_layer_idx:
                cache[i] = x

        
        if targets is not None:
            
            if model_outputs.detection_logits is not None:
                output_name = "detections"
                assert output_name in targets, f"Target for {output_name} not provided."

                # det_loss, det_loss_items = DetectionLossCalculator.compute_detection_loss(
                #     model_outputs.detection_logits,
                #     targets["detections"],
                #     num_anchors=len(self.module_list[self.detection_head_idx].anchors[0]),
                #     anchors_tensor=self.module_list[self.detection_head_idx].anchors,
                #     strides=[width/x.shape[2] for x in model_outputs.detection_logits]
                # )

                det_loss, det_loss_items = DetectionLossCalculator.compute_detection_loss_2(
                    model_outputs.detection_logits,
                    targets["detections"],
                    self.module_list[self.detection_head_idx].num_layers,
                    self.module_list[self.detection_head_idx].anchors,
                    self.module_list[self.detection_head_idx].stride
                )

                model_outputs.detection_loss = det_loss
                # print(f'Detection Loss: {det_loss}')

            if model_outputs.drivable_segmentation_logits is not None:
                output_name = "drivable_area_seg"
                assert output_name in targets, f"Target for {output_name} not provided."
                drivable_seg_loss = SegmentationLossCalculator.compute_segmentation_loss(
                    model_outputs.drivable_segmentation_logits,
                    targets["drivable_area_seg"]
                    )
                model_outputs.drivable_segmentation_loss = drivable_seg_loss
                # print(f'Drivable Area Segmentation Loss: {drivable_seg_loss}')
                
            if model_outputs.lane_segmentation_logits is not None:
                output_name = "lane_seg"
                assert output_name in targets, f"Target for {output_name} not provided."

                lane_seg_loss = SegmentationLossCalculator.compute_segmentation_loss(
                    model_outputs.lane_segmentation_logits,
                    targets["lane_seg"]
                )
                model_outputs.lane_segmentation_loss = lane_seg_loss
                # print(f'Lane Segmentation Loss: {lane_seg_loss}')

        return model_outputs
