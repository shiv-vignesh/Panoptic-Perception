
from dataclasses import dataclass, asdict
from typing import List, Union

import os
import torch

@dataclass
class PanopticModelOutputs:
    
    detection_logits: List[torch.Tensor] = None
    drivable_segmentation_logits: torch.Tensor = None
    lane_segmentation_logits: torch.Tensor = None
    
    detection_predictions: List[torch.Tensor] = None
    drivable_segmentation_predictions: torch.Tensor = None
    lane_segmentation_predictions: torch.Tensor = None        
    
    detection_loss: torch.Tensor = None
    drivable_segmentation_loss: torch.Tensor = None
    lane_segmentation_loss: torch.Tensor = None

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

class WeightsManager:
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def load(self, model:torch.nn.Module, ckpt_path:str, strict=False):
        
        if not os.path.exists(ckpt_path):
            if self.verbose:
                print(f'Model Checkpoint not found: {ckpt_path}')
            return
        
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt

        # Backward compatibility: copy anchors_grid to anchors if only anchors_grid exists
        # This handles old checkpoints that saved anchors_grid but not anchors
        keys_to_add = {}
        for key in state_dict.keys():
            if 'anchors_grid' in key and key.replace('anchors_grid', 'anchors') not in state_dict:
                new_key = key.replace('anchors_grid', 'anchors')
                keys_to_add[new_key] = state_dict[key]
        state_dict.update(keys_to_add)

        missing, unexpected = model.load_state_dict(
            state_dict, strict=False
        )
        
        if strict:
            if len(missing) or len(unexpected):
                raise RuntimeError(
                    f"Strict loading failed.\n"
                    f"Missing keys: {missing}\nUnexpected keys: {unexpected}"
                )
        
        loaded_keys = [k for k in state_dict.keys() if k not in unexpected]

        if self.verbose:
            print("=== Weights Loaded ===")
            print(f"Loaded     : {len(loaded_keys)} keys")
            print(f"Missing    : {len(missing)} keys")
            print(f"Unexpected : {len(unexpected)} keys")

        return missing, unexpected, loaded_keys

def parse_model_config(model_config:str):

    if (type(model_config) == str) and model_config.endswith('.cfg'):    
        file = open(model_config, 'r')
        lines = file.read().split('\n')    
        lines = [x.rstrip().lstrip() for x in lines if x]
        
        module_defs = []
        
        for line in lines:
            if line.startswith("["):
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].rstrip()
            else:
                key, value = line.split("=")
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()
        
        return module_defs

    elif (type(model_config) == list):
        
        module_defs = []

        # --- Parse head indices (first entry)
        if len(model_config) > 0 and isinstance(model_config[0], list):
            head_indices = model_config[0]
            if len(head_indices) >= 3:
                module_defs.append({
                    "type": "heads",
                    "detection_head_idx": str(head_indices[0]),
                    "segmentation_head_idx": str(head_indices[1]),
                    "lane_segmentation_head_idx": str(head_indices[2]),
                })

        # --- Parse layers
        for idx, layer_def in enumerate(model_config[1:]):
            if not isinstance(layer_def, list) or len(layer_def) < 3:
                continue

            from_layer = layer_def[0]
            layer_type = layer_def[1]
            params = layer_def[2]

            # Map layer names (to match cfg format)
            if layer_type == "Conv":
                layer_type = "ConvBlock"

            module_dict = {
                "type": layer_type,
                "layer_idx": str(idx),
            }

            # --- Handle route / from layer(s)
            if isinstance(from_layer, list):
                module_dict["route"] = ",".join(str(x) for x in from_layer)
            else:
                if layer_type not in ["Upsample"]:  # Upsample doesn’t have a route
                    module_dict["route"] = str(from_layer)

            # --- Parameter parsing per type
            if layer_type == "Focus":
                if len(params) >= 3:
                    module_dict.update({
                        "in_channels": str(params[0]),
                        "out_channels": str(params[1]),
                        "kernel_size": str(params[2])
                    })

            elif layer_type == "ConvBlock":
                if len(params) >= 2:
                    module_dict["in_channels"] = str(params[0])
                    module_dict["out_channels"] = str(params[1])
                if len(params) >= 3 and params[2] is not None:
                    module_dict["kernel_size"] = str(params[2])
                if len(params) >= 4 and params[3] is not None:
                    module_dict["stride"] = str(params[3])

            elif layer_type == "BottleneckCSP":
                module_dict["in_channels"] = str(params[0])
                module_dict["out_channels"] = str(params[1])
                module_dict["num_bottlenecks"] = str(params[2]) if len(params) >= 3 else "1"
                if len(params) >= 4:
                    residual = params[3]
                else:
                    residual = True
                module_dict["residual"] = "True" if residual else "False"

            elif layer_type == "SPP":
                module_dict["in_channels"] = str(params[0])
                module_dict["out_channels"] = str(params[1])
                pool_sizes = params[2]
                if isinstance(pool_sizes, list):
                    module_dict["pool_sizes"] = ",".join(str(x) for x in pool_sizes)
                else:
                    module_dict["pool_sizes"] = str(pool_sizes)

            elif layer_type == "Upsample":
                if len(params) >= 2 and params[1] is not None:
                    module_dict["scale_factor"] = str(params[1])
                if len(params) >= 3:
                    module_dict["mode"] = str(params[2])

            elif layer_type == "Concat":
                if len(params) >= 1:
                    module_dict["dimension"] = str(params[0])

            elif layer_type == "Detect":
                module_dict["num_classes"] = str(params[0])
                anchors = params[1]
                channels = params[2]
                # Format anchors nicely
                if isinstance(anchors, list):
                    formatted_anchors = "[" + ", ".join(
                        "[" + ",".join(str(a) for a in group) + "]"
                        for group in anchors
                    ) + "]"
                else:
                    formatted_anchors = str(anchors)
                module_dict["anchors"] = f"{formatted_anchors}, [{', '.join(str(c) for c in channels)}]"

                if isinstance(from_layer, list):
                    module_dict["route"] = ",".join(str(x) for x in from_layer)

            else:
                # Generic parameter fallback
                for i, p in enumerate(params):
                    module_dict[f"param_{i}"] = str(p)

            module_defs.append(module_dict)

        return module_defs


def initialize_weights(model):
    import torch.nn as nn
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            pass
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # nn.init.constant_(m.weight, 0.0)
            # if m.bias is not None:
            #     nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
            # nn.init.constant_(m.weight, 1.0)
            # nn.init.constant_(m.bias, 0.0)
