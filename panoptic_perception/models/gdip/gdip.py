import torch
import torch.nn as nn

from typing import List

from panoptic_perception.models.gdip.vision_encoder import build_vision_encoder
from panoptic_perception.models.gdip.ip_operations import (
    GateModule, GammaBalance, WhiteBalance, Contrast, Tone, Defog, Sharpening, tanh_range, normalize, identity
)

class GDIP(nn.Module):
    def __init__(self, latent_dim:int, num_gates:int=7):
        super(GDIP, self).__init__()

        # self.vision_encoder = build_vision_encoder(encoder_cfg)
        # latent_dim = encoder_cfg["latent_dim"]

        self.gate_module = GateModule(latent_dim=latent_dim, num_gates=num_gates)

        self.wb_module = WhiteBalance(latent_dim=latent_dim)
        self.gamma_module = GammaBalance(latent_dim=latent_dim)
        self.contrast = Contrast(latent_dim=latent_dim)
        self.tone = Tone(latent_dim=latent_dim)
        self.defog = Defog(latent_dim=latent_dim)
        self.sharpening = Sharpening(latent_dim=latent_dim)        

    def forward(self, x:torch.Tensor, latent_out:torch.Tensor):
        """
        Args:
            x - (bs, 3, h, w)
        Returns:

        """

        # latent_out = torch.nn.functional.relu_(self.vision_encoder(x))
        latent_out = torch.nn.functional.relu(latent_out)
        gate = self.gate_module(latent_out)

        wb_out = self.wb_module(x, latent_out, gate[:, 0])
        gamma_out = self.gamma_module(x, latent_out, gate[:, 1])
        sharpening_out = self.sharpening(x, latent_out, gate[:, 3])
        defog_out = self.defog(x, latent_out, gate[:, 4])
        contrast_out = self.contrast(x, latent_out, gate[:, 5])
        tone_out = self.tone(x, latent_out, gate[:, 6])

        out_x = wb_out + gamma_out + defog_out + sharpening_out + contrast_out + tone_out
        out_x = normalize(out_x)

        x = identity(x, out_x, gate[:, 2])
        return x, gate
    
class MultiLevelGDIP(nn.Module):
    def __init__(self, num_gdip_blocks:int, latent_dim:int, num_gates:int=7):
        super(MultiLevelGDIP, self).__init__()
        
        self.num_gdip_blocks = num_gdip_blocks
        self.gdip_blocks = nn.ModuleList([GDIP(latent_dim, num_gates) for _ in range(num_gdip_blocks)])
    
    def forward(self, x:torch.Tensor, latent_features:List[torch.Tensor], return_intermediates:bool=False):

        assert len(latent_features) == self.num_gdip_blocks, f'Number of latent features: {len(latent_features)} must match number of GDIP blocks: {self.num_gdip_blocks}'

        out_images = []
        gates = []

        for idx, module in enumerate(self.gdip_blocks):
            latent_out = latent_features[idx]
            x, gate = module(x, latent_out)

            if return_intermediates:
                out_images.append(x)
            gates.append(gate)

        return x, gates, out_images