import torch
import torch.nn as nn

from .lap_pyramid import LapPyramidConv
from .trans_low import TransLow
from .trans_high import TransHigh, UpGuide

class DENet(nn.Module):
    def __init__(self, num_high:int=3, gaussian_kernel_size:int=5,
                num_channels:int=3, channel_blocks:int=64, channel_mask:int=16,
                up_kernel_size:int=1, high_channels:int=32, high_kernel_size:int=3):
        super(DENet, self).__init__()

        self.num_high = num_high
        self.num_channels = num_channels
        
        self.lap_pyramid = LapPyramidConv(num_high, gaussian_kernel_size, channels=num_channels)
        self.trans_low = TransLow(num_channels, channel_blocks, channel_mask)
        
        self.trans_high = nn.ModuleList()
        self.up_guide = nn.ModuleList()
        
        for i in range(self.num_high):
            self.up_guide.append(
                UpGuide(up_kernel_size, ch=num_channels)
            )
            self.trans_high.append(
                TransHigh(num_channels, high_channels, high_kernel_size)
            )

    def forward(self, x:torch.Tensor):
        pyrs = self.lap_pyramid.pyramid_decom(x)        
        
        trans_pyrs = []
        trans_pyr, guide = self.trans_low(pyrs[-1])
        trans_pyrs.append(trans_pyr)
        
        for i in range(self.num_high):
            guide = self.up_guide[i](guide)
            trans_pyr = self.trans_high[i](
                pyrs[-2-i], guide
            )
            
            trans_pyrs.append(trans_pyr)
            
        out = self.lap_pyramid.pyramid_recons(trans_pyrs)        

        return out