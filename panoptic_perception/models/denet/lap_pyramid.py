import cv2

import torch
import torch.nn as nn

from typing import List

class LapPyramidConv(nn.Module):
    def __init__(self, num_high:int=3, kernel_size:int=5, channels=3):
        super(LapPyramidConv, self).__init__()
        
        self.num_high = num_high
        self.kernel_size = kernel_size

        self.kernel = self._gauss_kernel(kernel_size, channels)
        
    def _gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T
        ) # (kernel_size, kernel_size)

        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1
        ) #(channels, 1, kernel_size, kernel_size)

        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel
    
    def conv_gauss(self, x:torch.Tensor):
        n_channels, _, kw, kh = self.kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2), mode="reflect")
        x = torch.nn.functional.conv2d(x, self.kernel, groups=n_channels)
        return x
    
    def downsample(self, x:torch.Tensor):
        return x[:, :, ::2, ::2]
    
    def pyramid_down(self, x:torch.Tensor):
        return self.downsample(self.conv_gauss(x))
    
    def upsample(self, x:torch.Tensor):
        
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2), device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up)
    
    def pyramid_decom(self, x:torch.Tensor) -> List[torch.Tensor]:

        self.kernel = self.kernel.to(x.device)
        current = x
        pyramid = []
        
        for _ in range(self.num_high):
            down = self.pyramid_down(current) #(bs, 3, h // 2, h // 2)
            up = self.upsample(down) #(bs, 3, h, w) == current.shape

            diff = current - up 

            pyramid.append(diff)
            current = down
            
        pyramid.append(current)
        return pyramid
    
    def pyramid_recons(self, pyramid:List[torch.Tensor]):
        
        x = pyramid[0]
        for level in pyramid[1:]:
            up = self.upsample(x)
            x = up + level
        return x