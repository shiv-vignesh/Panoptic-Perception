import torch
import torch.nn as nn

from typing import List, Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=1, stride:int=1, padding:int=None, 
                activation:bool=True, batch_norm:bool=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=self.autopad(kernel_size, padding))        

        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = nn.Hardswish() if activation else nn.Identity()
        
    def autopad(self, kernel_size:int, padding:int=None):
        if padding is None:
            padding = kernel_size // 2
        return padding

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

class Focus(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=1, stride:int=1, padding:int=None):
        super(Focus, self).__init__()
        
        self.conv = ConvBlock(in_channels * 4, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        
        # x shape: (batch_size, channels, height, width)
        # Focus operation: slice and concatenate
        x1 = x[:, :, ::2, ::2]  #top-left
        x2 = x[:, :, 1::2, ::2] #bottom-left
        x3 = x[:, :, ::2, 1::2] #top-right
        x4 = x[:, :, 1::2, 1::2] #bottom-right

        x = torch.cat((x1, x2, x3, x4), dim=1)  # concatenate along channel dimension
        x = self.conv(x)

        return x 

class Bottleneck(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, residual:bool=True):
        super(Bottleneck, self).__init__()
        
        c_ = in_channels // 2
        
        self.conv1 = ConvBlock(in_channels, c_, kernel_size=1, stride=1)
        self.conv2 = ConvBlock(c_, out_channels, kernel_size=3, stride=1)
        self.add = (in_channels == out_channels) and residual
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class BottleneckCSP(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, n:int, residual:bool=True):
        super(BottleneckCSP, self).__init__()
        
        c_ = in_channels // 2
        
        self.conv1 = ConvBlock(in_channels, c_, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, c_, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, kernel_size=1, stride=1, bias=False)
        self.conv4 = ConvBlock(2 * c_, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1)
        self.bottlenecks = nn.Sequential(*[Bottleneck(c_, c_, residual) for _ in range(n)])

    def forward(self, x):
        y1 = self.conv3(self.bottlenecks(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
    
class SPP(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_sizes:list=[5, 9, 13]):
        super(SPP, self).__init__()
        
        c_ = in_channels // 2
        self.conv1 = ConvBlock(in_channels, c_, kernel_size=1, stride=1)
        self.conv2 = ConvBlock(c_ * (len(kernel_sizes) + 1), out_channels, kernel_size=1, stride=1)        
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [mp(x) for mp in self.maxpools], dim=1))

class Upsample(nn.Module):
    def __init__(self, scale_factor:int=2, mode:str='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):        
        return torch.functional.F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
    
class Detect(nn.Module):
    def __init__(self, anchors:List[Tuple[int]], num_classes:int, in_channels:List[int]):
        super().__init__()

        self.num_classes = num_classes
        self.num_outputs = num_classes + 5      # xywh + obj + classes
        self.num_layers = len(anchors)

        # anchors: list of N scale lists
        # Use register_buffer so anchors are saved/loaded with state_dict
        # Note: old checkpoints may have 'anchors_grid', new ones have 'anchors'
        anchors_tensor = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer("anchors", anchors_tensor)
        # Keep anchors_grid as alias for backward compatibility with old checkpoints
        self.register_buffer("anchors_grid", anchors_tensor)

        self.convs = nn.ModuleList()
        for i, in_ch in enumerate(in_channels):
            na = self.anchors[i].shape[0]
            out_ch = na * self.num_outputs
            self.convs.append(nn.Conv2d(in_ch, out_ch, 1, 1))

        self.stride = None


    def forward(self, x:List[torch.Tensor], image_size:Tuple[int,int]):

        for i in range(self.num_layers):
            bs, _, ny, nx = x[i].shape
            x[i] = self.convs[i](x[i])
            x[i] = x[i].view(bs, -1, self.num_outputs, ny, nx).permute(0,1,3,4,2).contiguous()

        if self.stride is None:
            self.stride = torch.tensor(
                [image_size[0] // x[i].shape[2] for i in range(self.num_layers)],
                device=x[0].device
            )

        return x

    def activation(self, x: List[torch.Tensor], use_yolop_style: bool = True):
        """
        Decode predictions to image-space boxes (pixels).

        Args:
            x: List of raw detection outputs per layer
            use_yolop_style: If True, use YOLOP-style decoding (sigmoid*2)^2
                            If False, use exp() style decoding
        """
        outputs = []
        for i in range(self.num_layers):
            bs, na, ny, nx, _ = x[i].shape
            stride = self.stride[i]

            # Grid
            yv, xv = torch.meshgrid(
                torch.arange(ny, device=x[i].device),
                torch.arange(nx, device=x[i].device),
                indexing='ij'
            )
            grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
            anchor = self.anchors[i].view(1, na, 1, 1, 2).to(x[i].device)

            if use_yolop_style:
                # YOLOP-style decoding
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride
                y[..., 2:4] = (y[..., 2:4] * 2.) ** 2 * anchor
            else:
                # Original exp() style decoding
                y = x[i].clone()
                y[..., 0:2] = (y[..., 0:2].sigmoid() + grid) * stride
                y[..., 2:4] = torch.exp(y[..., 2:4]) * anchor
                y[..., 4:] = y[..., 4:].sigmoid()

            outputs.append(y)

        return outputs
