import torch
import torch.nn as nn
import torchvision.ops as ops

from typing import List, Tuple
from functools import lru_cache

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=1, stride:int=1, padding:int=None, 
                activation:bool=True, batch_norm:bool=True, activation_func:str="hardswish"):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=self.autopad(kernel_size, padding))        

        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        if activation:
            if activation_func == "silu":
                self.act = nn.SiLU()
            elif activation_func == "hardswish":
                self.act = nn.Hardswish()
            else:
                self.act = nn.ReLU()        
        else:
            self.act = nn.Identity()

        # self.act = nn.Hardswish() if activation else nn.Identity()
        
    def autopad(self, kernel_size:int, padding:int=None):
        if padding is None:
            padding = kernel_size // 2
        return padding

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, 
                kernel_size:int=3, stride:int=1, padding:int=1, 
                dilation:int=1, groups:int=1,
                activation:bool=True, batch_norm:bool=True):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)        

        # offset predictions (2 values per kernel position: dx, dy)
        # offsets need bias for learning
        self.offset_conv = nn.Conv2d(in_channels, 
                                    2 * self.kernel_size[0] * self.kernel_size[1], 
                                    kernel_size=self.kernel_size, 
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = ops.DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = nn.SiLU() if activation else (activation if isinstance(activation, nn.Module) else nn.Identity())

    def _init_weights(self):

        # Zero-init offsets for stable training (acts like standard conv initially)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.deform_conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.bn(x)
        x = self.act(x)
        return x

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
    def __init__(self, in_channels:int, out_channels:int, residual:bool=True, use_deform:bool=False):
        super(Bottleneck, self).__init__()

        c_ = in_channels // 2

        self.conv1 = ConvBlock(in_channels, c_, kernel_size=1, stride=1)
        if use_deform:
            self.conv2 = DeformableConv2d(c_, out_channels, kernel_size=3, stride=1)
        else:
            self.conv2 = ConvBlock(c_, out_channels, kernel_size=3, stride=1)

        self.add = (in_channels == out_channels) and residual

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class BottleneckCSP(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, n:int, residual:bool=True, use_deform:bool=False):
        super(BottleneckCSP, self).__init__()
        
        c_ = in_channels // 2
        
        self.conv1 = ConvBlock(in_channels, c_, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, c_, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, kernel_size=1, stride=1, bias=False)
        self.conv4 = ConvBlock(2 * c_, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1)
        self.bottlenecks = nn.Sequential(*[Bottleneck(c_, c_, residual, use_deform) for _ in range(n)])

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

class BottleneckV8(nn.Module):
    def __init__(self, in_channels:int, out_channels, residual:bool=True, e=0.5):
        super(BottleneckV8, self).__init__()

        c_ = int(in_channels * e) #expansion_factor
        self.conv1 = ConvBlock(in_channels, c_,
                               kernel_size=3, stride=1, padding=1, 
                               activation=True, activation_func="silu")
        self.conv2 = ConvBlock(c_, out_channels, 
                               kernel_size=3, stride=1, padding=1,
                               activation=True, activation_func="silu")
        self.add = residual and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))
    
class C2F(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, n:int, residual:bool=True, e=0.5):
        super(C2F, self).__init__()
        
        _c1 = int(out_channels * e)
        self.conv1 = ConvBlock(in_channels, _c1 * 2, 
                               kernel_size=1, stride=1, padding=0,
                               activation=True, activation_func="silu")

        self.bottlenecks = nn.Sequential(*[
            BottleneckV8(_c1, _c1, residual=residual, e=1.0) for _ in range(n)
        ])

        _c2 = (2 + n) * _c1
        self.conv2 = ConvBlock(_c2, out_channels, 
                               kernel_size=1, stride=1, padding=0,
                               activation=True, activation_func="silu")
        
    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1)) #split num_chunks and channel_dim
        y.extend(bottlneck(y[-1]) for bottlneck in self.bottlenecks)
        return self.conv2(torch.cat(y, dim=1))
    
class SPPF(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, k=5):
        super(SPPF, self).__init__()
        
        c_ = in_channels // 2
        self.conv1 = ConvBlock(in_channels, c_, 
                               kernel_size=1, stride=1, padding=0,
                               activation=True, activation_func="silu")

        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

        self.conv2 = ConvBlock(c_ * 4, out_channels,
                               kernel_size=1, stride=1, padding=0,
                               activation=True, activation_func="silu")        
        
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        y4 = self.maxpool(y3)

        y = torch.cat([y1, y2, y3, y4], dim=1)        

        return self.conv2(y)
    
class DetectV8(nn.Module):
    def __init__(self, num_classes:int, in_channels:List[int], reg_max:int=16):
        super(DetectV8, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(in_channels)
        
        self.bbox_branch = nn.ModuleList()
        self.cls_branch = nn.ModuleList()
        
        for i, in_ch in enumerate(in_channels):
            c2 = max(reg_max, in_ch // 4, reg_max * 4)
            c3 = max(in_ch, min(self.num_classes, 100))
            
            bbox_block = nn.Sequential(
                ConvBlock(in_channels=in_ch, out_channels=c2, 
                        kernel_size=3, stride=1, padding=1,
                        activation=True, activation_func="silu"),
                ConvBlock(in_channels=c2, out_channels=c2, 
                        kernel_size=3, stride=1, padding=1,
                        activation=True, activation_func="silu"),
                nn.Conv2d(in_channels=c2, out_channels=reg_max*4, 
                          kernel_size=1, stride=1, padding=0)
            )
            
            cls_block = nn.Sequential(
                ConvBlock(in_channels=in_ch, out_channels=c3,
                        kernel_size=3, stride=1, padding=1,
                        activation=True, activation_func="silu"),
                ConvBlock(in_channels=c3, out_channels=c3,
                        kernel_size=3, stride=1, padding=1,
                        activation=True, activation_func="silu"),
                nn.Conv2d(in_channels=c3, out_channels=self.num_classes, 
                        kernel_size=1, stride=1, padding=0)
            )
            
            self.bbox_branch.append(bbox_block)
            self.cls_branch.append(cls_block)
            
    @lru_cache(maxsize=128)
    def compute_anchors(self, h:int, w:int, stride:int):
        
        #create meshgrid
        y, x = torch.meshgrid(
            torch.arange(h),
            torch.arange(w),
            indexing='ij'
        )
        
        #stack into (H*W, 2)
        grid = torch.stack((x, y), dim=-1).reshape(-1, 2).float()

        #Convert to pixel center coordinates
        anchor_tensor = (grid + 0.5) * stride
        stride_tensor = torch.full((h*w, 1), stride)

        return anchor_tensor, stride_tensor

    def forward(self, x:List[torch.Tensor], image_size:Tuple[int, int]):

        bbox_outputs = []
        cls_outputs = []
        anchor_points = []
        strides = []

        for i in range(self.num_layers):
            bs, _, ny, nx = x[i].shape
            bbox_output = self.bbox_branch[i](x[i])
            cls_output = self.cls_branch[i](x[i])            

            stride = image_size[0] // x[i].shape[2]
            anchor_tensor, stride_tensor = self.compute_anchors(
                ny, nx, stride
            )

            bs, _, gy, gx = bbox_output.shape
            bs, _, gy, gx = cls_output.shape

            bbox_outputs.append(bbox_output.view(bs, gy*gx, -1))
            cls_outputs.append(cls_output.view(bs, gy*gx, -1))
            anchor_points.append(anchor_tensor)
            strides.append(stride_tensor)

        bbox_outputs = torch.cat(bbox_outputs, dim=1)
        cls_outputs = torch.cat(cls_outputs, dim=1)
        anchor_points = torch.cat(anchor_points, dim=0).to(bbox_outputs.device)
        strides = torch.cat(strides, dim=0).to(bbox_outputs.device)

        return bbox_outputs, cls_outputs, anchor_points, strides
    
    def activation(self, bbox_logits_raw:torch.Tensor, 
                   cls_logits_raw:torch.Tensor,
                   anchor_points:torch.Tensor,
                   strides:torch.Tensor, 
                   xyxy2xywh:bool=True):
        """
        Decode predictions to image-space boxes (pixels).

        Args:
            bbox_logits_raw: [B, 8400, 64] raw distribution logits
            cls_logits_raw:  [B, 8400, C]  raw class logits
            anchor_points:   [8400, 2]     pixel coords
            strides:         [8400, 1]

        Returns:
            [B, 8400, 4+C]  → (x, y, w, h, cls1, cls2, ...) in pixel coords        
        
        """

        device = bbox_logits_raw.device

        bs, num_dets, _ = bbox_logits_raw.shape
        reg_max = bbox_logits_raw.shape[-1] // 4
        
        pred_distri_logits = bbox_logits_raw.view(bs, num_dets, 4, reg_max)
        pred_distri = torch.softmax(pred_distri_logits, dim=-1)
        
        project = torch.arange(pred_distri.shape[-1], dtype=pred_distri.dtype, device=device)
        pred_ltrb = (pred_distri * project).sum(dim=-1)
        
        anchor = anchor_points.unsqueeze(0)
        stride = strides.unsqueeze(0)
        
        x1 = anchor[..., 0] - pred_ltrb[..., 0] * stride.squeeze(-1)
        y1 = anchor[..., 1] - pred_ltrb[..., 1] * stride.squeeze(-1)
        x2 = anchor[..., 0] + pred_ltrb[..., 2] * stride.squeeze(-1)
        y2 = anchor[..., 1] + pred_ltrb[..., 3] * stride.squeeze(-1)
        
        cls_scores = cls_logits_raw.sigmoid()
        
        if xyxy2xywh:
            # xyxy → xywh (pixel coords, matching Detect.activation output format)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            # Concat: [B, 8400, 4+C] → (cx, cy, w, h, cls1, cls2, ...)
            return torch.cat([cx.unsqueeze(-1), cy.unsqueeze(-1),
                            w.unsqueeze(-1), h.unsqueeze(-1),
                            cls_scores], dim=-1)

        return torch.cat([x1.unsqueeze(-1), y1.unsqueeze(-1),
                          x2.unsqueeze(-1), y2.unsqueeze(-1),
                          cls_scores], dim=-1)