import torch
import torch.nn as nn
import torchvision.ops as ops

import math
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
        
        # ----- ATSS ------
        self._anchor_proposals : List[torch.Tensor] = None
        self._proposal_shape = None 
        
        self._anchor_cxcy : List[torch.Tensor] = None
        self._anchor_wh : List[torch.Tensor] = None
        self._anchor_strides : List[torch.Tensor] = None

    def _build_anchor_proposals(self, x):

        proposals = []
        proposals_cxcy = []
        proposals_wh = []
        proposals_strides = []
        for i in range(self.num_layers):
            stride = self.stride[i]
            _, _, ny, nx = x[i].shape
            na = self.anchors[i].shape[0]
            device = x[i].device

            yv, xv = torch.meshgrid(
                torch.arange(ny, device=device),
                torch.arange(nx, device=device),
                indexing='ij'
            )

            centers = torch.stack([
                (xv + 0.5) * stride, (yv + 0.5) * stride
            ], dim=-1).float()
            centers = centers.reshape(-1, 2)

            # Anchor half-sizes: (na, 2)
            half_wh = self.anchors[i].to(device) / 2

            # Broadcast: centers (ny*nx, 1, 2) +/- half_wh (1, na, 2) → (ny*nx, na, 2)
            x1y1 = centers.unsqueeze(1) - half_wh.unsqueeze(0)
            x2y2 = centers.unsqueeze(1) + half_wh.unsqueeze(0)
            boxes = torch.cat([x1y1, x2y2], dim=-1) # (ny*nx, na, 4)

            # Flatten cell-major, anchor-minor: row0_col0_anch0, row0_col0_anch1, ...
            boxes = boxes.reshape(-1, 4) # (ny*nx*na, 4)
            proposals.append(boxes) 

            _cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2
            _wh = boxes[:, 2:] - boxes[:, :2]
            _anchor_stride = torch.full((boxes.shape[0],), float(self.stride[i].item()), device=device)

            proposals_cxcy.append(_cxcy)
            proposals_wh.append(_wh)
            proposals_strides.append(_anchor_stride)

        self._anchor_proposals = proposals
        self._anchor_cxcy = proposals_cxcy
        self._anchor_wh = proposals_wh
        self._anchor_strides = proposals_strides

    def forward(self, x:List[torch.Tensor], image_size:Tuple[int,int]):

        if self.stride is None:
            self.stride = torch.tensor(
                [image_size[0] // x[i].shape[2] for i in range(self.num_layers)],
                device=x[0].device
            )

        current_shape = tuple((xi.shape[2], xi.shape[3]) for xi in x)
        if self._proposal_shape != current_shape:
            self._build_anchor_proposals(x)
            self._proposal_shape = current_shape

        for i in range(self.num_layers):
            bs, _, ny, nx = x[i].shape
            x[i] = self.convs[i](x[i])
            x[i] = x[i].view(bs, -1, self.num_outputs, ny, nx).permute(0,1,3,4,2).contiguous()

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
        
        
# ------ CLRNet Lane Detection Modules ------

# adapted from: https://github.com/Turoad/CLRNet/blob/main/clrnet/models/heads/clr_head.py

class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, self.size)
        return x.flatten(2)

class SegDecoder(nn.Module):
    '''
    Optionaly seg decoder
    '''
    def __init__(self,
                 image_height,
                 image_width,
                 num_class,
                 prior_feat_channels=64,
                 refine_layers=3):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(prior_feat_channels * refine_layers, num_class, 1)
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = torch.nn.functional.interpolate(x,
                          size=[self.image_height, self.image_width],
                          mode='bilinear',
                          align_corners=False)
        return x

class LaneROIGather(nn.Module):
    
    def __init__(self, feat_channels, num_priors, refine_layers,
                fc_hidden_dim, sample_points, mid_channels=48):
        super(LaneROIGather, self).__init__()

        self.in_channels = feat_channels
        self.num_priors = num_priors
        
        self.f_query = nn.Sequential(
            nn.Conv1d(in_channels=num_priors,
                      out_channels=num_priors,
                      kernel_size=1,
                      stride=1,
                      padding=0, groups=num_priors),
            nn.ReLU()
        )

        self.f_key = ConvBlock(
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            batch_norm=True,
            activation_func="relu"
        )

        self.f_value = ConvBlock(
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            batch_norm=False,
            activation=False
        )
        
        self.W = nn.Conv1d(in_channels=num_priors,
                           out_channels=num_priors,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=num_priors)

        self.resize = FeatureResize()
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        
        self.feature_resize = FeatureResize()

        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        
        for i in range(refine_layers):
            self.convs.append(
                ConvBlock(
                    in_channels=feat_channels,
                    out_channels=mid_channels,
                    kernel_size=(9, 1),
                    padding=(4, 0)
                )
            )
            
            self.catconv.append(
                ConvBlock(
                    in_channels=mid_channels * (i + 1),
                    out_channels=feat_channels,
                    kernel_size=(9, 1),
                    padding=(4, 0)
                )
            )
            
        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)
        
    def roi_feat(self, prior_features_stages, layer_idx) -> torch.Tensor:

        feats = [] #accumulate features , [layer_1] [layer_1, layer_2] [layer_1, layer_2, layer_3]
        for i, feature in enumerate(prior_features_stages):
            feat_trans = self.convs[i](feature) #(192, 64, 36, 1) -> #(192, 48, 36, 1)
            feats.append(feat_trans)
        
        cat_feat = torch.cat(feats, dim=1) 
        cat_feat = self.catconv[layer_idx](cat_feat) #(192, 48, 36, 1) -> (192, 64, 36, 1)

        return cat_feat
        
    def forward(self, prior_feature_stages:List[torch.Tensor], x:List[torch.Tensor], layer_index:int):
        '''
        Args:
            prior_feature_stages: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)
            x: feature map
            layer_index: currently on which layer to refine
        Return: 
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        
        prior_feature_stages = self.roi_feat(prior_feature_stages, layer_index)
        batch_size = x.shape[0]
        
        # torch.Size([192, 64, 36, 1]) -> torch.Size([192, 2304])
        prior_feature_stages = prior_feature_stages.contiguous().view(batch_size * self.num_priors, -1)

        # torch.Size([192, 2304]) -> torch.Size([192, 64])
        prior_feature_stages = torch.nn.functional.relu(self.fc_norm(self.fc(prior_feature_stages)))
        prior_feature_stages = prior_feature_stages.view(batch_size, self.num_priors, -1) #torch.Size([1, 192, 64])

        query = prior_feature_stages

        # torch.Size([1, 64, 20, 20]) -> torch.Size([1, 64, 20, 20]) -> torch.Size([1, 64, 250])
        value = self.resize(self.f_value(x))
        query = self.f_query(query) # (bs, 192, 64)
        key = self.resize(self.f_key(x)) #(bs, 64, 250)
        
        sim_map = torch.bmm(query, key) #(bs, 192, 250)
        sim_map = (self.in_channels**-.5) * sim_map
        
        sim_map = torch.nn.functional.softmax(sim_map, dim=-1)
        
        context = torch.bmm(sim_map, value.permute(0, 2, 1)) #(bs, 192, 64)
        context = self.W(context)
        
        prior_feature_stages = prior_feature_stages + torch.nn.functional.dropout(context, p=0.1, training=self.training)
        return prior_feature_stages

class LaneDetect(nn.Module):
    def __init__(self, 
                in_channels:List[int],
                img_h: int = 768,
                img_w: int = 1280,
                num_points: int = 72,
                num_priors: int = 192,
                feat_channels: int = 64,
                sample_points: int = 36,
                refine_layers: int = 3,
                num_fc: int = 2,
                num_classes:int = 2,
                num_categories: int = 4,
                use_seg_decoder:bool = True):
        super(LaneDetect, self).__init__()
        
        self.img_h = img_h
        self.img_w = img_w
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.feat_channels = feat_channels
        self.num_categories = num_categories
        self.use_seg_decoder = use_seg_decoder
        
        # -- Fixed buffers --
        self.register_buffer(
            'sample_x_indexs',
            (torch.linspace(0, 1, steps=sample_points) * self.n_strips).long()
        ) # torch.Size([36])

        self.register_buffer(
            'prior_feat_ys',
            torch.flip(1 - torch.linspace(0, 1, steps=sample_points), dims=[-1])
        ) # self.prior_feat_ys torch.Size([36])
        
        self.register_buffer(
            'prior_ys',
            torch.linspace(1, 0, steps=self.n_offsets)
        )        
        
        # -- Channel projection (FPN channels -> 64) --
        self.channel_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, feat_channels, 1, bias=False),
                nn.BatchNorm2d(feat_channels),
                nn.SiLU(inplace=True)
            ) for ch in in_channels
        ])
        
        self._init_prior_embeddings()

        init_priors, priors_on_featmap = self._generate_priors_from_embeddings()
        self.register_buffer('priors', init_priors)
        self.register_buffer('priors_on_featmap', priors_on_featmap)
        
        # -- ROIGather --
        self.roi_gather = LaneROIGather(
            feat_channels, num_priors, refine_layers,
            fc_hidden_dim=feat_channels,
            sample_points=sample_points
        )
        
        self.seg_decoder = None
        if self.use_seg_decoder:
            self.seg_decoder = SegDecoder(
                image_height=img_h,
                image_width=img_w, 
                num_class=num_classes,
                prior_feat_channels=self.feat_channels,
                refine_layers=self.refine_layers
            )
        
        # -- Classification branch --
        cls_modules = []
        for _ in range(num_fc):
            cls_modules.extend([
                nn.Linear(feat_channels, feat_channels),
                nn.LayerNorm(feat_channels),
                nn.SiLU(inplace=True)
            ])
        self.cls_modules = nn.Sequential(*cls_modules)
        self.cls_layers = nn.Linear(feat_channels, 2)
        
        # -- Regression branch --
        reg_modules = []
        for _ in range(num_fc):
            reg_modules.extend([
                nn.Linear(feat_channels, feat_channels),
                nn.LayerNorm(feat_channels),
                nn.SiLU(inplace=True)
            ])
        self.reg_modules = nn.Sequential(*reg_modules)
        self.reg_layers = nn.Linear(feat_channels, self.n_offsets + 1 + 2 + 1)
        
        # -- Category branch (BDD100K addition) --
        self.category_layers = nn.Linear(feat_channels, num_categories)
        self._init_weights()

    def _init_weights(self):
        for m in self.cls_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        for m in self.reg_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)
        for m in self.category_layers.parameters():
            nn.init.normal_(m, mean=0., std=1e-3)        
        
    def _init_prior_embeddings(self):
        
        # [start_y, start_x, theta] -> all normalize
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8

        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)

        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.prior_embeddings.weight[i, 0], 0.)
            nn.init.constant_(self.prior_embeddings.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.2 * (i % 4 + 1))

        for i in range(left_priors_nums + bottom_priors_nums, self.num_priors):
            nn.init.constant_(
                self.prior_embeddings.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *
                strip_size)
            nn.init.constant_(self.prior_embeddings.weight[i, 1], 1.)
            nn.init.constant_(self.prior_embeddings.weight[i, 2],
                              0.68 if i % 2 == 0 else 0.84)        
                        
    def _generate_priors_from_embeddings(self):
        """
        Convert 3-param embeddings into full prior vectors with 72 x-coordinates.
        """
        predictions = self.prior_embeddings.weight  # (192, 3)

        priors = predictions.new_zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets)
        )
        priors[:, 2:5] = predictions.clone()

        start_x = priors[:, 3].unsqueeze(1).repeat(1, self.n_offsets)
        start_y = priors[:, 2].unsqueeze(1).repeat(1, self.n_offsets)
        theta = priors[:, 4].unsqueeze(1).repeat(1, self.n_offsets)
        ys = self.prior_ys.repeat(self.num_priors, 1)

        priors[:, 6:] = (
            start_x * (self.img_w - 1) +
            ((1 - ys - start_y) * self.img_h /
            torch.tan(theta * torch.pi + 1e-5))
        ) / (self.img_w - 1)

        priors_on_featmap = priors[..., 6 + self.sample_x_indexs].clone()
        return priors, priors_on_featmap
    
    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        """
        Sample features along each anchor's x-positions using grid_sample.

        Args:
            batch_features: (B, feat_channels, H_feat, W_feat)
            prior_xs: (B, num_priors, sample_points) normalized x-coords [0,1]

        Returns:
            (B * num_priors, feat_channels, sample_points, 1)
        """
        batch_size = batch_features.shape[0]
        # (bs, 192, 36) -> (bs, 192, 36, 1)
        prior_xs = prior_xs.view(batch_size, num_priors, -1, 1)
        
        # self.prior_feat_ys - torch.Size([36])
        # prior_ys - torch.Size([bs, 192, 36, 1])
        prior_ys = self.prior_feat_ys.repeat(batch_size * num_priors).view(
            batch_size, num_priors, -1, 1
        ) 

        # Convert [0,1] -> [-1,1] for grid_sample
        grid_xs = prior_xs * 2.0 - 1.0
        grid_ys = prior_ys * 2.0 - 1.0
        grid = torch.cat((grid_xs, grid_ys), dim=-1)  # (B, 192, 36, 2)

        # Total: 192 × 36 = 6,912 bilinear lookups
        # Each lookup returns 64 channel values
        # Output: (bs, 64, 192, 36)
            # For each of these 36 coordinates, grid_sample does bilinear interpolation on the 20x20 feature map and returns all 64 channels
        feature = torch.nn.functional.grid_sample(
            batch_features, grid, align_corners=True
        ).permute(0, 2, 1, 3)

        feature = feature.reshape(
            batch_size * num_priors, self.feat_channels, self.sample_points, 1
        )
        return feature
    
    def forward(self, features: List[torch.Tensor], image_size):
        """
        Args:
            features: list of 3 FPN feature maps [P3, P4, P5]
                    shapes: [(B,256,H/8,W/8), (B,512,H/16,W/16), (B,512,H/32,W/32)]

        Returns:
            dict with 'predictions_lists' (per-stage) and 'category_logits'
        """    
        
        if image_size and image_size != (self.img_h, self.img_w):
            self.img_h = image_size[0]
            self.img_w = image_size[1]
            
            if self.use_seg_decoder and self.training:
                self.seg_decoder.image_height = self.img_h
                self.seg_decoder.image_width = self.img_w

        # Project all FPN features to feat_channels (64)
        batch_features = []
        for i, feat in enumerate(features):
            batch_features.append(self.channel_projections[i](feat))
            
        # Process coarse -> fine (P5 -> P4 -> P3)
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]
        
        if self.training:
            # Priors: torch.Size([192, 78]) Priors_on_featmap: torch.Size([192, 36])
            self.priors, self.priors_on_featmap = self._generate_priors_from_embeddings()

        priors = self.priors.repeat(batch_size, 1, 1) #torch.Size([bs, 192, 78])
        priors_on_featmap = self.priors_on_featmap.repeat(batch_size, 1, 1) #torch.Size([bs, 192, 36])

        predictions_lists = []
        prior_features_stages = [] # iterative refine

        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = torch.flip(priors_on_featmap, dims=[2])

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs
            ) 

            #List[(bs, 64, 192, 36)]
            prior_features_stages.append(batch_prior_features)

            # # ROIGather: cross-attention
            fc_features = self.roi_gather(
                prior_features_stages, batch_features[stage], stage
            ) # (bs, 192, 64)
            
            # (bs * 192, 64)
            fc_features = fc_features.view(num_priors, batch_size, -1).reshape(batch_size * num_priors, self.feat_channels)
            
            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)
                
            cls_logits = self.cls_layers(cls_features) ## (bs * 192, 2)
            reg = self.reg_layers(reg_features) # (bs * 192, 76)
            
            cls_logits = cls_logits.reshape(
                batch_size, -1, cls_logits.shape[-1]
            ) # (bs , 192, 2)
            reg = reg.reshape(batch_size, -1, reg.shape[-1]) # (bs , 192, 76)
            
            predictions = priors.clone() #(bs, 192, 78)
            predictions[:, :, :2] = cls_logits
            
            predictions[:, :, 2:5] += reg[:, :, :3] # # also reg theta angle here
            predictions[:, :, 5] = reg[:, :, 3] # length
            
            def trans_tensor(t):
                return t.unsqueeze(2).clone().repeat(1, 1, self.n_offsets)
            
            predictions[..., 6:] = (
                trans_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.repeat(batch_size, num_priors, 1) -
                  trans_tensor(predictions[..., 2])) * self.img_h /
                 torch.tan(trans_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]

            predictions_lists.append(predictions) #(bs, 192, 78)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors[..., 6 + self.sample_x_indexs]

        seg_logits = None
        if self.use_seg_decoder and self.training:
            target_size = batch_features[-1].shape[2:]
            seg_features = torch.cat([
                torch.nn.functional.interpolate(feat, size=target_size,
                                                mode='bilinear', align_corners=False)
                for feat in batch_features
            ], dim=1)
            seg_logits = self.seg_decoder(seg_features)

        return predictions_lists, seg_logits

    def activation(self, output:List[torch.Tensor]):
        """
        Apply activations to raw lane detection logits (pre-NMS).
        output: List[(bs, 192, 78)] per refinement stage
        Returns: (bs, 192, 78) from final stage, with softmax on cls.

        Note: x-offset and length scaling happens in lane_nms / predictions_to_pred,
        matching the official CLRNet get_lanes flow.
        """
        predictions = output[-1]                                       # (bs, 192, 78) final stage
        activated = predictions.clone()
        activated[:, :, :2] = torch.softmax(predictions[:, :, :2], dim=-1)  # bg/fg probs
        return activated

    def predictions_to_pred(self, predictions):
        """
        Convert NMS-surviving predictions to pixel-space lane coordinates.
        predictions: (K, 78) — [bg, fg, start_y, start_x, theta, length, x0..x71]
                     cls already softmaxed, geometry in normalized space.
        Returns: list of (valid_points, 2) arrays in pixel coords [(x, y), ...].
        """
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # (72,) normalized x
            start_y = lane[2]
            length = int(torch.round(lane[5] * self.n_strips).item())

            if length == 0:
                continue

            # start index in the 72-point grid
            start = min(max(0, int(torch.round(start_y * self.n_strips).item())),
                        self.n_strips)
            end = min(start + length + 1, self.n_offsets)

            xs_px = lane_xs[start:end] * (self.img_w - 1)
            ys_px = self.prior_ys[start:end] * (self.img_h - 1)

            # filter invalid points
            valid = xs_px >= 0
            if valid.sum() < 2:
                continue

            points = torch.stack([xs_px[valid], ys_px[valid]], dim=1)  # (P, 2)
            lanes.append(points)

        return lanes