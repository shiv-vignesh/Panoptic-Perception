import torch
import torch.nn as nn
import torchvision

import math

def tanh_range(x, left, right):
    return (torch.tanh(x) * 0.5 + 0.5) * (right - left) + left

def normalize(x, per_image=True):
    """
    With batch-global, the day image's normalization is dictated by the night image's minimum. 
    Change batch composition and the same image gets different enhancement.
    
    At batch_size=1 during inference vs batch_size=8 during training, you get different outputs for the same input — that's not reproducible.
    """    
    if per_image:
        mins = x.amin(dim=(1, 2, 3), keepdim=True)
        maxs = x.amax(dim=(1, 2, 3), keepdim=True)
    else:
        mins = x.min()
        maxs = x.max()
    return (x - mins) / (maxs - mins + 1e-8)

def identity(x:torch.Tensor, out_x:torch.Tensor, identity_gate:torch.Tensor):

    g = identity_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    x = (x*g) + ((1. - g) * out_x)
    return x

class GateModule(nn.Module):
    def __init__(self, latent_dim:int, num_gates:int):
        
        super(GateModule, self).__init__()
        self.gate_linear = nn.Linear(latent_dim, num_gates, bias=True)
        
    def forward(self, latent_embed:torch.Tensor):
        """
        Args:
            latent_embed - Output from the last layer of an encoder (b, latent_dim)
        Returns:
            gate outputs - (b, num_gates)
        """
        return tanh_range(self.gate_linear(latent_embed), 0.01, 1.0)

class WhiteBalance(nn.Module):
    def __init__(self, latent_dim:int, log_wb_range = 0.5):
        super(WhiteBalance, self).__init__()

        self.wb_linear = nn.Linear(latent_dim, 3, bias=True)
        self.log_wb_range = log_wb_range
        
    def forward(self, x:torch.Tensor, latent_embed:torch.Tensor, wb_gate:torch.Tensor):
        """
        Args:
            x - Image (b, 3, h, w)
            latent_embed - Output from the last layer of an encoder (b, latent_dim)
        Returns
            whiteBalancedImage - (b, 3, h, w)
        """

        wb = self.wb_linear(latent_embed)
        wb = torch.exp(tanh_range(wb, -self.log_wb_range, self.log_wb_range))

        color_scaling = 1./(1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] + 0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1) * wb

        wb_out = wb.unsqueeze(2).unsqueeze(3) * x
        wb_out = normalize(wb_out)
        wb_out = wb_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3) * wb_out

        return wb_out

class GammaBalance(nn.Module):
    def __init__(self, latent_dim:int):
        super(GammaBalance, self).__init__()
        
        self.gamma_linear = nn.Linear(latent_dim, 1, bias=True)
        self.register_buffer('log_gamma', torch.log(torch.tensor(2.5)))
        
    def forward(self, x:torch.Tensor, latent_embed:torch.Tensor, gamma_gate:torch.Tensor):
        
        gamma = self.gamma_linear(latent_embed).unsqueeze(2).unsqueeze(3)
        gamma = torch.exp(tanh_range(gamma, -self.log_gamma, self.log_gamma))
        
        g = torch.pow(x.clamp(min=1e-4), gamma)
        g = normalize(g)
        g = g * gamma_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        return g
        
class Sharpening(nn.Module):
    def __init__(self, latent_dim:int, kernel_size:int=13, sigma=(0.1, 5.0)):
        super(Sharpening, self).__init__()

        self.sharpen_linear = nn.Linear(latent_dim, 1, bias=True)
        self.blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)

    def forward(self, x:torch.Tensor, latent_embed:torch.Tensor, sharpning_gate:torch.Tensor):

        out_x = self.blur(x)

        y = self.sharpen_linear(latent_embed).unsqueeze(2).unsqueeze(3)
        y = tanh_range(y, 0.1, 1.0)
        s = x + (y*(x - out_x))
        s = normalize(s)
        s = s * (sharpning_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3))

        return s

class Defog(nn.Module):
    def __init__(self, latent_dim:int):
        super(Defog, self).__init__()
        
        self.defog_linear = nn.Linear(latent_dim, 1, bias=True)
    
    def atmospheric_light(self, x:torch.Tensor, dark:torch.Tensor=None, topk:int = 1000):

        if dark is not None:
            # Dark-channel guided: find haziest pixels, sample RGB at those locations
            dark_squeezed = dark.squeeze(1)           # [B, H, W]
            B, H, W = dark_squeezed.shape
            imsz = H * W
            numpx = max(imsz // topk, 1)

            dark_flat = dark_squeezed.reshape(B, imsz)
            _, indices = dark_flat.topk(numpx, dim=1)             # [B, numpx] — top-k haziest

            img_flat = x.reshape(B, 3, imsz)                      # [B, 3, H*W]
            indices_3ch = indices.unsqueeze(1).expand(-1, 3, -1)   # [B, 3, numpx]
            sampled = torch.gather(img_flat, 2, indices_3ch)       # [B, 3, numpx]

            a = sampled.mean(dim=2).unsqueeze(2).unsqueeze(3)      # [B, 3, 1, 1]

        else:
            # Per-channel: sort pixels descending, take top-k brightest, average
            r, _ = torch.sort(x[:, 0, :, :].reshape(x.shape[0], -1), descending=True)
            g, _ = torch.sort(x[:, 1, :, :].reshape(x.shape[0], -1), descending=True)
            b, _ = torch.sort(x[:, 2, :, :].reshape(x.shape[0], -1), descending=True)

            a = torch.cat([
                r[:, :topk].mean(dim=1, keepdim=True),
                g[:, :topk].mean(dim=1, keepdim=True),
                b[:, :topk].mean(dim=1, keepdim=True)
            ], dim=1).unsqueeze(2).unsqueeze(3)  # [B, 3, 1, 1]

        return torch.maximum(a, torch.tensor(0.01, device=x.device))

    def dark_channel(self, x:torch.Tensor):
        dark_i = x.min(dim=1)[0].unsqueeze(1)
        return dark_i

    def forward(self, x:torch.Tensor, latent_embed:torch.Tensor, fog_gate:torch.Tensor):
        """Defogging module is used for removing the fog from the image using Atmospheric Scattering Model.
            I(X) = (1-T(X)) * J(X) + T(X) * A(X)
                I(X) => image containing the fog.
                T(X) => Transmission map of the image.
                J(X) => True image Radiance.
                A(X) => Atmospheric scattering factor.

        Args:
            x (torch.tensor): Input image I(X)
            latent_out (torch.tensor): Feature representation from DIP Module.
            fog_gate (torch.tensor): Gate value raning from (0. - 1.) which enables defog module.

        Returns:
            torch.tensor : Returns defogged image with true image radiance.
        """

        omega = self.defog_linear(latent_embed).unsqueeze(2).unsqueeze(3)
        omega = tanh_range(omega, 0.1, 1.0)

        #atmospheric light + dark channel
        dark_i = self.dark_channel(x)
        atmos_light = self.atmospheric_light(x, dark=dark_i)
        i = x / atmos_light
        i = self.dark_channel(i)
        t = 1. - (omega * i)
        j = ((x - atmos_light)/(t.clamp(min=0.01))) + atmos_light
        j = normalize(j)

        j = j * fog_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return j
        
class Contrast(nn.Module):
    def __init__(self, latent_dim:int):
        super(Contrast, self).__init__()
        
        self.contrast_linear = nn.Linear(latent_dim, 1, bias=True)
        
    def rgb2lum(self, x:torch.Tensor):
        return 0.27 * x[:, 0, :, :] + 0.67 * x[:, 1, :,:] + 0.06 * x[:, 2, :, :]

    def lerp(self, a:torch.Tensor, b:torch.Tensor, l:torch.Tensor):
        return (1 - l.unsqueeze(2).unsqueeze(3)) * a + l.unsqueeze(2).unsqueeze(3) * b

    def forward(self, x:torch.Tensor, latent_embed:torch.Tensor, contrast_gate:torch.Tensor):
        
        alpha = torch.tanh(self.contrast_linear(latent_embed))
        
        luminance = self.rgb2lum(x).clamp(0.0, 1.0).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = self.lerp(x, contrast_image, alpha)
        contrast_image = normalize(contrast_image)
        contrast_image = contrast_image * contrast_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        
        return contrast_image        
        
class Tone(nn.Module):
    def __init__(self, latent_dim:int, curve_steps:int = 8):
        super(Tone, self).__init__()
        
        self.tone_linear = nn.Linear(latent_dim, 8, bias=True)
        self.curve_steps = curve_steps

    def forward(self, x:torch.Tensor, latent_embed:torch.Tensor, tone_gate:torch.Tensor):

        tone_curve = self.tone_linear(latent_embed).reshape(-1, 1, self.curve_steps)
        tone_curve = tanh_range(tone_curve,0.5, 2)
        tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30

        total_image = x * 0

        for i in range(self.curve_steps):
            total_image += torch.clamp(x - 1.0 * i /self.curve_steps, 0, 1.0 /self.curve_steps) \
                            * tone_curve[:,:,i].unsqueeze(2).unsqueeze(3)

        total_image *= self.curve_steps / tone_curve_sum.unsqueeze(2).unsqueeze(3)
        total_image = normalize(total_image)
        total_image = total_image * tone_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        return total_image         