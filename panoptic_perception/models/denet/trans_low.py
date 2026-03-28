import torch
import torch.nn as nn

from .attention import SpatialAttention, ChannelAttention

class TransGuide(nn.Module):
    def __init__(self, num_channels:int=3, ch=16,
                attention_kernel_size:int = 3):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(num_channels * 2, ch, 3, padding=1),
            nn.LeakyReLU(True),
            SpatialAttention(attention_kernel_size),
            nn.Conv2d(ch, num_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.layer(x)

class TransLow(nn.Module):
    def __init__(self, num_channels:int = 3, 
                channel_blocks:int=64, channel_mask:int=16,
                attention_kernel_size:int = 3):
        super(TransLow, self).__init__()

        self.encoder = torch.nn.Sequential(
            nn.Conv2d(num_channels, channel_mask, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_mask, channel_blocks, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.mm1 = nn.Conv2d(channel_blocks, 
                            channel_mask,
                            kernel_size=1, padding=0)

        self.mm2 = nn.Conv2d(channel_blocks, 
                            channel_mask,
                            kernel_size=3, padding=3//2)

        self.mm3 = nn.Conv2d(channel_blocks, 
                            channel_mask,
                            kernel_size=5, padding=5//2)

        self.mm4 = nn.Conv2d(channel_blocks, 
                            channel_mask,
                            kernel_size=7, padding=7//2)

        self.decoder = torch.nn.Sequential(nn.Conv2d(channel_blocks, channel_mask, kernel_size=3, padding=1),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(channel_mask, num_channels, kernel_size=3, padding=1))

        self.trans_guide = TransGuide(num_channels, channel_mask, attention_kernel_size)
        
    def forward(self, x:torch.Tensor):

        x1 = self.encoder(x)
        x1_1 = self.mm1(x1)
        x1_2 = self.mm2(x1)
        x1_3 = self.mm3(x1)
        x1_4 = self.mm4(x1)
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1 = self.decoder(x1)

        out = x + x1
        out = torch.relu(out)

        mask = self.trans_guide(torch.cat([x, out], dim=1))
        return out, mask