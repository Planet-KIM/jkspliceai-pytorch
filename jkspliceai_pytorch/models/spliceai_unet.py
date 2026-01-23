
import torch
import torch.nn as nn
from einops import rearrange
from ..layers.blocks import ResidualBlock, ResidualBlock2

class SpliceAI_10k_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv_in
        self.conv_in = nn.Conv1d(4, 32, kernel_size=1, padding='same')

        # Encoder 1 (block1 - dilation=1)
        # offset = 4*(11-1)=40
        self.enc1 = nn.Sequential(
            ResidualBlock2(32, 32, 11, 1),
            ResidualBlock2(32, 32, 11, 1),
            ResidualBlock2(32, 32, 11, 1),
            ResidualBlock2(32, 32, 11, 1),
        )

        # Encoder 2 (block2 - dilation=4)
        # offset = 4 * 4*(11-1)=160
        self.enc2 = nn.Sequential(
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4),
        )

        # Center (block3 - dilation=10)
        # offset = 4 * 10*(21-1)=800
        self.center = nn.Sequential(
            ResidualBlock2(32, 32, 21, 10),
            ResidualBlock2(32, 32, 21, 10),
            ResidualBlock2(32, 32, 21, 10),
            ResidualBlock2(32, 32, 21, 10),
        )

        # Decoder 2 (block4 - dilation=25)
        # offset = 4 * 25*(41-1)=4000
        self.dec2 = nn.Sequential(
            ResidualBlock2(32, 32, 41, 25),
            ResidualBlock2(32, 32, 41, 25),
            ResidualBlock2(32, 32, 41, 25),
            ResidualBlock2(32, 32, 41, 25),
        )

        self.conv_last = nn.Conv1d(32, 3, kernel_size=1, padding='same')
        self.crop_offset = 40 + 160 + 800 + 4000

    def forward(self, x):
        x0 = self.conv_in(x)
        skip1 = self.enc1(x0)
        skip2 = self.enc2(skip1)
        c = self.center(skip2)
        d2 = c + skip2
        d2 = self.dec2(d2)
        out = d2 + skip1
        out = self.conv_last(out)

        valid_length = out.shape[-1] - 2 * self.crop_offset
        out = out[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(out, 'b c l -> b l c')

class SpliceAI_80nt_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv1d(4, 32, kernel_size=1, padding='same')

        self.enc1 = nn.Sequential(
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
        )

        self.enc2 = nn.Sequential(
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
        )

        self.center = nn.Sequential(
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
        )

        self.dec2 = nn.Sequential(
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
        )

        self.dec1 = nn.Sequential(
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
            ResidualBlock(32, 32, kernel_size=11, dilation=1),
        )

        self.conv_last = nn.Conv1d(32, 3, kernel_size=1, padding='same')
        self.crop_offset = 4 * (11 - 1)

    def forward(self, x):
        x0 = self.conv_in(x)
        skip1 = self.enc1(x0)
        skip2 = self.enc2(skip1)
        c = self.center(skip2)
        d2 = c + skip2
        d2 = self.dec2(d2)
        d1 = d2 + skip1
        d1 = self.dec1(d1)
        out = self.conv_last(d1)

        valid_length = out.shape[-1] - 2 * self.crop_offset
        out = out[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(out, 'b c l -> b l c')

class SpliceAI2:
    @staticmethod
    def from_preconfigured(model_name):
        if model_name == '80nt':
            return SpliceAI_80nt_UNet()
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))
