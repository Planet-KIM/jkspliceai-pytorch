
import torch
import torch.nn as nn
from einops import rearrange
from ..layers.blocks import ResidualBlock, SEBlock, SelfAttnBlock

class SpliceAI_10k_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, padding='same')

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1)
        )
        self.res_conv2 = nn.Conv1d(32, 32, 1, padding='same')

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4)
        )
        self.res_conv3 = nn.Conv1d(32, 32, 1, padding='same')

        self.block3 = nn.Sequential(
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10)
        )

        self.self_attn_block3 = SelfAttnBlock(
            dim=32, nhead=4, dim_feedforward=128, dropout=0.1
        )

        self.res_conv4 = nn.Conv1d(32, 32, 1, padding='same')

        self.block4 = nn.Sequential(
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25)
        )
        self.conv_last = nn.Conv1d(32, 3, 1, padding='same')

        self.crop_offset = 40 + 160 + 800 + 4000

    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)
        x = self.block1(x)
        detour = detour + self.res_conv2(x)
        x = self.block2(x)
        detour = detour + self.res_conv3(x)
        x = self.block3(x)
        x = self.self_attn_block3(x)
        detour = detour + self.res_conv4(x)
        x = self.block4(x) + detour
        x = self.conv_last(x)
        valid_length = x.shape[-1] - 2 * self.crop_offset
        x = x[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(x, 'b c l -> b l c')

class SpliceAI_10k_Transformer_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, padding='same')

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1)
        )
        self.se1 = SEBlock(32, reduction=16)
        self.res_conv2 = nn.Conv1d(32, 32, 1, padding='same')

        self.block2 = nn.Sequential(
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4),
            ResidualBlock(32, 32, 11, 4)
        )
        self.se2 = SEBlock(32, reduction=16)
        self.res_conv3 = nn.Conv1d(32, 32, 1, padding='same')

        self.block3 = nn.Sequential(
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10),
            ResidualBlock(32, 32, 21, 10)
        )
        self.self_attn_block3 = SelfAttnBlock(32, 4, 128, 0.1)
        self.se3 = SEBlock(32, reduction=16)
        self.res_conv4 = nn.Conv1d(32, 32, 1, padding='same')

        self.block4 = nn.Sequential(
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25)
        )
        self.se4 = SEBlock(32, reduction=16)

        self.conv_last = nn.Conv1d(32, 3, 1, padding='same')
        self.crop_offset = 40 + 160 + 800 + 4000

    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)
        x = self.block1(x)
        x = self.se1(x)
        detour = detour + self.res_conv2(x)
        x = self.block2(x)
        x = self.se2(x)
        detour = detour + self.res_conv3(x)
        x = self.block3(x)
        x = self.self_attn_block3(x)
        x = self.se3(x)
        detour = detour + self.res_conv4(x)
        x = self.block4(x) + detour
        x = self.se4(x)
        x = self.conv_last(x)
        valid_length = x.shape[-1] - 2 * self.crop_offset
        x = x[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(x, 'b c l -> b l c')
