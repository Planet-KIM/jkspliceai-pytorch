
import torch
import torch.nn as nn
from einops import rearrange
from ..layers.blocks import ResidualBlock, ResidualBlock2

# SpliceAI_80nt
class SpliceAI_80nt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, padding='same')
        
        self.block1 = nn.Sequential(
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            ResidualBlock(32, 32, 11, 1),
            nn.Conv1d(32, 32, 1, padding='same')
        )
        self.conv_last = nn.Conv1d(32, 3, 1, padding='same')
        
        # crop_offset = 40
        self.crop_offset = 4 * (11 - 1)
    
    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)
        x = detour + self.block1(x)
        x = self.conv_last(x)
        valid_length = x.shape[-1] - 2 * self.crop_offset
        x = x[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(x, 'b c l -> b l c')

# SpliceAI_400nt
class SpliceAI_400nt(nn.Module):
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
            ResidualBlock(32, 32, 11, 4),
            nn.Conv1d(32, 32, 1, padding='same')
        )
        self.conv_last = nn.Conv1d(32, 3, 1, padding='same')
        
        # crop_offset = 200
        self.crop_offset = 40 + 160
    
    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)
        x = self.block1(x)
        detour = detour + self.res_conv2(x)
        x = self.block2(x) + detour
        x = self.conv_last(x)
        valid_length = x.shape[-1] - 2 * self.crop_offset
        x = x[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(x, 'b c l -> b l c')

# SpliceAI_2k
class SpliceAI_2k(nn.Module):
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
            ResidualBlock(32, 32, 21, 10),
            nn.Conv1d(32, 32, 1, padding='same')
        )
        self.conv_last = nn.Conv1d(32, 3, 1, padding='same')
        
        # crop_offset = 1000
        self.crop_offset = 40 + 160 + 800
    
    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)
        x = self.block1(x)
        detour = detour + self.res_conv2(x)
        x = self.block2(x)
        detour = detour + self.res_conv3(x)
        x = self.block3(x) + detour
        x = self.conv_last(x)
        valid_length = x.shape[-1] - 2 * self.crop_offset
        x = x[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(x, 'b c l -> b l c')

# SpliceAI_10k
class SpliceAI_10k(nn.Module):
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
        self.res_conv4 = nn.Conv1d(32, 32, 1, padding='same')
        
        self.block4 = nn.Sequential(
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25),
            ResidualBlock(32, 32, 41, 25)
        )
        self.conv_last = nn.Conv1d(32, 3, 1, padding='same')
        
        # crop_offset = 5000
        self.crop_offset = 40 + 160 + 800 + 4000
    
    def forward(self, x):
        x = self.conv1(x)
        detour = self.res_conv1(x)
        x = self.block1(x)
        detour = detour + self.res_conv2(x)
        x = self.block2(x)
        detour = detour + self.res_conv3(x)
        x = self.block3(x)
        detour = detour + self.res_conv4(x)
        x = self.block4(x) + detour
        x = self.conv_last(x)
        valid_length = x.shape[-1] - 2 * self.crop_offset
        x = x[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(x, 'b c l -> b l c')

# SpliceAI_10k_drop
class SpliceAI_10k_drop(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 1, padding='same')
        self.res_conv1 = nn.Conv1d(32, 32, 1, padding='same')
        
        self.block1 = nn.Sequential(
            ResidualBlock2(32, 32, 11, 1),
            ResidualBlock2(32, 32, 11, 1),
            ResidualBlock2(32, 32, 11, 1),
            ResidualBlock2(32, 32, 11, 1)
        )
        self.res_conv2 = nn.Conv1d(32, 32, 1, padding='same')
        
        self.block2 = nn.Sequential(
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4)
        )
        self.res_conv3 = nn.Conv1d(32, 32, 1, padding='same')
        
        self.block3 = nn.Sequential(
            ResidualBlock2(32, 32, 21, 10),
            ResidualBlock2(32, 32, 21, 10),
            ResidualBlock2(32, 32, 21, 10),
            ResidualBlock2(32, 32, 21, 10)
        )
        self.res_conv4 = nn.Conv1d(32, 32, 1, padding='same')
        
        self.block4 = nn.Sequential(
            ResidualBlock2(32, 32, 41, 25),
            ResidualBlock2(32, 32, 41, 25),
            ResidualBlock2(32, 32, 41, 25),
            ResidualBlock2(32, 32, 41, 25)
        )
        self.res_conv5 = nn.Conv1d(32, 32, 1, padding='same')
        
        self.block5 = nn.Sequential(
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4),
            ResidualBlock2(32, 32, 11, 4)
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
        detour = detour + self.res_conv4(x)
        x = self.block4(x)
        detour = detour + self.res_conv5(x)
        x = self.block5(x) + detour
        x = self.conv_last(x)
        valid_length = x.shape[-1] - 2 * self.crop_offset
        x = x[..., self.crop_offset : self.crop_offset + valid_length]
        return rearrange(x, 'b c l -> b l c')

# Factory
class SpliceAI:
    @staticmethod
    def from_preconfigured(model_name):
        from .spliceai_unet import SpliceAI_10k_UNet
        from .spliceai_trans import SpliceAI_10k_Transformer, SpliceAI_10k_Transformer_SE
        
        if model_name == '80nt':
            return SpliceAI_80nt()
        elif model_name == '400nt':
            return SpliceAI_400nt()
        elif model_name == '2k':
            return SpliceAI_2k()
        elif model_name == '10k':
            return SpliceAI_10k()
        elif model_name == '10k_drop':
            return SpliceAI_10k_drop()
        elif model_name == '10k_unet':
            return SpliceAI_10k_UNet()
        elif model_name == '10k_multi':
            return SpliceAI_10k_Transformer_SE()
        elif model_name == '10k_trans':
            return SpliceAI_10k_Transformer()
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))
