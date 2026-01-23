
import torch
import torch.nn as nn
from einops import rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return x + self.fn(x)

def ResidualBlock(in_channels, out_channels, kernel_size, dilation):
    return Residual(nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding='same')
    ))

def ResidualBlock2(in_channels, out_channels, kernel_size, dilation):
    return Residual(nn.Sequential(
        nn.BatchNorm1d(in_channels),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding='same')
    ))

class SelfAttnBlock(nn.Module):
    """
    Simple Self-Attention block using TransformerEncoderLayer.
    """
    def __init__(self, dim=32, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False   # (src_len, batch, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        # x: (batch, dim, length)
        # 1) to (length, batch, dim)
        x = rearrange(x, 'b c l -> l b c')
        # 2) Transformer Encoder
        x = self.transformer(x)
        # 3) back to (batch, dim, length)
        x = rearrange(x, 'l b c -> b c l')
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.shape
        y = self.avg_pool(x).view(b, c)           # (b, c)
        y = self.fc(y).view(b, c, 1)             # (b, c, 1)
        return x * y

class MultiScaleBlock(nn.Module):
    """
    Applies multiple dilations in parallel and concatenates results.
    """
    def __init__(self, in_ch, out_ch, dilations=(1,4,10,25), kernel_size=3):
        super().__init__()
        self.branches = nn.ModuleList([])
        n_branch = len(dilations)
        ch_per_branch = out_ch // n_branch

        for d in dilations:
            conv = nn.Conv1d(
                in_ch, ch_per_branch,
                kernel_size=kernel_size,
                dilation=d,
                padding='same'
            )
            self.branches.append(conv)

        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        outs = []
        for conv in self.branches:
            outs.append(conv(x))
        x_cat = torch.cat(outs, dim=1)  # (b, out_ch, length)
        x_cat = self.bn(x_cat)
        x_cat = self.act(x_cat)
        return x_cat
