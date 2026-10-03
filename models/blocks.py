import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvStem(nn.Module):
    """
    Stem module: Rapid spatial downsampling (x2) with minimal channels to preserve
    memory efficiency while learning rich low-level edge features.
    """
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.stem(x)

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution:
    Decomposes standard convolution into Depthwise (spatial) + Pointwise (channel)
    reducing FLOPs and parameter count by ~80-90%.
    """
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=dilation, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class ResidualBlock(nn.Module):
    """
    Inverted / Depthwise Residual Block:
    Preserves gradient flow during training from scratch.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))

class StripContextBottleneck(nn.Module):
    """
    Strip & Atrous Context Bottleneck (SCB):
    Tailored specifically for traffic lanes. Road lanes are long, continuous vertical structures
    with perspective convergence. This block combines:
      1. Local 1x1 Conv branch
      2. Dilated 3x3 Conv branch (dilation=2) for widened context
      3. Asymmetric Strip Convolutions (1x5 horizontal + 5x1 vertical) for lane continuity
      4. Global Context Branch (Adaptive Average Pooling)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4

        # Branch 1: Local features
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True)
        )

        # Branch 2: Dilated context
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True)
        )

        # Branch 3: Strip convolutions for long lane lines
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.SiLU(inplace=True)
        )

        # Branch 4: Global scene context
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True)
        )

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(inter_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        f1 = self.b1(x)
        f2 = self.b2(x)
        f3 = self.b3(x)
        f4 = F.interpolate(self.b4(x), size=(h, w), mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([f1, f2, f3, f4], dim=1))
