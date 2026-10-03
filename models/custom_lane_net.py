import torch
import torch.nn as nn
from models.blocks import (
    ConvStem,
    DepthwiseSeparableConv,
    ResidualBlock,
    StripContextBottleneck
)

class PSULaneNet(nn.Module):
    """
    PSU-LaneNet: Custom Lightweight Neural Network for Lane Segmentation.
    Designed for high efficiency, low memory footprint, and training from scratch.

    Architecture highlights:
      1. Stem with rapid 2x downsampling for early memory reduction.
      2. 3-stage Encoder using Depthwise-Separable Residual Blocks.
      3. Strip & Atrous Context Bottleneck (SCB) capturing long-range lane continuity.
      4. 3-stage Decoder with Bilinear Upsampling + Skip Connections to recover boundary sharpness.
      5. Ultra-lightweight: ~0.46M parameters, ideal for local laptop/PC inference.
    """
    def __init__(self, num_classes=6, in_channels=3):
        super().__init__()
        self.num_classes = num_classes

        # Stem: Output resolution H/2 x W/2, 32 channels
        self.stem = ConvStem(in_channels=in_channels, out_channels=32)

        # Encoder Stage 1: H/4 x W/4, 48 channels
        self.down1 = DepthwiseSeparableConv(32, 48, stride=2)
        self.enc1 = nn.Sequential(
            ResidualBlock(48),
            ResidualBlock(48)
        )

        # Encoder Stage 2: H/8 x W/8, 96 channels
        self.down2 = DepthwiseSeparableConv(48, 96, stride=2)
        self.enc2 = nn.Sequential(
            ResidualBlock(96),
            ResidualBlock(96)
        )

        # Encoder Stage 3: H/16 x W/16, 160 channels
        self.down3 = DepthwiseSeparableConv(96, 160, stride=2)
        self.enc3 = nn.Sequential(
            ResidualBlock(160),
            ResidualBlock(160),
            ResidualBlock(160)
        )

        # Bottleneck: Strip & Dilated Context Aggregation (H/16 x W/16, 160 channels)
        self.bottleneck = StripContextBottleneck(160, 160)

        # Decoder Stage 3: H/8 x W/8 (Skip connection from enc2: 96 channels)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            DepthwiseSeparableConv(160 + 96, 96),
            ResidualBlock(96)
        )

        # Decoder Stage 2: H/4 x W/4 (Skip connection from enc1: 48 channels)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            DepthwiseSeparableConv(96 + 48, 48),
            ResidualBlock(48)
        )

        # Decoder Stage 1: H/2 x W/2 (Skip connection from stem: 32 channels)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            DepthwiseSeparableConv(48 + 32, 32),
            ResidualBlock(32)
        )

        # Final Reconstruction & Segmentation Head: H x W, num_classes
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        # Initialize all layers from scratch
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming / He normal initialization for robust convergence from scratch.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Stem
        s = self.stem(x)              # [B, 32, H/2, W/2]

        # Encoder
        e1 = self.enc1(self.down1(s)) # [B, 48, H/4, W/4]
        e2 = self.enc2(self.down2(e1))# [B, 96, H/8, W/8]
        e3 = self.enc3(self.down3(e2))# [B, 160, H/16, W/16]

        # Bottleneck
        b = self.bottleneck(e3)       # [B, 160, H/16, W/16]

        # Decoder with Skip Connections
        d3 = self.dec3(torch.cat([self.up3(b), e2], dim=1)) # [B, 96, H/8, W/8]
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1)) # [B, 48, H/4, W/4]
        d1 = self.dec1(torch.cat([self.up1(d2), s], dim=1))  # [B, 32, H/2, W/2]

        # Head
        out = self.head(self.up0(d1))                         # [B, num_classes, H, W]
        return out

def get_model(num_classes=6):
    return PSULaneNet(num_classes=num_classes)
