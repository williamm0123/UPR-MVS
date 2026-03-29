from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthRefinementHead(nn.Module):
    """
    Lightweight 2D depth refinement head.
    
    Takes coarse depth and optional features, predicts residual to refine depth.
    Similar to a shallow UNet but simpler for efficiency.
    
    Args:
        in_channels: Number of input channels (default: 1 for depth only)
        hidden_channels: Hidden layer channels (default: 16)
        out_channels: Output channels (default: 1 for depth residual)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        
        # Decoder
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection if input channels match
        if in_channels == out_channels:
            self.skip_conv = nn.Identity()
        else:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """
        Refine depth by predicting residual.
        
        Args:
            x: Coarse depth map (B, 1, H, W) or concatenated [depth, features]
            features: Optional additional features to concatenate
        
        Returns:
            Refined depth = coarse_depth + predicted_residual
        """
        # Concatenate with features if provided
        if features is not None:
            x = torch.cat([x, features], dim=1)
        
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        
        # Decoder
        x3 = F.relu(self.bn3(self.conv3(x2)))
        residual = self.conv_out(x3)
        
        # Add skip connection
        skip = self.skip_conv(x)
        
        return residual + skip
