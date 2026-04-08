"""Segmentation model
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # --- SYMMETRIC DECODER ---
        
        # Block 5: Upsample 7x7 -> 14x14
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            # Input channels = 512 (upsampled) + 512 (skip connection) = 1024
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Block 4: Upsample 14x14 -> 28x28
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Block 3: Upsample 28x28 -> 56x56
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Block 2: Upsample 56x56 -> 112x112
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Block 1: Upsample 112x112 -> 224x224
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final 1x1 Convolution to map to the 3 classes (Foreground, Background, Unclassified)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # 1. Running the encoder and grabbing the skip connection features
        bottleneck, features = self.encoder(x, return_features=True)

        # 2. Decoder Block 5
        x = self.up5(bottleneck)
        x = torch.cat([x, features['block5']], dim=1) # Concatenate along channel dimension
        x = self.dec5(x)

        # 3. Decoder Block 4    
        x = self.up4(x)
        x = torch.cat([x, features['block4']], dim=1)
        x = self.dec4(x)

        # 4. Decoder Block 3
        x = self.up3(x)
        x = torch.cat([x, features['block3']], dim=1)
        x = self.dec3(x)

        # 5. Decoder Block 2
        x = self.up2(x)
        x = torch.cat([x, features['block2']], dim=1)
        x = self.dec2(x)

        # 6. Decoder Block 1
        x = self.up1(x)
        x = torch.cat([x, features['block1']], dim=1)
        x = self.dec1(x)

        # 7. Final Output (Raw Logits)
        logits = self.final_conv(x)
        
        return logits
