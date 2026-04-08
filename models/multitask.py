"""Unified multi-task model
"""

import torch
import torch.nn as nn
import os

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, classifier_path: str = "classifier.pth", localizer_path: str = "localizer.pth", unet_path: str = "unet.pth"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()
        try:
            import gdown
            gdown.download(id="1hKeNEAws--sD2c3Veb-AinfvnC8HjUmk", output=classifier_path, quiet=False)
            gdown.download(id="1oPPtA4kPKm_kpWLYc35yNBsiWb_vqhrb", output=localizer_path, quiet=False)
            gdown.download(id="1dY7vJmz5g7mZVqVrKz6V0LFaqbL_Rfgg", output=unet_path, quiet=False)
        except Exception as e:
            print(f"Skipping gdown download during local dev. Update IDs before submission. Error: {e}")

        # Instantiating the Sub-Models
        self.classifier_model = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer_model = VGG11Localizer(in_channels=in_channels, dropout_p=0.0) # Dropout off for inference backbone
        self.unet_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Pointing the encoders of the localizer and segmenter to the classifier's encoder.
        # This ensures there is only ONE backbone in memory, shared across all tasks.
        self.localizer_model.encoder = self.classifier_model.encoder
        self.unet_model.encoder = self.classifier_model.encoder

        # Load Pre-trained Weights if available
        if os.path.exists(classifier_path):
            self.classifier_model.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        if os.path.exists(localizer_path):
            self.localizer_model.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        if os.path.exists(unet_path):
            self.unet_model.load_state_dict(torch.load(unet_path, map_location="cpu"))

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # 1. Running the SHARED Encoder exactly once
        bottleneck, features = self.classifier_model.encoder(x, return_features=True)

        # 2. Classification Head
        c = self.classifier_model.avgpool(bottleneck)
        c = torch.flatten(c, 1)
        class_out = self.classifier_model.classifier(c)

        # 3. Localization Head
        l = self.localizer_model.avgpool(bottleneck)
        l = torch.flatten(l, 1)
        loc_raw = self.localizer_model.regressor(l)
        loc_out = torch.sigmoid(loc_raw) * 224.0 # Scale to 224x224 pixel space

        # 4. Segmentation Head (U-Net Decoder)
        s = self.unet_model.up5(bottleneck)
        s = torch.cat([s, features['block5']], dim=1)
        s = self.unet_model.dec5(s)
        
        s = self.unet_model.up4(s)
        s = torch.cat([s, features['block4']], dim=1)
        s = self.unet_model.dec4(s)
        
        s = self.unet_model.up3(s)
        s = torch.cat([s, features['block3']], dim=1)
        s = self.unet_model.dec3(s)
        
        s = self.unet_model.up2(s)
        s = torch.cat([s, features['block2']], dim=1)
        s = self.unet_model.dec2(s)
        
        s = self.unet_model.up1(s)
        s = torch.cat([s, features['block1']], dim=1)
        s = self.unet_model.dec1(s)
        
        seg_out = self.unet_model.final_conv(s)

        # 5. Returning the unified dictionary
        return {
            'classification': class_out,
            'localization': loc_out,
            'segmentation': seg_out
        }
