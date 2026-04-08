"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, the value recieved is: {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor for shape [B, C, H, W].

        Returns:
            Output tensor.
        """
        # If the model is in evaluation mode (model.eval()) or p is 0, doing nothing.
        if not self.training or self.p == 0.0:
            return x

        # 1. Createing a binary mask. 
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) > self.p).float()

        # 2 & 3. Multiplying by the mask to drop neurons, and dividing by keep_prob to scale.
        return (x * mask) / keep_prob
