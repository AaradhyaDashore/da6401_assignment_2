"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        # Validating reduction in {"none", "mean", "sum"}
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction type: {reduction}. Expected 'none', 'mean', or 'sum'.")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # 1. Converting [x_center, y_center, w, h] to [x1, y1, x2, y2] format
        preds_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        preds_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        preds_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        preds_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        targets_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        targets_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        targets_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        targets_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # 2. Finding the coordinates of the intersection rectangle
        inter_x1 = torch.max(preds_x1, targets_x1)
        inter_y1 = torch.max(preds_y1, targets_y1)
        inter_x2 = torch.min(preds_x2, targets_x2)
        inter_y2 = torch.min(preds_y2, targets_y2)

        # 3. Calculating Intersection Area
        # clamp(min=0) ensures that if boxes don't overlap, area is 0 (not negative)
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        intersection_area = inter_w * inter_h

        # 4. Calculating Union Area
        preds_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        targets_area = target_boxes[:, 2] * target_boxes[:, 3]
        union_area = preds_area + targets_area - intersection_area

        # 5. Calculating IoU using the initialized self.eps
        iou = intersection_area / (union_area + self.eps)

        # 6. Loss = 1 - IoU
        loss = 1.0 - iou

        # 7. Applying Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss