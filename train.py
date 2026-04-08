"""Training entrypoint
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

def calculate_segmentation_metrics(preds, targets):
    """Calculates Pixel Accuracy and Macro Dice Score."""
    preds = torch.argmax(preds, dim=1)
    
    # Pixel Accuracy
    correct = (preds == targets).sum().item()
    total = torch.numel(preds)
    pixel_acc = correct / total
    
    # Macro Dice Score (ignoring background class 2 if desired, but here we do all 3)
    dice_scores = []
    for cls in range(3):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        
        if union == 0:
            dice_scores.append(1.0) # Both empty
        else:
            dice_scores.append((2.0 * intersection) / union)
            
    return pixel_acc, np.mean(dice_scores)

def set_transfer_learning_mode(model, mode="full"):
    """Freezes or unfreezes backbone layers based on the selected mode."""
    encoder = model.classifier_model.encoder
    
    if mode == "frozen":
        # Strict Feature Extractor: Freeze entire backbone
        for param in encoder.parameters():
            param.requires_grad = False
    elif mode == "partial":
        # Freeze blocks 1, 2, 3. Unfreeze 4 and 5.
        for param in encoder.block1.parameters(): param.requires_grad = False
        for param in encoder.block2.parameters(): param.requires_grad = False
        for param in encoder.block3.parameters(): param.requires_grad = False
        for param in encoder.block4.parameters(): param.requires_grad = True
        for param in encoder.block5.parameters(): param.requires_grad = True
    elif mode == "full":
        # Full Fine-Tuning: Unfreeze everything
        for param in encoder.parameters():
            param.requires_grad = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mode", type=str, default="full", choices=["frozen", "partial", "full"], 
                        help="Transfer learning strategy.")
    args = parser.parse_args()

    # 1. Initializing W&B
    wandb.init(project="da6401_assignment_2", name=f"multitask_{args.mode}_finetune", config=args)

    # 2. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 3. DataLoaders
    train_dataset = OxfordIIITPetDataset(split="train")
    val_dataset = OxfordIIITPetDataset(split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4. Model Setup
    os.makedirs("checkpoints", exist_ok=True)
    model = MultiTaskPerceptionModel().to(device)
    set_transfer_learning_mode(model, args.mode)

    # 5. Loss Functions & Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_loc_mse = nn.MSELoss()
    criterion_loc_iou = IoULoss(reduction="mean")
    criterion_seg = nn.CrossEntropyLoss()
    
    # Adam Optimizer (Only passing parameters that require gradients)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # 6. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            target_cls = targets['classification'].to(device)
            target_loc = targets['localization'].to(device)
            target_seg = targets['segmentation'].to(device)

            optimizer.zero_grad()
            
            # Forward Pass
            outputs = model(images)
            
            # Calculate Losses
            loss_cls = criterion_cls(outputs['classification'], target_cls)
            # MSE is scaled down slightly to balance with IoU, as pixel coordinates are large (0-224)
            loss_loc = (criterion_loc_mse(outputs['localization'], target_loc) * 0.001) + \
                       criterion_loc_iou(outputs['localization'], target_loc)
            loss_seg = criterion_seg(outputs['segmentation'], target_seg)
            
            # Combine multi-task loss
            total_loss = loss_cls + loss_loc + loss_seg
            
            # Backprop
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
        train_loss /= len(train_loader)
        
        # 7. Validation Loop
        model.eval()
        val_loss = 0.0
        all_cls_preds, all_cls_targets = [], []
        total_iou, total_dice, total_pixel_acc = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                target_cls = targets['classification'].to(device)
                target_loc = targets['localization'].to(device)
                target_seg = targets['segmentation'].to(device)

                outputs = model(images)
                
                # Losses
                v_loss_cls = criterion_cls(outputs['classification'], target_cls)
                v_loss_loc = (criterion_loc_mse(outputs['localization'], target_loc) * 0.001) + \
                             criterion_loc_iou(outputs['localization'], target_loc)
                v_loss_seg = criterion_seg(outputs['segmentation'], target_seg)
                val_loss += (v_loss_cls + v_loss_loc + v_loss_seg).item()
                
                # Metrics Accumulation
                preds_cls = torch.argmax(outputs['classification'], dim=1)
                all_cls_preds.extend(preds_cls.cpu().numpy())
                all_cls_targets.extend(target_cls.cpu().numpy())
                
                # Batch IoU (1 - IoU Loss)
                batch_iou = 1.0 - criterion_loc_iou(outputs['localization'], target_loc).item()
                total_iou += batch_iou
                
                # Segmentation Metrics
                batch_pixel_acc, batch_dice = calculate_segmentation_metrics(outputs['segmentation'], target_seg)
                total_pixel_acc += batch_pixel_acc
                total_dice += batch_dice

        val_loss /= len(val_loader)
        
        # Finalizing Metrics
        macro_f1 = f1_score(all_cls_targets, all_cls_preds, average='macro')
        avg_iou = total_iou / len(val_loader)
        avg_pixel_acc = total_pixel_acc / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val F1: {macro_f1:.4f} | Val IoU: {avg_iou:.4f} | Val Dice: {avg_dice:.4f} | Val PixAcc: {avg_pixel_acc:.4f}")

        # Logging to Weights & Biases
        wandb.log({
            "Train Loss (Total)": train_loss,
            "Val Loss (Total)": val_loss,
            "Val Macro F1 (Classification)": macro_f1,
            "Val Mean IoU (Localization)": avg_iou,
            "Val Dice Score (Segmentation)": avg_dice,
            "Val Pixel Accuracy (Segmentation)": avg_pixel_acc
        })

        # Saving Best Models Checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("--> Saving best model checkpoints!")
            # Save individual state dicts so they align with the gdown download structure later
            torch.save(model.classifier_model.state_dict(), "checkpoints/classifier.pth")
            torch.save(model.localizer_model.state_dict(), "checkpoints/localizer.pth")
            torch.save(model.unet_model.state_dict(), "checkpoints/unet.pth")

if __name__ == "__main__":
    main()