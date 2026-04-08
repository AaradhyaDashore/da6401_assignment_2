"""Inference and evaluation
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from models.multitask import MultiTaskPerceptionModel

def load_image(image_path):
    """Loads and preprocesses an image for the model."""
    # Loading using matplotlib
    image_np = plt.imread(image_path)
    
    # Handles PNG alpha channels or grayscale
    if len(image_np.shape) == 2:
        image_np = np.stack((image_np,)*3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]
        
    # Scales float images (0-1) to uint8 (0-255) for Albumentations if necessary
    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
        image_np = (image_np * 255).astype(np.uint8)

    orig_image = image_np.copy()

    # Applying the exact same transformation used in training
    transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    augmented = transform(image=image_np)
    tensor_img = torch.from_numpy(augmented['image']).permute(2, 0, 1).float().unsqueeze(0)
    
    return orig_image, tensor_img

def visualize_predictions(orig_image, tensor_img, model, device):
    """Runs inference and plots the original image with bounding box and segmentation mask."""
    model.eval()
    with torch.no_grad():
        tensor_img = tensor_img.to(device)
        outputs = model(tensor_img)
        
        # 1. Classification
        pred_class_idx = torch.argmax(outputs['classification'], dim=1).item()
        
        # 2. Localization (Bounding Box)
        # Bounding box is predicted in 224x224 space: [xc, yc, w, h]
        bbox = outputs['localization'][0].cpu().numpy()
        xc, yc, w, h = bbox
        
        # Converting to top-left corner for Matplotlib
        x_min = xc - (w / 2)
        y_min = yc - (h / 2)
        
        # 3. Segmentation
        # Argmax over the 3 classes (Foreground, Background, Unclassified)
        seg_mask = torch.argmax(outputs['segmentation'][0], dim=0).cpu().numpy()

    # Plotting:
    # We must resize the original image to 224x224 just for plotting so the bbox aligns perfectly
    plot_transform = A.Resize(height=224, width=224)
    resized_orig = plot_transform(image=orig_image)['image']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Pipeline Prediction (Predicted Class Index: {pred_class_idx})", fontsize=16)

    # Plot 1: Image + Bounding Box
    ax1.imshow(resized_orig)
    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='r', facecolor='none', label='Predicted Box')
    ax1.add_patch(rect)
    ax1.set_title("Detection")
    ax1.legend()
    ax1.axis('off')

    # Plot 2: Segmentation Trimap Overlay
    ax2.imshow(resized_orig)
    # Overlaying the mask with some transparency
    ax2.imshow(seg_mask, cmap='jet', alpha=0.5)
    ax2.set_title("Semantic Segmentation (Trimap)")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a novel pet image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the downloaded pet image.")
    args = parser.args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initializing model using local checkpoints
    print("Loading MultiTask Perception Model...")
    model = MultiTaskPerceptionModel(
        classifier_path="checkpoints/classifier.pth",
        localizer_path="checkpoints/localizer.pth",
        unet_path="checkpoints/unet.pth"
    ).to(device)
    
    # Loading Image & Predicting
    print(f"Processing image: {args.image}")
    orig_img, tensor_img = load_image(args.image)
    visualize_predictions(orig_img, tensor_img, model, device)