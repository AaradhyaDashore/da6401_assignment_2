"""Dataset skeleton for Oxford-IIIT Pet.
"""
import os
import xml.etree.ElementTree as ET
from typing import Tuple, Dict

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""
    def __init__(self, data_dir: str = "data", split: str = "train"):
        """
        Initialize the dataset.
        """
        self.images_dir = os.path.join(data_dir, "images")
        self.xml_dir = os.path.join(data_dir, "annotations", "xmls")
        self.trimap_dir = os.path.join(data_dir, "annotations", "trimaps")
        
        self.img_size = 224
        
        self.transform = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.samples = []
        self.class_to_idx = {}
        
        self._prepare_dataset()
        
        # Simple deterministic split (80% train, 20% val)
        split_idx = int(len(self.samples) * 0.8)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def _prepare_dataset(self):
        """Finds all valid images that have BOTH a bounding box XML and a Trimap."""
        unique_classes = set()
        valid_files = []
        
        for filename in sorted(os.listdir(self.images_dir)):
            if not filename.endswith(".jpg"):
                continue
                
            base_name = os.path.splitext(filename)[0]
            xml_path = os.path.join(self.xml_dir, f"{base_name}.xml")
            trimap_path = os.path.join(self.trimap_dir, f"{base_name}.png")
            
            if os.path.exists(xml_path) and os.path.exists(trimap_path):
                valid_files.append(base_name)
                class_name = base_name.rsplit('_', 1)[0]
                unique_classes.add(class_name)
                
        for idx, class_name in enumerate(sorted(list(unique_classes))):
            self.class_to_idx[class_name] = idx
            
        self.samples = valid_files

    def _get_bbox(self, xml_path: str, orig_w: int, orig_h: int) -> torch.Tensor:
        """Parses XML, scales to 224x224, and converts to [xc, yc, w, h]."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find('object').find('bndbox')
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        
        xmin, xmax = xmin * scale_x, xmax * scale_x
        ymin, ymax = ymin * scale_y, ymax * scale_y
        
        w = xmax - xmin
        h = ymax - ymin
        xc = xmin + w / 2.0
        yc = ymin + h / 2.0
        
        return torch.tensor([xc, yc, w, h], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        base_name = self.samples[idx]
        
        img_path = os.path.join(self.images_dir, f"{base_name}.jpg")
        xml_path = os.path.join(self.xml_dir, f"{base_name}.xml")
        trimap_path = os.path.join(self.trimap_dir, f"{base_name}.png")
        
        # 1. Loading Image using matplotlib
        image_np = plt.imread(img_path)
        # Ensure RGB (handle grayscale images)
        if len(image_np.shape) == 2:
            image_np = np.stack((image_np,)*3, axis=-1)
        elif image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
            
        orig_h, orig_w = image_np.shape[:2]
        
        # 2. Loading Trimap using matplotlib
        trimap_np = plt.imread(trimap_path)
        if trimap_np.dtype == np.float32 or trimap_np.dtype == np.float64:
            if trimap_np.max() <= 1.0:
                trimap_np = np.round(trimap_np * 255)
        trimap_np = trimap_np.astype(np.uint8)
        if len(trimap_np.shape) == 3:
            trimap_np = trimap_np[:, :, 0]
            
        # 3. Applying Albumentations (Resizes and Normalizes simultaneously)
        # Albumentations automatically uses Nearest Neighbor for masks
        augmented = self.transform(image=image_np, mask=trimap_np)
        image = augmented['image']
        trimap = augmented['mask']
        
        # Converting HWC numpy arrays to CHW PyTorch Tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # 4. Class Label & Bbox
        class_name = base_name.rsplit('_', 1)[0]
        class_label = torch.tensor(self.class_to_idx[class_name], dtype=torch.long)
        bbox = self._get_bbox(xml_path, orig_w, orig_h)
        
        # 5. Shifting Trimap (1,2,3 -> 0,1,2) for PyTorch CrossEntropy
        trimap = torch.from_numpy(trimap).long() - 1
        
        targets = {
            'classification': class_label,
            'localization': bbox,
            'segmentation': trimap
        }
        
        return image, targets