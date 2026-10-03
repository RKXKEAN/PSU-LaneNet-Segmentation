import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# Mapping from YOLO class IDs to Mask class IDs
# YOLO classes: 0: Line_L, 1: Line_C, 2: Line_R, 3: Lane, 4: Sideway
# Target Mask classes: 0: Background, 1: Line_L, 2: Line_C, 3: Line_R, 4: Lane, 5: Sideway
YOLO_TO_MASK_CLASS = {
    0: 1, # Line_L
    1: 2, # Line_C
    2: 3, # Line_R
    3: 4, # Lane
    4: 5  # Sideway
}

# Drawing order to ensure thin line markings are drawn on top of the lane area
# Sideway -> Lane -> Lines
DRAW_ORDER = [
    (4, 5), # Sideway
    (3, 4), # Lane
    (0, 1), # Line_L
    (1, 2), # Line_C
    (2, 3)  # Line_R
]

# Color map for visualization (BGR format for OpenCV)
CLASS_COLORS_BGR = {
    0: (0, 0, 0),       # Background: Black
    1: (0, 0, 255),     # Line_L: Red
    2: (0, 255, 255),   # Line_C: Yellow
    3: (0, 140, 255),   # Line_R: Orange
    4: (0, 255, 0),     # Lane: Green
    5: (255, 0, 255)    # Sideway: Magenta
}

CLASS_NAMES = {
    0: "Background",
    1: "Line_L",
    2: "Line_C",
    3: "Line_R",
    4: "Lane",
    5: "Sideway"
}

def rasterize_polygons(lbl_path, target_h, target_w):
    """
    Renders polygon annotations from YOLO format text file into a 2D integer mask (H, W).
    """
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    if not os.path.exists(lbl_path):
        return mask

    polys_by_yolo_class = {0: [], 1: [], 2: [], 3: [], 4: []}
    with open(lbl_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] = coords[:, 0] * target_w
            coords[:, 1] = coords[:, 1] * target_h
            pts = np.round(coords).astype(np.int32)
            if cls_id in polys_by_yolo_class:
                polys_by_yolo_class[cls_id].append(pts)

    # Fill polygons in ordered sequence
    for yolo_cls, mask_cls in DRAW_ORDER:
        for pts in polys_by_yolo_class[yolo_cls]:
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], mask_cls)

    return mask

def swap_horizontal_lanes(mask):
    """
    When horizontally flipping the image/mask, Line_L (class 1) and Line_R (class 3)
    must swap their semantic class identities.
    """
    swapped = mask.copy()
    line_l_idx = mask == 1
    line_r_idx = mask == 3
    swapped[line_l_idx] = 3
    swapped[line_r_idx] = 1
    return swapped

class PSULaneDataset(Dataset):
    """
    Dataset for PSU-Reservoir Lane Segmentation.
    """
    def __init__(self, image_paths, labels_dir, target_size=(384, 640), is_train=True):
        """
        Args:
            image_paths (list): List of paths to .jpg images.
            labels_dir (str): Directory containing corresponding .txt label files.
            target_size (tuple): (height, width) for model input.
            is_train (bool): Whether to apply data augmentation.
        """
        self.image_paths = sorted(image_paths)
        self.labels_dir = labels_dir
        self.target_h, self.target_w = target_size
        self.is_train = is_train

        # Normalization constants (ImageNet standards)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(self.labels_dir, f"{stem}.txt")

        # Load RGB image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Rasterize mask to original image size, then resize to target size with NEAREST
        h_orig, w_orig = img_rgb.shape[:2]
        mask_orig = rasterize_polygons(lbl_path, h_orig, w_orig)

        # Resize image (bilinear) and mask (nearest)
        img_resized = cv2.resize(img_rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_orig, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)

        # Augmentation during training
        if self.is_train:
            # 1. Random Horizontal Flip (p=0.5)
            if random.random() > 0.5:
                img_resized = cv2.flip(img_resized, 1)
                mask_resized = cv2.flip(mask_resized, 1)
                mask_resized = swap_horizontal_lanes(mask_resized)

            # 2. Photometric jitter
            if random.random() > 0.5:
                # Random brightness & contrast
                alpha = random.uniform(0.85, 1.15) # contrast
                beta = random.uniform(-20, 20)      # brightness
                img_resized = np.clip(alpha * img_resized + beta, 0, 255).astype(np.uint8)

            # 3. Random Gaussian Blur
            if random.random() > 0.7:
                ksize = random.choice([3, 5])
                img_resized = cv2.GaussianBlur(img_resized, (ksize, ksize), 0)

        # Normalize to [0, 1] then standardize
        img_normalized = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        # Transpose from (H, W, C) to (C, H, W)
        tensor_img = torch.from_numpy(img_normalized).permute(2, 0, 1).contiguous().float()
        tensor_mask = torch.from_numpy(mask_resized).contiguous().long()

        return tensor_img, tensor_mask, img_name

def mask_to_color(mask_2d):
    """
    Converts 2D integer mask (H, W) into an RGB color visualization (H, W, 3).
    """
    h, w = mask_2d.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, (b, g, r) in CLASS_COLORS_BGR.items():
        color_mask[mask_2d == cls_id] = [r, g, b]
    return color_mask
