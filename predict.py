import os
import sys
import argparse
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import PSULaneDataset, CLASS_NAMES, CLASS_COLORS_BGR, mask_to_color, rasterize_polygons
from models.custom_lane_net import PSULaneNet

def create_visual_comparison(original_img_rgb, gt_mask, pred_mask, output_path, img_name="Sample"):
    """
    Creates a 4-panel comprehensive visualization:
      1. Original Input Image
      2. Ground Truth Mask Overlay
      3. Predicted Mask (Color-coded)
      4. Prediction Alpha-blended on Original Image
    """
    h, w = original_img_rgb.shape[:2]

    # Convert masks to RGB color
    gt_color = mask_to_color(gt_mask)
    pred_color = mask_to_color(pred_mask)

    # Resize color masks to match original image if needed
    if gt_color.shape[:2] != (h, w):
        gt_color = cv2.resize(gt_color, (w, h), interpolation=cv2.INTER_NEAREST)
    if pred_color.shape[:2] != (h, w):
        pred_color = cv2.resize(pred_color, (w, h), interpolation=cv2.INTER_NEAREST)

    # Alpha blended overlays
    # Non-background pixels only
    fg_mask_gt = (gt_color.sum(axis=2) > 0)[:, :, np.newaxis]
    gt_overlay = original_img_rgb.copy()
    gt_overlay = np.where(fg_mask_gt, cv2.addWeighted(original_img_rgb, 0.4, gt_color, 0.6, 0), gt_overlay)

    fg_mask_pred = (pred_color.sum(axis=2) > 0)[:, :, np.newaxis]
    pred_overlay = original_img_rgb.copy()
    pred_overlay = np.where(fg_mask_pred, cv2.addWeighted(original_img_rgb, 0.4, pred_color, 0.6, 0), pred_overlay)

    # Create 2x2 grid figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    # Panel 1: Original
    axes[0, 0].imshow(original_img_rgb)
    axes[0, 0].set_title(f"(A) Input Road Image: {img_name}", fontsize=13, fontweight='bold', pad=8)
    axes[0, 0].axis('off')

    # Panel 2: Ground Truth Overlay
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title("(B) Ground Truth Polygon Mask Overlay", fontsize=13, fontweight='bold', pad=8)
    axes[0, 1].axis('off')

    # Panel 3: Predicted Mask
    axes[1, 0].imshow(pred_color)
    axes[1, 0].set_title("(C) PSU-LaneNet Semantic Prediction", fontsize=13, fontweight='bold', pad=8)
    axes[1, 0].axis('off')

    # Panel 4: Prediction Overlay
    axes[1, 1].imshow(pred_overlay)
    axes[1, 1].set_title("(D) Lane Inference Overlay (Predicted)", fontsize=13, fontweight='bold', pad=8)
    axes[1, 1].axis('off')

    # Legend at bottom
    legend_elements = [
        plt.Line2D([0], [0], color=np.array(CLASS_COLORS_BGR[1][::-1])/255.0, lw=5, label='Line_L'),
        plt.Line2D([0], [0], color=np.array(CLASS_COLORS_BGR[2][::-1])/255.0, lw=5, label='Line_C'),
        plt.Line2D([0], [0], color=np.array(CLASS_COLORS_BGR[3][::-1])/255.0, lw=5, label='Line_R'),
        plt.Line2D([0], [0], color=np.array(CLASS_COLORS_BGR[4][::-1])/255.0, lw=5, label='Lane (Drivable)'),
        plt.Line2D([0], [0], color=np.array(CLASS_COLORS_BGR[5][::-1])/255.0, lw=5, label='Sideway'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved visual snapshot to: {output_path}")

def run_inference_samples(num_samples=5, checkpoint_path="checkpoints/best_model.pth", config_path="configs/default_config.yaml", output_dir="snapshots"):
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    num_classes = cfg["dataset"]["num_classes"]
    model = PSULaneNet(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load validation sample paths
    with open("data/splits/val_images.txt", "r", encoding="utf-8") as f:
        val_paths = [line.strip() for line in f if line.strip()]

    target_h = cfg["dataset"]["input_height"]
    target_w = cfg["dataset"]["input_width"]
    labels_dir = cfg["dataset"]["labels_dir"]

    # Pick samples evenly distributed across val set
    step = max(1, len(val_paths) // num_samples)
    selected_samples = val_paths[::step][:num_samples]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for idx, img_path in enumerate(selected_samples):
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(labels_dir, f"{stem}.txt")

        # Load original image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        # Ground truth mask
        gt_mask = rasterize_polygons(lbl_path, h_orig, w_orig)

        # Preprocess for model
        img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        input_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize prediction back to original dimensions for high-res overlay
        pred_highres = cv2.resize(pred, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Generate snapshot
        save_path = os.path.join(output_dir, f"inference_snapshot_{idx+1}_{stem}.png")
        create_visual_comparison(img_rgb, gt_mask, pred_highres, save_path, img_name=img_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Snapshots for PSU-LaneNet")
    parser.add_argument("--samples", type=int, default=5, help="Number of validation images to infer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--output", type=str, default="snapshots")
    args = parser.parse_args()

    run_inference_samples(num_samples=args.samples, checkpoint_path=args.checkpoint, config_path=args.config, output_dir=args.output)
