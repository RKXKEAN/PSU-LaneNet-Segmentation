import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import PSULaneDataset, CLASS_NAMES
from models.custom_lane_net import PSULaneNet
from utils.metrics import SegmentationMetrics
from utils.losses import CompoundLaneLoss

def load_split_paths(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths

def main():
    parser = argparse.ArgumentParser(description="Evaluate PSU-LaneNet Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--split", type=str, default="data/splits/val_images.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Evaluating model on: {device}")

    # Load paths
    val_paths = load_split_paths(args.split)
    labels_dir = cfg["dataset"]["labels_dir"]
    target_size = (cfg["dataset"]["input_height"], cfg["dataset"]["input_width"])

    dataset = PSULaneDataset(val_paths, labels_dir, target_size=target_size, is_train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load model
    num_classes = cfg["dataset"]["num_classes"]
    model = PSULaneNet(num_classes=num_classes).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from: {args.checkpoint} (Epoch {checkpoint.get('epoch', 'N/A')})")

    criterion = CompoundLaneLoss(num_classes=num_classes).to(device)
    tracker = SegmentationMetrics(num_classes=num_classes, class_names=list(CLASS_NAMES.values()))

    total_loss = 0.0
    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss, _, _ = criterion(logits, masks)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            tracker.update(preds, masks)

    avg_loss = total_loss / len(dataloader)
    metrics = tracker.compute()
    metrics["test_loss"] = avg_loss

    print("\n" + "=" * 75)
    print("                PSU-LaneNet Evaluation Results")
    print("=" * 75)
    print(f"  Test Loss:        {avg_loss:.4f}")
    print(f"  Pixel Accuracy:   {metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"  Mean IoU (mIoU):  {metrics['mIoU'] * 100:.2f}%")
    print(f"  Mean Dice (mF1):  {metrics['mDice'] * 100:.2f}%")
    print("-" * 75)
    print(f"{'Class':<15} | {'IoU (%)':<10} | {'Dice / F1 (%)':<15} | {'Precision (%)':<15} | {'Recall (%)':<12}")
    print("-" * 75)
    for cls_name in list(CLASS_NAMES.values()):
        iou = metrics['per_class_iou'].get(cls_name, 0.0) * 100
        dice = metrics['per_class_dice'].get(cls_name, 0.0) * 100
        prec = metrics['per_class_precision'].get(cls_name, 0.0) * 100
        rec = metrics['per_class_recall'].get(cls_name, 0.0) * 100
        print(f"{cls_name:<15} | {iou:>8.2f}% | {dice:>13.2f}% | {prec:>13.2f}% | {rec:>10.2f}%")
    print("=" * 75)

    # Save to json
    results_path = os.path.join(os.path.dirname(args.checkpoint), "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {results_path}")

    return metrics

if __name__ == "__main__":
    main()
