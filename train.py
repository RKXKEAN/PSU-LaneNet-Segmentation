import os
import sys
import time
import json
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import PSULaneDataset, CLASS_NAMES
from models.custom_lane_net import PSULaneNet
from utils.losses import CompoundLaneLoss
from utils.metrics import SegmentationMetrics

def load_split_paths(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    focal_loss_sum = 0.0
    dice_loss_sum = 0.0
    num_batches = len(dataloader)

    t0 = time.time()
    for batch_idx, (images, masks, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss, l_focal, l_dice = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        focal_loss_sum += l_focal.item()
        dice_loss_sum += l_dice.item()

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch:02d} | Batch {batch_idx+1:03d}/{num_batches:03d}] "
                  f"Loss: {loss.item():.4f} (Focal: {l_focal.item():.4f}, Dice: {l_dice.item():.4f}) "
                  f"Time: {elapsed:.1f}s")

    avg_loss = total_loss / num_batches
    avg_focal = focal_loss_sum / num_batches
    avg_dice = dice_loss_sum / num_batches
    return avg_loss, avg_focal, avg_dice

@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device, metrics_tracker):
    model.eval()
    total_loss = 0.0
    focal_loss_sum = 0.0
    dice_loss_sum = 0.0
    num_batches = len(dataloader)
    metrics_tracker.reset()

    for images, masks, _ in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss, l_focal, l_dice = criterion(logits, masks)

        total_loss += loss.item()
        focal_loss_sum += l_focal.item()
        dice_loss_sum += l_dice.item()

        preds = torch.argmax(logits, dim=1)
        metrics_tracker.update(preds, masks)

    avg_loss = total_loss / num_batches
    avg_focal = focal_loss_sum / num_batches
    avg_dice = dice_loss_sum / num_batches
    results = metrics_tracker.compute()
    return avg_loss, avg_focal, avg_dice, results

def plot_history(history, save_dir):
    epochs = [h["epoch"] for h in history]

    # Plot 1: Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], 'b-o', label='Train Loss')
    plt.plot(epochs, [h["val_loss"] for h in history], 'r-s', label='Val Loss')
    plt.title('PSU-LaneNet Training & Validation Loss Convergence', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'loss_convergence.png')
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()

    # Plot 2: mIoU & Pixel Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [h["val_mIoU"] * 100 for h in history], 'g-^', label='Val mIoU (%)')
    plt.plot(epochs, [h["val_pixel_acc"] * 100 for h in history], 'm-d', label='Val Pixel Acc (%)')
    plt.title('PSU-LaneNet Validation Metrics (mIoU & Accuracy)', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Score (%)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    metrics_plot_path = os.path.join(save_dir, 'metrics_convergence.png')
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()

    print(f"Generated convergence plots saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train PSU-LaneNet From Scratch")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Overrides
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    device_str = args.device or cfg["training"]["device"]
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  PSU-LaneNet: Lane Segmentation Neural Network Training")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | Init LR: {lr}")
    print("=" * 60)

    save_dir = cfg["training"]["save_dir"]
    log_dir = cfg["training"]["log_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load splits
    train_paths = load_split_paths("data/splits/train_images.txt")
    val_paths = load_split_paths("data/splits/val_images.txt")
    labels_dir = cfg["dataset"]["labels_dir"]
    target_size = (cfg["dataset"]["input_height"], cfg["dataset"]["input_width"])

    train_dataset = PSULaneDataset(train_paths, labels_dir, target_size=target_size, is_train=True)
    val_dataset = PSULaneDataset(val_paths, labels_dir, target_size=target_size, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    print(f"Dataset: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")

    # Initialize model from scratch
    num_classes = cfg["dataset"]["num_classes"]
    model = PSULaneNet(num_classes=num_classes).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PSU-LaneNet initialized from scratch! Trainable parameters: {num_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = CompoundLaneLoss(
        num_classes=num_classes,
        gamma=cfg["training"]["focal_gamma"],
        focal_weight=cfg["training"]["focal_weight"],
        dice_weight=cfg["training"]["dice_weight"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["training"]["min_lr"])

    writer = SummaryWriter(log_dir=log_dir)
    metrics_tracker = SegmentationMetrics(num_classes=num_classes, class_names=list(CLASS_NAMES.values()))

    best_miou = 0.0
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n--- Epoch {epoch:02d}/{epochs:02d} (lr={current_lr:.6f}) ---")
        train_loss, train_focal, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_focal, val_dice, val_metrics = evaluate_epoch(model, val_loader, criterion, device, metrics_tracker)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        val_miou = val_metrics["mIoU"]
        val_acc = val_metrics["pixel_accuracy"]

        print(f"Summary Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
              f"| Val mIoU: {val_miou * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | Time: {epoch_duration:.1f}s")

        # TensorBoard logging
        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/train_focal", train_focal, epoch)
        writer.add_scalar("Loss/train_dice", train_dice, epoch)
        writer.add_scalar("Loss/val_total", val_loss, epoch)
        writer.add_scalar("Loss/val_focal", val_focal, epoch)
        writer.add_scalar("Loss/val_dice", val_dice, epoch)
        writer.add_scalar("Metrics/mIoU", val_miou, epoch)
        writer.add_scalar("Metrics/mDice", val_metrics["mDice"], epoch)
        writer.add_scalar("Metrics/PixelAccuracy", val_acc, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        for cls_name, cls_iou in val_metrics["per_class_iou"].items():
            if not np.isnan(cls_iou):
                writer.add_scalar(f"PerClass_IoU/{cls_name}", cls_iou, epoch)

        # Record history
        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_focal": train_focal,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_focal": val_focal,
            "val_dice": val_dice,
            "val_mIoU": val_miou,
            "val_mDice": val_metrics["mDice"],
            "val_pixel_acc": val_acc,
            "per_class_iou": val_metrics["per_class_iou"]
        }
        history.append(record)

        # Save checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": cfg
        }
        torch.save(checkpoint_data, os.path.join(save_dir, "latest_model.pth"))

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(checkpoint_data, os.path.join(save_dir, "best_model.pth"))
            print(f"  >>> Best model updated! mIoU: {best_miou * 100:.2f}%")

    total_training_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training Complete! Total time: {total_training_time / 60:.2f} minutes")
    print(f"Best Validation mIoU: {best_miou * 100:.2f}%")
    print(f"Best model saved to: {os.path.join(save_dir, 'best_model.pth')}")
    print("=" * 60)

    # Save training history JSON
    history_file = os.path.join(save_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot convergence figures
    plot_history(history, save_dir)
    writer.close()

if __name__ == "__main__":
    main()
