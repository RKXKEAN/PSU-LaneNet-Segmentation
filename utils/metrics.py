import numpy as np
import torch

class SegmentationMetrics:
    """
    Computes comprehensive semantic segmentation metrics using a confusion matrix:
      - Mean Intersection over Union (mIoU)
      - Per-class IoU
      - Dice Coefficient / F1-Score
      - Precision & Recall
      - Overall Pixel Accuracy
    """
    def __init__(self, num_classes=6, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        """
        Args:
            preds: numpy array or tensor [B, H, W] of predicted class labels
            targets: numpy array or tensor [B, H, W] of ground truth class labels
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = preds.flatten()
        targets = targets.flatten()

        mask = (targets >= 0) & (targets < self.num_classes)
        # Confusion matrix: rows = true, columns = pred
        indices = self.num_classes * targets[mask].astype(int) + preds[mask].astype(int)
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        Returns a dictionary containing all computed metrics.
        """
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # Pixel Accuracy
        total_pixels = cm.sum()
        pixel_acc = tp.sum() / (total_pixels + 1e-10)

        # Per-class IoU
        union = tp + fp + fn
        ious = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            if union[i] > 0:
                ious[i] = tp[i] / union[i]
            else:
                ious[i] = np.nan

        miou = np.nanmean(ious)

        # Per-class Dice (F1)
        dice_denominator = 2 * tp + fp + fn
        dices = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            if dice_denominator[i] > 0:
                dices[i] = (2 * tp[i]) / dice_denominator[i]
            else:
                dices[i] = np.nan

        mdice = np.nanmean(dices)

        # Precision & Recall
        precisions = np.zeros(self.num_classes, dtype=np.float64)
        recalls = np.zeros(self.num_classes, dtype=np.float64)
        for i in range(self.num_classes):
            denom_p = tp[i] + fp[i]
            denom_r = tp[i] + fn[i]
            precisions[i] = tp[i] / denom_p if denom_p > 0 else 0.0
            recalls[i] = tp[i] / denom_r if denom_r > 0 else 0.0

        results = {
            "pixel_accuracy": float(pixel_acc),
            "mIoU": float(miou),
            "mDice": float(mdice),
            "per_class_iou": {self.class_names[i]: float(ious[i]) for i in range(self.num_classes)},
            "per_class_dice": {self.class_names[i]: float(dices[i]) for i in range(self.num_classes)},
            "per_class_precision": {self.class_names[i]: float(precisions[i]) for i in range(self.num_classes)},
            "per_class_recall": {self.class_names[i]: float(recalls[i]) for i in range(self.num_classes)},
        }
        return results
