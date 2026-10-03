import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to address severe foreground-background class imbalance
    where thin lane lines represent less than 2% of total pixels.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none') # [B, H, W]
        pt = torch.exp(-ce_loss) # Probability of true class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss: Directly optimizes spatial mask overlap (Intersection over Union).
    """
    def __init__(self, num_classes=6, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        probs = F.softmax(logits, dim=1) # [B, C, H, W]
        b, c, h, w = probs.shape

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=c).permute(0, 3, 1, 2).float() # [B, C, H, W]

        # Flatten spatial dims
        probs_flat = probs.view(b, c, -1)
        targets_flat = targets_one_hot.view(b, c, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        cardinality = (probs_flat ** 2 + targets_flat ** 2).sum(dim=2)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth) # [B, C]
        dice_loss = 1.0 - dice_score.mean(dim=1) # Average over classes per sample

        return dice_loss.mean()

class CompoundLaneLoss(nn.Module):
    """
    Combined Focal Cross-Entropy + Soft Dice Loss for robust lane segmentation.
    """
    def __init__(self, num_classes=6, gamma=2.0, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma)
        self.dice = SoftDiceLoss(num_classes=num_classes)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        total_loss = self.focal_weight * loss_focal + self.dice_weight * loss_dice
        return total_loss, loss_focal, loss_dice
