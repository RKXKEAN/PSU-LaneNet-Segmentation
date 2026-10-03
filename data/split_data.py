import os
import glob
import random
from pathlib import Path

def create_dataset_splits(images_base_dir="Dataset/Sampling Image",
                          labels_dir="Dataset/Labels Data/Datatset/labels/train",
                          output_dir="data/splits",
                          train_ratio=0.8,
                          seed=42):
    """
    Finds all valid image-label pairs and generates train/val split files.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # Collect all image files
    all_images = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        all_images.extend(glob.glob(os.path.join(images_base_dir, "**", ext), recursive=True))

    all_images = sorted(all_images)
    print(f"Found {len(all_images)} total images in {images_base_dir}")

    # Check matching labels
    valid_pairs = []
    for img_path in all_images:
        stem = Path(img_path).stem
        lbl_path = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.exists(lbl_path):
            valid_pairs.append(img_path)

    print(f"Verified {len(valid_pairs)} matching image-label pairs.")

    # Shuffle and split
    random.shuffle(valid_pairs)
    n_train = int(len(valid_pairs) * train_ratio)
    train_paths = valid_pairs[:n_train]
    val_paths = valid_pairs[n_train:]

    train_file = os.path.join(output_dir, "train_images.txt")
    val_file = os.path.join(output_dir, "val_images.txt")

    with open(train_file, "w", encoding="utf-8") as f:
        for p in train_paths:
            f.write(p.replace("\\", "/") + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for p in val_paths:
            f.write(p.replace("\\", "/") + "\n")

    print(f"Saved {len(train_paths)} train samples to {train_file}")
    print(f"Saved {len(val_paths)} val samples to {val_file}")
    return train_paths, val_paths

if __name__ == "__main__":
    create_dataset_splits()
