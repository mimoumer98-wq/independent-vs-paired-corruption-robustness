import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_tn3k_paths(data_dir):
    """Expects data_dir/{train,test}/{images,masks}."""
    data_dir = Path(data_dir)

    def get_pairs(split: str):
        img_dir = data_dir / split / "images"
        mask_dir = data_dir / split / "masks"

        img_paths = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        if len(img_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        images, masks = [], []
        for p in img_paths:
            m = mask_dir / p.name
            if not m.exists():
                m = mask_dir / f"{p.stem}.png"
            if m.exists():
                images.append(str(p))
                masks.append(str(m))

        if len(images) == 0:
            raise RuntimeError(f"No image/mask pairs in {img_dir} / {mask_dir}")
        return images, masks

    train_images, train_masks = get_pairs("train")
    test_images, test_masks = get_pairs("test")
    return {
        "train": {"images": train_images, "masks": train_masks},
        "test": {"images": test_images, "masks": test_masks},
    }


def load_ddti_paths(ddti_root):
    """Expects ddti_root/{images,masks}. Returns (image_paths, mask_paths)."""
    ddti_root = Path(ddti_root)
    img_dir = ddti_root / "images"
    mask_dir = ddti_root / "masks"
    images = sorted(img_dir.glob("*.png"))
    image_paths, mask_paths = [], []
    for p in images:
        m = mask_dir / p.name
        if m.exists():
            image_paths.append(str(p))
            mask_paths.append(str(m))
    return image_paths, mask_paths


def split_train_val(images, masks, val_ratio=0.2, seed=42):
    if len(images) != len(masks):
        raise ValueError("Number of images != number of masks")

    n = len(images)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_val = int(round(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    tr_images = [images[i] for i in train_idx]
    tr_masks = [masks[i] for i in train_idx]
    va_images = [images[i] for i in val_idx]
    va_masks = [masks[i] for i in val_idx]
    return (tr_images, tr_masks), (va_images, va_masks)


class TN3KDataset(Dataset):
    """Binary segmentation dataset (nodule vs. background).
    image: float32 [1,H,W]; mask: float32 [1,H,W] in {0,1}."""

    def __init__(self, images, masks, transform=None, img_size=224):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.img_size = (img_size, img_size)
        if len(self.images) != len(self.masks):
            raise ValueError("Number of images != number of masks")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            if img.ndim == 2:
                img = img.unsqueeze(0)
            if img.max() > 1.0:
                img = img / 255.0

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        if img.ndim == 2:
            img = img.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        mask = (mask > 0.5).float()
        return img, mask
