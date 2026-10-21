import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def build_transforms(cfg):
    train_tf = A.Compose(
        [
            A.Resize(cfg.img_size, cfg.img_size),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.35,
                scale_limit=0.35,
                rotate_limit=35,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
    test_tf = A.Compose(
        [
            A.Resize(cfg.img_size, cfg.img_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )
    return train_tf, test_tf
