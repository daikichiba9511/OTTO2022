from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]


def get_base_transform(img_size: tuple[int, int]) -> dict[str, A.Compose]:
    base_transform = [
        A.augmentations.geometric.resize.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=IN_MEAN, std=IN_STD, max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ]

    return {"train": A.Compose(base_transform), "val": A.Compose(base_transform), "test": A.Compose(base_transform)}
