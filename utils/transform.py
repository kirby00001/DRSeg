import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    return A.Compose(
        [
            A.RandomResizedCrop(960, 1440, scale=(2000, 2000)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(transpose_mask=True),
        ]
    )


def get_valid_transform():
    return A.Compose(
        [
            # transforms.RandomResizedCrop(size=(720, 1440), antialias=False),
            A.Resize(960, 1440),
            ToTensorV2(transpose_mask=True),
        ]
    )
