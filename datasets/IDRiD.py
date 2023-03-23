from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader

from utils.transform import get_train_transform, get_valid_transform
from utils.load import load_image, path2paths, load_mask, mask2label


class IDRiD(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = load_image(img_path)
        mask = load_mask(path2paths(img_path))

        img = self.transform(img)
        mask = self.transform(mask)

        return img, mask


def get_train_dataloader_IDRiD(transform, batch_size=1, shuffle=True):
    img_paths = glob(
        "./data/IDRiD/A. Segmentation/1. Original Images/a. Training Set/*.jpg"
    )
    img_paths.sort()
    # print("image paths:")
    # for img_path in img_paths:
    #     print(img_path)
    dataset = IDRiD(img_paths=img_paths, transform=transform)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def get_valid_dataloader_IDRiD(transform, batch_size=1, shuffle=True):
    image_paths = glob(
        "./data/IDRiD/A. Segmentation/1. Original Images/b. Testing Set/*.jpg"
    )
    image_paths.sort()
    # print("image paths:")
    # for img_path in img_paths:
    #     print(img_path)
    dataset = IDRiD(img_paths=image_paths, transform=transform)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


if __name__ == "__main__":
    train_dataloader = get_train_dataloader_IDRiD(
        batch_size=1, transform=get_train_transform()
    )
    test_dataloader = get_valid_dataloader_IDRiD(
        batch_size=1, transform=get_valid_transform()
    )

    # dataloader = get_dataloader_IDRiD(batch_size=1)
    image, mask = next(iter(test_dataloader))
    print("img.max:", image.max())
    print("img.min:", image.min())
    print("img.shape:", image.shape)
    print("mask.max:", mask.max())
    print("mask.min", mask.min())
    print("mask.shape:", mask.shape)
