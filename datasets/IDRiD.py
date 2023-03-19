import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
from utils.load import load_image, path2paths, load_mask
from utils.transform import get_transform


class IDRiD(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = load_image(img_path)
        mask = load_mask(path2paths(img_path))
        if self.transform != None:
            img = self.transform(img)
            mask = self.transform(mask)
        else:
            img = torch.Tensor(img.transpose(2, 0, 1))
            mask = torch.Tensor(mask.transpose(2, 0, 1))
        return img, mask


def get_dataloader_IDRiD(batch_size=1, shuffle=True, transform=None):
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


if __name__ == "__main__":
    dataloader = get_dataloader_IDRiD(batch_size=1, transform=get_transform())
    # dataloader = get_dataloader_IDRiD(batch_size=1)
    img, mask = next(iter(dataloader))
    print("img.shape:", img.shape)
    print("mask.shape:", mask.shape)
