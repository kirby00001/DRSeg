import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
from utils.utils import load_image, path2paths, load_mask


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
        return torch.Tensor(img), torch.Tensor(mask)


def get_dataloader_IDRiD(batch_size):
    img_paths = glob(
        "./data/IDRiD/A. Segmentation1. Original Images/a. Training Set/*.jpg"
    )
    img_paths.sort()
    dataset = IDRiD(img_paths=img_paths)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
    )


if __name__ == "__main__":
    dataloader = get_dataloader_IDRiD(1)
    print(type(dataloader))
