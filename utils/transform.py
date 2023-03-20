import torchvision.transforms as transforms


def get_transform(resize=True):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(480, 720), antialias=False),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )
