import torchvision.transforms as transforms


def get_train_transform(resize=True):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(size=(720, 1440), antialias=False),
            transforms.Resize(size=(480, 720), antialias=False), # type: ignore
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )
    
def get_valid_transform(resize=True):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(size=(720, 1440), antialias=False),
            transforms.Resize(size=(480, 720), antialias=False), # type: ignore
        ]
    )

