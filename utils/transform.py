import torchvision.transforms as transforms

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((960,1440)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])