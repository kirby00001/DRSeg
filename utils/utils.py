import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img


def path2id(path):
    return int(path.split("/")[-2][0])


def load_mask(paths):
    masks = np.zeros(shape=(2848, 4288, 4))
    for path in paths:
        masks[:, :, path2id(path) - 1] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return masks / 255.0


# data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms
def path2paths(image_path):
    infos = image_path.split("/")
    train_or_test = infos[-2]
    print(train_or_test)
    i = infos[-1][:-4]
    print(i)
    return glob.glob(
        f"./data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/{train_or_test}/[1234]*/{i}_*.tif"
    )


if __name__ == "__main__":
    img_path = (
        "./data/IDRiD/A. Segmentation/1. Original Images/a. Training Set/IDRiD_01.jpg"
    )
    img = load_image(img_path)
    print(img)
    mask_paths=path2paths(img_path)
    print(mask_paths)
    mask=load_mask(mask_paths)
    print(mask.shape)