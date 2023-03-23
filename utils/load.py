import cv2
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    img = np.array(Image.open(path))
    return img


# data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms
def path2paths(image_path):
    infos = image_path.split("/")
    train_or_test = infos[-2]
    # print("train_or_test:", train_or_test)
    i = infos[-1][:-4]
    # print("ID:",i)
    return glob.glob(
        f"../data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/{train_or_test}/[1234]*/{i}_*.tif"
    )


def path2id(path):
    return int(path.split("/")[-2][0])


def load_mask(paths):
    masks = np.zeros(shape=(2848, 4288, 4))
    for path in paths:
        # print("path2id:",path2id(path))
        image_id = path2id(path)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(f"mask{image_id}.max", mask.max())
        masks[:, :, image_id - 1] = mask
    return masks / 76.0


def mask2label(masks):
    pad_masks = np.pad(masks, pad_width=[(0, 0), (0, 0), (1, 0)])
    label = np.argmax(pad_masks, axis=-1)
    return label


if __name__ == "__main__":
    img_path = (
        "../data/IDRiD/A. Segmentation/1. Original Images/a. Training Set/IDRiD_17.jpg"
    )
    img_path = (
        "../data/IDRiD/A. Segmentation/1. Original Images/b. Testing Set/IDRiD_81.jpg"
    )
    # load image
    img = load_image(img_path)
    print("img.max:", img.max())
    print("img.shape", img.shape)
    # plt.imshow(img)
    # from image path to mask paths
    mask_paths = path2paths(img_path)
    print("mask paths:")
    for mask_path in mask_paths:
        print(mask_path)
    # load masks
    masks = load_mask(mask_paths)
    print("mask.max", masks.max())
    print("mask.shape", masks.shape)

    # Visualization
    # cmaps = ["Greens_r", "Purples_r", "Blues_r", "Reds_r"]
    # masks = np.ma.masked_where(masks == 0, masks)
    # print(masks)
    # for i in range(4):
    #     plt.imshow(masks[:, :, i], alpha=1, cmap=cmaps[i])

    # to label
    label = mask2label(masks)
    print(label)
    print("label.shape", label.shape)
    print("label.max", label.max())
