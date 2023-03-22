import cv2
import glob
import numpy as np


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


# data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/1. Microaneurysms
def path2paths(image_path):
    infos = image_path.split("/")
    train_or_test = infos[-2]
    # print("train_or_test:", train_or_test)
    i = infos[-1][:-4]
    # print("ID:",i)
    return glob.glob(
        f"./data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/{train_or_test}/[1234]*/{i}_*.tif"
    )


def path2id(path):
    return int(path.split("/")[-2][0])


def load_mask(paths):
    masks = np.zeros(shape=(2848, 4288, 4))
    for path in paths:
        # print("path2id:",path2id(path))
        image_id = path2id(path)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        masks[:, :, image_id - 1] = mask
    return masks/76.0


if __name__ == "__main__":

    mask_path = "./data/IDRiD/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates/IDRiD_81_EX.tif"
    mask = cv2.imread(mask_path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    print(mask.max())
    
    img_path = (
        "./data/IDRiD/A. Segmentation/1. Original Images/b. Testing Set/IDRiD_81.jpg"
    )
    # load image
    img = load_image(img_path)
    print("img.max:", img.max())
    print("img.shape", img.shape)
    # from image path to mask paths
    mask_paths = path2paths(img_path)
    print("mask paths:")
    for mask_path in mask_paths:
        print(mask_path)
    # load masks
    mask = load_mask(mask_paths)
    print("mask.max", mask.max())
    print("mask.shape", mask.shape)
