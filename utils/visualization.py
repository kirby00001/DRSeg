import numpy as np
import matplotlib.pyplot as plt 

def image_show(image, masks):
    plt.imshow(image)
    cmaps = ["Greens_r", "Purples_r", "Blues_r", "Reds_r"]
    masks = np.ma.masked_where(masks == 0, masks)
    print(masks)
    for i in range(1,5):
        plt.imshow(masks[:, :, i], alpha=1, cmap=cmaps[i-1])
