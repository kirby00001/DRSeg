import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def mauc_coef(y_true, y_pred):
    C = y_true.shape[-3]
    H = y_true.shape[-2]
    W = y_true.shape[-1]
    y_true = y_true.squeeze().int()
    y_pred = y_pred.squeeze()
    auc = []
    for i in range(4):
        label = y_true[i].reshape(H * W)
        pred = y_pred[i].reshape(H * W)
        if label.max() != 0:
            auc.append(roc_auc_score(y_true=label, y_score=pred))
    return np.mean(auc)


def dice_coef(y_true, y_pred, threshold=0.5, dim=(2, 3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > threshold).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, threshold=0.5, dim=(2, 3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > threshold).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


if __name__ == "__main__":
    y_true = torch.randint(low=0, high=2, size=(1, 4, 3, 6))
    y_pred = torch.rand(size=(1, 4, 3, 6))
    print(mauc_coef(y_true=y_true, y_pred=y_pred))
