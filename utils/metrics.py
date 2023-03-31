import torch
from sklearn.metrics import roc_auc_score


# def mauc_coef(y_true, y_pred):
#     y_true = y_true.flatten().int().cpu().detach().numpy()
#     y_pred = y_pred.flatten().cpu().detach().numpy()
#     mauc = roc_auc_score(y_true=y_true, y_score=y_pred)
#     return mauc


def dice_coef(y_true, y_pred, threshold=0.5, dim=(2, 3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, threshold=0.5, dim=(2, 3), epsilon=1e-9):
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


if __name__ == "__main__":
    y_true = torch.randint(low=0, high=2, size=(1, 5, 960, 1440))
    y_pred = torch.rand(size=(1, 5, 960, 1440))
