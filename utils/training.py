import gc
import numpy as np
from tqdm import tqdm

import torch
from utils.transform import get_transform
from datasets.IDRiD import get_train_dataloader_IDRiD, get_valid_dataloader_IDRiD
from models.unetplusplus import get_model_unetplusplus
from utils.loss import get_loss_CE
from torch.optim import Adam
from utils.metrics import mauc_coef, dice_coef, iou_coef


def train_one_epoch(model, optimizer, loss_fn, dataloader, device):
    model.to(device)
    model.train()

    dataset_size = 0
    epoch_loss = 0.0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
    for _, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.shape[0]

        y_pred = model(images)
        loss = loss_fn(y_pred, masks)

        # back propgation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate running loss and epoch_loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0

        # display info
        pbar.set_postfix(
            train_loss=f"{epoch_loss:0.4f}",
            gpu_memory=f"{mem:0.2f} GB",
        )

    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    model.to(device)

    dataset_size = 0
    epoch_loss = 0.0
    running_loss = 0.0

    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid")
    for _, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        batch_size = images.shape[0]

        y_pred = model(images)
        loss = loss_fn(y_pred, masks)

        # calculate running loss and epoch_loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        mauc = mauc_coef(y_true=masks, y_pred=y_pred)
        dice = dice_coef(y_true=masks, y_pred=y_pred).cpu().detach().numpy()
        iou = iou_coef(y_true=masks, y_pred=y_pred).cpu().detach().numpy()
        val_scores.append([mauc, dice, iou])

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            valid_loss=f"{epoch_loss:0.4f}",
            gpu_memory=f"{mem:0.2f} GB",
        )

    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores


if __name__ == "__main__":
    model = get_model_unetplusplus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
    )

    train_dataloader = get_train_dataloader_IDRiD(transform=get_transform(resize=True))
    valid_dataloader = get_valid_dataloader_IDRiD(transform=get_transform(resize=True))

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = get_loss_CE()

    train_one_epoch(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dataloader=train_dataloader,
        device="cuda",
    )

    loss, val_socres = valid_one_epoch(
        model=model,
        dataloader=valid_dataloader,
        loss_fn=loss_fn,
        device="cuda",
    )
    print(val_socres)
