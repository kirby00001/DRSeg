import gc
import numpy as np
from tqdm import tqdm
# from torchinfo import summary

import torch
from utils.transform import get_transform
from datasets.IDRiD import get_train_dataloader_IDRiD
from models.unetplusplus import get_model_unetplusplus
from utils.loss import get_loss_CE
from torch.optim import Adam
from utils.metrics import mauc_coef, dice_coef, iou_coef


def train_one_epoch(model, optimizer, loss_fn, dataloader, device):
    model.to(device)
    model.train()
    loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        y_pred = model(images)
        loss = loss_fn(y_pred, masks)
        # back propgation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # show loss
        pbar.set_postfix(train_loss=f"{loss:0.4f}")
    # torch.cuda.empty_cache()
    gc.collect()
    return loss

@torch.no_grad()
def valid_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    model.to(device)

    val_scores = []
    loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid")
    
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        y_pred = model(images)
        
        loss = loss_fn(y_pred, masks)
        mauc = mauc_coef(y_true=masks, y_pred=y_pred)
        dice = dice_coef(y_true=masks, y_pred=y_pred)
        iou = iou_coef(y_true=masks, y_pred=y_pred)
        
        val_scores.append([mauc, dice, iou])
        pbar.set_postfix(
            valid_loss=f"{loss:0.4f}",
            mauc_coef=f"{mauc:0.4f}",
            dice_coef=f"{dice:0.4f}",
            iou_coef=f"{iou:0.4f}",
        )
    
    val_scores = np.mean(val_scores, axis=0)
    # torch.cuda.empty_cache()
    gc.collect()

    return loss, val_scores


if __name__ == "__main__":
    model = get_model_unetplusplus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
    )

    dataloader = get_train_dataloader_IDRiD(transform=get_transform(resize=True))

    # model summary
    # summary(model, input_size=(1, 3, 480, 720), device="cpu")

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = get_loss_CE()
    
    # training(
    #     model=model,
    #     optimizer=optimizer,
    #     loss_fn=loss_fn,
    #     dataloader=dataloader,
    #     device="cpu",
    # )
    
    loss, val_socres = valid_one_epoch(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device="cpu",
    )
    print(val_socres)
