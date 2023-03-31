import gc
import copy
import time
import wandb
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn.functional as F

from datasets.IDRiD import get_dataloader_IDRiD
from models.unetplusplus import get_model_unetplusplus
from utils import get_transform, get_loss_CE, dice_coef, iou_coef

from colorama import Fore, Style
from collections import defaultdict

color = Fore.GREEN
reset = Style.RESET_ALL


def train_one_epoch(model, optimizer, loss_fn, dataloader, device):
    model.train()
    model.to(device)

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

        # display info
        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            train_epoch_loss=f"{epoch_loss:0.4f}",
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

        y_score = F.softmax(y_pred, dim=1)
        # mauc = mauc_coef(y_true=masks, y_pred=y_score)
        dice = dice_coef(y_true=masks, y_pred=y_score).cpu().detach().numpy()
        iou = iou_coef(y_true=masks, y_pred=y_score).cpu().detach().numpy()
        val_scores.append([dice, iou])

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        pbar.set_postfix(
            valid_epoch_loss=f"{epoch_loss:0.4f}",
            gpu_memory=f"{mem:0.2f} GB",
        )

    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss, val_scores


def run_training(model, loss_fn, optimizer, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    # Initilization
    start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_iou = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    # Load Data
    train_dataloader = get_dataloader_IDRiD(
        batch_size=1,
        transform=get_transform(mode="train"),
        shuffle=True,
        mode="train",
    )
    valid_dataloader = get_dataloader_IDRiD(
        batch_size=1,
        transform=get_transform(mode="valid"),
        shuffle=True,
        mode="valid",
    )

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            dataloader=train_dataloader,
            device=device,
        )

        val_loss, val_scores = valid_one_epoch(
            model=model,
            dataloader=valid_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        val_dice, val_iou = val_scores
        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        history["Valid Dice"].append(val_dice)
        history["Valid IoU"].append(val_iou)

        # Log loss and metrics
        wandb.log(
            {
                "Train Loss": train_loss,
                "Valid Loss": val_loss,
                "Valid Dice": val_dice,
                "Valid IoU": val_iou,
                # "LR": scheduler.get_last_lr()[0],
            }
        )

        print(
            f"Valid Dice: {val_dice:0.4f} | Valid IoU: {val_iou:0.4f}"
        )

        # deep copy the model
        if val_dice > best_dice:
            print(
                f"{color}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})"
            )
            best_dice = val_dice
            best_iou = val_iou
            best_epoch = epoch
            if run != None:
                run.summary["Best Dice"] = best_dice
                run.summary["Best IoU"] = best_iou
                run.summary["Best Epoch"] = best_epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            PATH = f"./checkpoints/best_epoch.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            wandb.save(PATH)
            print(f"Model Saved{reset}")

        last_model_weights = copy.deepcopy(model.state_dict())
        PATH = f"./checkpoints/last_epoch.bin"
        torch.save(model.state_dict(), PATH)

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Score: {:.4f}".format(best_iou))
    # load best model weights
    model.load_state_dict(best_model_weights)
    
    if run != None:
        run.finish() 
    
    return model, history


if __name__ == "__main__":
    model = get_model_unetplusplus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        classes=5,
    )

    optimizer = Adam(model.parameters(), lr=2e-3)
    loss_fn = get_loss_CE()
    device = "cuda"

    wandb.login(key="b9b9bfc9d98eada98a991a294a1e40ad81437726")
    anonymous = None
    run = wandb.init(
        project="DR Segmentation",
        name=f"Dim 960x1440|model U-net++",
        anonymous=anonymous,
        group="U-net++ efficientnet_b0 960x1440",
        config={"epoch": 10},
    )

    run_training(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
    )
