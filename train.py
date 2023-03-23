import gc
import copy
import time
import wandb
import numpy as np
from colorama import Fore, Back, Style
from collections import defaultdict


import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from utils.loss import get_loss_CE
from utils.transform import get_train_transform, get_valid_transform
from utils.training import train_one_epoch, valid_one_epoch
from datasets.IDRiD import get_train_dataloader_IDRiD, get_valid_dataloader_IDRiD  # type: ignore

from models.unetplusplus import get_model_unetplusplus

color = Fore.GREEN
reset = Style.RESET_ALL


def run_training(model, optimizer, device, num_epochs):
    # To automatically log gradients
    # wandb.watch(model, log_freq=100)

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
    train_dataloader = get_train_dataloader_IDRiD(transform=get_train_transform())
    valid_dataloader = get_valid_dataloader_IDRiD(transform=get_valid_transform())

    loss_fn = get_loss_CE()

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

        val_mauc, val_dice, val_iou = val_scores
        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        history["Valid mAUC"].append(val_mauc)
        history["Valid Dice"].append(val_dice)
        history["Valid IoU"].append(val_iou)

        # Log loss and metrics
        # wandb.log(
        #     {
        #         "Train Loss": train_loss,
        #         "Valid Loss": val_loss,
        #         "Valid mAUC": val_mauc,
        #         "Valid Dice": val_dice,
        #         "Valid IoU": val_iou,
        #         # "LR": scheduler.get_last_lr()[0],
        #     }
        # )

        print(
            f"Valid mAUC: {val_mauc:0.4f} | Valid Dice: {val_dice:0.4f} | Valid IoU: {val_iou:0.4f}"
        )

        # deep copy the model
        if val_dice > best_dice:
            print(
                f"{color}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})"
            )
            best_dice = val_dice
            best_iou = val_iou
            best_epoch = epoch
            # if run != None:
            #     run.summary["Best Dice"] = best_dice
            #     run.summary["Best IoU"] = best_iou
            #     run.summary["Best Epoch"] = best_epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            PATH = f"./checkpoints/best_epoch.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            # wandb.save(PATH)
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
    # if run != None:
    #     run.finish()
    return model, history


if __name__ == "__main__":
    model = get_model_unetplusplus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        classes=5,
    )

    optimizer = Adam(model.parameters(), lr=2e-3)
    # loss_fn = get_loss_CE()
    loss_fn = BCEWithLogitsLoss()
    device = "cuda"

    # wandb.login(key="b9b9bfc9d98eada98a991a294a1e40ad81437726")
    # anonymous = None
    # run = wandb.init(
    #     project="DR Segmentation",
    #     name=f"Dim 480x720|model U-net++",
    #     anonymous=anonymous,
    #     group="U-net++ efficientnet_b0 480x720",
    #     config={"epoch": 1},
    # )

    run_training(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
    )
