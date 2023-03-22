import gc
import copy
import time
import torch

# import wandb
import numpy as np
from collections import defaultdict

from torch.optim import Adam

from utils.transform import get_transform
from datasets.IDRiD import get_train_dataloader_IDRiD, get_valid_dataloader_IDRiD
from utils.loss import get_loss_CE
from utils.training import train_one_epoch, valid_one_epoch

from models.unetplusplus import get_model_unetplusplus


def run_training(model, optimizer, device, num_epochs):
    # To automatically log gradients
    # wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    # Initilization
    start = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    # Load Data
    train_dataloader = get_train_dataloader_IDRiD(transform=get_transform(resize=True))
    valid_dataloader = get_valid_dataloader_IDRiD(transform=get_transform(resize=True))
    
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

        # Log the metrics
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
        # if val_dice >= best_dice:
        #     print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
        #     best_dice = val_dice
        #     best_jaccard = val_iou
        #     best_epoch = epoch
        #     run.summary["Best Dice"] = best_dice
        #     run.summary["Best Jaccard"] = best_jaccard
        #     run.summary["Best Epoch"] = best_epoch
        #     best_model_weights = copy.deepcopy(model.state_dict())
        #     PATH = f"best_epoch-{fold:02d}.bin"
        #     torch.save(model.state_dict(), PATH)
        #     # Save a model file from the current directory
        #     wandb.save(PATH)
        #     print(f"Model Saved{sr_}")

        # last_model_weights = copy.deepcopy(model.state_dict())
        # PATH = f"last_epoch-{fold:02d}.bin"
        # torch.save(model.state_dict(), PATH)

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    # print("Best Score: {:.4f}".format(best_jaccard))

    # load best model weights
    # model.load_state_dict(best_model_weights)

    return model, history


if __name__ == "__main__":
    model = get_model_unetplusplus(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
    )

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = get_loss_CE()
    device = "cuda"
    run_training(
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=5,
    )
