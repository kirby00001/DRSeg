import torch
import gc
from tqdm import tqdm
from models.unetplusplus import get_unetpp
from utils.transform import get_transform
from datasets.IDRiD import get_dataloader_IDRiD
from torchinfo import summary
from utils.loss import get_loss_CE
from torch.optim import SGD, Adam


def training(model, optimizer, loss_fn, dataloader, device):
    model.to(device)
    model.train()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)
        y_pred = model(images)
        loss = loss_fn(y_pred, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(train_loss=f"{loss:0.4f}")
    # torch.cuda.empty_cache()
    gc.collect()
    return


if __name__ == "__main__":
    model = get_unetpp(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
    )
    dataloader = get_dataloader_IDRiD(transform=get_transform(resize=True))
    # image,masks=
    # summary(model, input_size=(1, 3, 480, 720), device="cpu")
    optimizer = Adam(model.parameters(), lr=1e-3)
    training(
        model=model,
        optimizer=optimizer,
        loss_fn=get_loss_CE(),
        dataloader=dataloader,
        device="cpu",
    )
