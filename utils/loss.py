import torch
from torch.nn import CrossEntropyLoss

def get_loss_CE(weight=None):
    return CrossEntropyLoss(
        reduction="mean",
    )


if __name__ == "__main__":
    loss_fn = get_loss_CE()
    input = torch.rand(size=(1, 5, 480, 720))
    target = torch.randint(low=0, high=1, size=(1, 5, 480, 720))
    print(loss_fn(input, target))
