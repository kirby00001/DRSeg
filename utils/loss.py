from torch.nn import CrossEntropyLoss


def get_loss_CE(weight=None):
    return CrossEntropyLoss(
        reduction="mean",
    )


if __name__ == "__main__":
    loss = get_loss_CE()
    print(loss)
