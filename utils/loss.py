from torch.nn import CrossEntropyLoss

def get_loss_CE():
    return CrossEntropyLoss(
        reduction='sum'
    )
    
if __name__=="__main__":
    loss=get_loss_CE()
    print(loss)