import segmentation_models_pytorch as smp


def get_unetpp(encoder_name, encoder_weights, classes):
    """_summary_

    Args:
        encoder_name (_type_): _description_
        encoder_weights (_type_): _description_
        classes (_type_): _description_
    """
    smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
    )
