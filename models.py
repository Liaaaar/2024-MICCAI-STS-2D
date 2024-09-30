import segmentation_models_pytorch as smp


def load_model(
    name,
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=1,
    classes=52,
    activation="sigmoid",
):
    map = {
        "deeplabv3p": smp.DeepLabV3Plus,
        "unetpp": smp.UnetPlusPlus,
        "unet": smp.Unet,
        "manet": smp.MAnet,
    }
    return map[name](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )


# model = get_model(
#     name="deeplabv3p",
#     encoder_name="resnet50",
#     encoder_weights="imagenet",
#     in_channels=1,
#     classes=52,
# )
