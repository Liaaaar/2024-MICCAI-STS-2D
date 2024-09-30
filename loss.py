import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        union = torch.sum(inputs + targets, dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - torch.mean(dice_score)
        return dice_loss


class dice_mse(nn.Module):
    def __init__(self):
        super(dice_mse, self).__init__()
        self.dice = SoftDiceLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, inputs, targets):
        return self.dice(inputs, targets) + self.mse(inputs, targets)


class bce_dice(nn.Module):
    def __init__(self):
        super(bce_dice, self).__init__()
        self.dice = SoftDiceLoss()
        self.bce = torch.nn.BCELoss()

    def forward(self, inputs, targets):
        return 0.2 * self.dice(inputs, targets) + 0.8 * self.bce(inputs, targets)


def load_loss(loss_name):
    map = {
        "dice": SoftDiceLoss,
        "mse": torch.nn.MSELoss,
        "mae": torch.nn.L1Loss,
        "dice_mse": dice_mse,
        "bce_dice": bce_dice,
    }
    return map[loss_name]()


# loss = load_loss("bce_dice")
# print(loss)
