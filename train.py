"""
用30张有标签的图像进行5-fold训练
"""

import torch
import torch.optim as optim
from loss import load_loss
from models import load_model
from data import load_data, Basedataset, AugDataset, transform
import random
from utils import EMA
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 训练函数
def train_one_epoch(model, dataloader, criterion, optimizer, device, ema=None):
    model.train()
    running_loss = 0.0
    dice_scores = []

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # 更新EMA
        if ema is not None:
            ema.update()

        # 计算Dice分数
        outputs = (outputs > 0.5).float()
        intersection = torch.sum(outputs * masks, dim=(2, 3))
        union = torch.sum(outputs + masks, dim=(2, 3))
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice_score.mean().item())

    epoch_loss = running_loss / len(dataloader.dataset)
    mean_dice = sum(dice_scores) / len(dice_scores)
    return epoch_loss, mean_dice


# 评估函数
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, ema=None):
    if ema is not None:
        ema.apply_shadow()

    model.eval()
    running_loss = 0.0
    dice_scores = []

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)

        # 计算Dice分数
        outputs = (outputs > 0.5).float()
        intersection = torch.sum(outputs * masks, dim=(2, 3))
        union = torch.sum(outputs + masks, dim=(2, 3))
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice_score.mean().item())

    epoch_loss = running_loss / len(dataloader.dataset)
    mean_dice = sum(dice_scores) / len(dice_scores)

    if ema is not None:
        ema.restore()

    return epoch_loss, mean_dice


# 训练和评估的主函数
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    model_save_path,
    fold_idx,
    ema_decay=0.99,
):
    best_dice = 0.0
    ema = EMA(model, decay=ema_decay)
    # ema = None

    for epoch in range(num_epochs):
        print(f"Fold {fold_idx} Epoch {epoch+1}/{num_epochs}")

        # 训练一个epoch
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, ema
        )
        print(f"Training Loss: {train_loss:.4f} | Training Dice: {train_dice:.4f}")

        # 评估
        val_loss, val_dice = evaluate(model, val_loader, criterion, device, ema)
        print(f"Validation Loss: {val_loss:.4f} | Validation Dice: {val_dice:.4f}")

        # 显示当前学习率
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current Learning Rate: {current_lr:.6f}")

        # 学习率调度
        scheduler.step()

        # 保存最好的模型
        if val_dice > best_dice:
            best_dice = val_dice
            if ema is not None:
                ema.apply_shadow()
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, f"fold{fold_idx}.pth"),
            )
            if ema is not None:
                ema.restore()
            print(f"Best model saved with Dice score: {best_dice:.4f}")


# 超参数设置
batch_size = 4
learning_rate = 1e-3
num_epochs = 1000
fold_num = 5
model_save_path = "0.0dice1.0bce"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Basedataset(
    "data/STS24_2D_XRAY/Train-Labeled/Resized_Images",
    "data/STS24_2D_XRAY/Train-Labeled/Resized_Masks",
)
skf = KFold(n_splits=fold_num, shuffle=True)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dataset)):
    print(f"Training fold {fold_idx} ......")
    train_dataset = AugDataset(Subset(dataset, train_idx), transform=transform)
    val_dataset = AugDataset(Subset(dataset, val_idx), transform=None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )
    # 模型、损失函数、优化器
    model = (
        load_model(
            name="deeplabv3p",
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=52,
            activation="sigmoid",
        )
        .to(device)
        .eval()
    )

    criterion = load_loss("bce_dice")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 余弦学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 开始训练
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        device,
        model_save_path,
        fold_idx,
    )
