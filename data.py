import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
import torchvision.transforms as T
import cv2

transform = A.Compose(
    [
        # A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf(
            [
                A.RandomGamma(p=1),
                A.RandomBrightnessContrast(p=1),
                A.Blur(p=1),
                A.OpticalDistortion(p=1),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.ElasticTransform(p=1),
                A.GridDistortion(p=1),
                A.MotionBlur(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
)


class Basedataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.npy")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.load(self.mask_paths[idx]).transpose(1, 2, 0)
        # transformed = transform(image=img, mask=mask)
        # img, mask = transformed["image"], transformed["mask"]
        # return totensor(img), totensor(mask)
        return img, mask


def load_data(train_rate=0.8):
    dataset = Basedataset(
        "data/STS24_2D_XRAY/Train-Labeled/Resized_Images",
        "data/STS24_2D_XRAY/Train-Labeled/Resized_Masks",
    )
    total_size = len(dataset)
    train_size = int(train_rate * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    return train_loader, val_loader


# 进行数据增强
class AugDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.totensor = T.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        if self.transform:
            transformed = transform(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]
            img, mask = self.totensor(img), self.totensor(mask)
        else:
            img, mask = self.totensor(img), self.totensor(mask)
        return img, mask


# import matplotlib.pyplot as plt
# import os
# import numpy as np

# # 创建保存叠加图像和标签的函数
# def save_overlay_sample(image, mask, index, output_dir):
#     # 转换为 numpy 数组并移除批量维度
#     image = image.squeeze().numpy()

#     # 将原图像转换为RGB格式（灰度图像扩展为3个通道）
#     image_rgb = np.stack([image, image, image], axis=-1)

#     # 定义颜色映射表（可以为每个通道分配不同的颜色）
#     colors = plt.cm.get_cmap("hsv", 52)(range(52))[:, :3]  # 生成52种不同的颜色

#     # 创建叠加图像
#     overlay_image = image_rgb.copy()
#     for i in range(52):
#         mask_channel = mask[i].squeeze().numpy()
#         color = colors[i]  # 获取当前通道的颜色
#         for c in range(3):  # 对RGB三个通道分别叠加颜色
#             overlay_image[:, :, c] = np.where(
#                 mask_channel > 0, color[c] * 255, overlay_image[:, :, c]
#             )

#     # 创建一个图形对象并显示原图像和叠加图像
#     fig, ax = plt.subplots(2, 1, figsize=(6, 6))

#     # 显示原始图像
#     ax[0].imshow(image, cmap="gray")
#     ax[0].set_title(f"Original Image {index}")
#     ax[0].axis("off")

#     # 显示叠加图像
#     ax[1].imshow(overlay_image.astype(np.uint8))
#     ax[1].set_title(f"Overlay Image {index}")
#     ax[1].axis("off")

#     # 检查输出目录是否存在，不存在则创建
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 保存图像
#     save_path = os.path.join(output_dir, f"sample_{index}.png")
#     plt.savefig(save_path)
#     plt.close()  # 关闭图形对象，防止占用内存

# # 加载数据并保存前几个叠加样本
# output_dir = "samples"  # 设置保存目录
# data = Basedataset(
#     "data/STS24_2D_XRAY/Train-Labeled/Resized_Images",
#     "data/STS24_2D_XRAY/Train-Labeled/Resized_Masks",
# )

# # 保存前几个样本的叠加图像
# for idx in range(5):
#     image, mask = data[idx]
#     save_overlay_sample(image, mask, idx, output_dir)

# img = np.random.randint(0, 256, (640, 320, 1), dtype=np.uint8)
# mask = np.random.randint(0, 256, (640, 320, 52), dtype=np.uint8)
# data_ = transform(image=img, mask=mask)
# img_, mask_ = data_["image"], data_["mask"]
# print(img_.shape, mask_.shape)
