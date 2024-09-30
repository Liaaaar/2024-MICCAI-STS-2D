# # 用训练好的模型生成伪标签
# import os
# import cv2
# import torch
# import numpy as np
# import torchvision.transforms as T
# from skimage.morphology import opening, closing, disk
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# from models import load_model
# from utils import RemoveSmallObjects

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# # 加载模型
# model = load_model(
#     name="deeplabv3p",
#     encoder_name="resnet50",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=52,
#     activation="sigmoid",
# ).eval()

# # 读取权重
# cps_path = "test"
# models = []
# cps = os.listdir(cps_path)
# for cp in cps:
#     model.load_state_dict(torch.load(os.path.join(cps_path, cp)))
#     models.append(model.to("cuda"))

# # 路径设置
# img_path = "data/STS24_2D_XRAY/Resized-Train-Unlabeled"
# label_path = "data/STS24_2D_XRAY/Resized-Train-Unlabeled-Masks-v2"
# if not os.path.exists(label_path):
#     os.makedirs(label_path)

# imgs = os.listdir(img_path)


# # 并行处理函数，用于处理单个通道的形态学操作
# def process_channel(channel):
#     selem = disk(5)  # 结构元素
#     channel = closing(channel, selem, out=channel)  # 闭运算
#     channel = opening(channel, selem, out=channel)  # 开运算
#     return channel


# # 主处理循环
# for img in tqdm(imgs):
#     data = cv2.imread(os.path.join(img_path, img))
#     data = T.ToTensor()(data).unsqueeze(0).to("cuda")
#     data_ = data.flip(dims=[2])

#     # 聚合多个模型的预测结果
#     output = 0
#     for model in models:
#         output += model(data)
#         output += model(data_).flip(dims=[2])

#     output = (output / 2 / len(models)).detach().squeeze(0).cpu().numpy()
#     output[output >= 0.5] = 255
#     output[output < 0.5] = 0
#     output = output.astype(np.uint8)

#     # 使用并行处理加速形态学操作
#     with ThreadPoolExecutor() as executor:
#         output = np.array(list(executor.map(process_channel, output)))

#     # 保存结果
#     np.save(os.path.join(label_path, img).replace("jpg", "npy"), output)


# 用训练好的模型生成伪标签
import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from skimage.morphology import opening, closing, disk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 加载模型
model = load_model(
    name="deeplabv3p",
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=52,
    activation="sigmoid",
).eval()

# 读取权重
cps_path = "test"
models = []
cps = os.listdir(cps_path)
for cp in cps:
    model.load_state_dict(torch.load(os.path.join(cps_path, cp)))
    models.append(model.to("cuda"))

# 路径设置
img_path = "data/STS24_2D_XRAY/Resized-Train-Unlabeled"
label_path = "data/STS24_2D_XRAY/Resized-Train-Unlabeled-Masks-v2"
if not os.path.exists(label_path):
    os.makedirs(label_path)

imgs = os.listdir(img_path)


# # 并行处理函数，用于处理单个通道的形态学操作
# def process_channel(channel):
#     selem = disk(5)  # 结构元素
#     channel = closing(channel, selem, out=channel)  # 闭运算
#     channel = opening(channel, selem, out=channel)  # 开运算
#     return channel


def process_channel(channel):
    # 查找轮廓
    contours, _ = cv2.findContours(channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓，判断点的个数
    for contour in contours:
        if len(contour) <= 20:
            cv2.drawContours(
                channel, [contour], -1, 0, thickness=cv2.FILLED
            )  # 将其填充为0

    return channel


# 主处理循环
for img in tqdm(imgs):
    data = cv2.imread(os.path.join(img_path, img))
    data = T.ToTensor()(data).unsqueeze(0).to("cuda")
    data_ = data.flip(dims=[2])

    # 聚合多个模型的预测结果
    output = 0
    for model in models:
        output += model(data)
        output += model(data_).flip(dims=[2])

    output = (output / 2 / len(models)).detach().squeeze(0).cpu().numpy()
    output[output >= 0.5] = 255
    output[output < 0.5] = 0
    output = output.astype(np.uint8)

    # 使用并行处理加速形态学操作
    with ThreadPoolExecutor() as executor:
        output = np.array(list(executor.map(process_channel, output)))

    # 保存结果
    np.save(os.path.join(label_path, img).replace("jpg", "npy"), output)
