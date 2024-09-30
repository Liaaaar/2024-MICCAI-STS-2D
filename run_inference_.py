import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import json
from skimage.morphology import opening, closing, disk
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = "data/STS24_2D_XRAY/Validation-Public"
OUTPUT_DIR = "output"

map = {
    0: "11",
    1: "12",
    2: "13",
    3: "14",
    4: "15",
    5: "16",
    6: "17",
    7: "18",
    8: "21",
    9: "22",
    10: "23",
    11: "24",
    12: "25",
    13: "26",
    14: "27",
    15: "28",
    16: "31",
    17: "32",
    18: "33",
    19: "34",
    20: "35",
    21: "36",
    22: "37",
    23: "38",
    24: "41",
    25: "42",
    26: "43",
    27: "44",
    28: "45",
    29: "46",
    30: "47",
    31: "48",
    32: "51",
    33: "52",
    34: "53",
    35: "54",
    36: "55",
    37: "61",
    38: "62",
    39: "63",
    40: "64",
    41: "65",
    42: "71",
    43: "72",
    44: "73",
    45: "74",
    46: "75",
    47: "81",
    48: "82",
    49: "83",
    50: "84",
    51: "85",
}

# aaa = []


@torch.no_grad()
def process_single_channel(i, output, selem, w, h):
    temp = output[i]
    # temp = closing(temp, selem)
    # temp = opening(temp, selem)
    label = map.get(i)
    temp = cv2.resize(temp, (w, h))
    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        if len(contour) > 0:
            points = contour.squeeze().tolist()
            if len(points) > 45:
                # aaa.append(len(points))
                # print(min(aaa))
                # points = [[x * w / 640, y * h / 320] for x, y in points]
                shapes.append({"label": label, "points": points})
    return shapes


import time  # 添加这一行


@torch.no_grad()
def main():
    device = torch.device("cuda:0")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=52,
        activation="sigmoid",
    ).to(device)
    model.eval()
    model.load_state_dict(torch.load("new_train_v2/epoch199.pth"))
    to_tensor = T.ToTensor()
    selem = disk(5)

    total_inference_time = 0  # 初始化总推理时间
    num_images = 0  # 记录处理的图片数量

    for case_name in tqdm(sorted(os.listdir(INPUT_DIR))):
        start_time = time.time()  # 开始时间
        img = cv2.imread(os.path.join(INPUT_DIR, case_name))
        h, w, c = img.shape
        img = cv2.resize(img, (640, 320))
        img = to_tensor(img).unsqueeze(0).to(device)
        img_aug = img.flip(dims=[2])
        output = (model(img) + model(img_aug).flip(dims=[2])) / 2
        output = output.detach().squeeze(0).cpu().numpy()
        output[output >= 0.5] = 255
        output[output < 0.5] = 0
        output = output.astype(np.uint8)

        shapes = []

        # 使用 ThreadPoolExecutor 并行处理 52 个通道
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_single_channel, i, output, selem, w, h)
                for i in range(52)
            ]
            for future in futures:
                shapes.extend(future.result())

        output = {"shapes": shapes, "imageHeight": h, "imageWidth": w}

        json_output = json.dumps(output, indent=4)

        case_name = case_name.split(".")[0]
        with open(
            os.path.join(OUTPUT_DIR, f"{case_name}_Mask.json"),
            "w",
        ) as f:
            f.write(json_output)

        end_time = time.time()  # 结束时间
        inference_time = end_time - start_time  # 计算推理时间
        total_inference_time += inference_time  # 累加推理时间
        num_images += 1  # 增加处理图片计数

    average_inference_time = total_inference_time / num_images if num_images > 0 else 0
    print(
        f"Average inference time per image: {average_inference_time:.4f} seconds"
    )  # 打印平均推理时间


main()
