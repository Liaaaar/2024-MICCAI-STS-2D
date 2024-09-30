import json
import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm


# 得到牙齿label的映射字典
# json_dir：存放训练数据json文件的路径
def get_label_map(json_dir):
    labels = []
    json_paths = glob(os.path.join(json_dir, "*.json"))
    for json_path in json_paths:
        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
            shapes = data["shapes"]
            for shape in shapes:
                labels.append(shape["label"])
    labels = sorted(set(labels))
    map = {value: index for index, value in enumerate(labels)}
    return map


# get_label_map("data/STS24_2D_XRAY/Train-Labeled/Jsons")


# 将json文件转为ndarray格式的mask
# json_dir：存放json文件的文件夹
# output_dir：mask的输出路径
def json2label(json_dir, output_dir):
    map = get_label_map(json_dir)
    json_paths = glob(os.path.join(json_dir, "*.json"))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for json_path in tqdm(json_paths):
        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
            shapes = data["shapes"]
            h, w = data["imageHeight"], data["imageWidth"]
            # print(h, w)
            # continue
            name = data["imagePath"]
            mask = np.zeros((52, h, w), dtype=np.uint8)
            for shape in shapes:
                points = np.array(shape["points"], dtype=np.int32)
                label = map[shape["label"]]
                cv2.fillPoly(mask[label], [points], 255)
        np.save(os.path.join(output_dir, name).replace("jpg", "npy"), mask)


def json2label2(json_dir, output_dir):
    temp = []
    # 获取标签映射
    label_map = get_label_map(json_dir)  # 你应该已经有此函数，返回 label 对应的数值映射
    json_paths = glob(os.path.join(json_dir, "*.json"))

    # 如果输出目录不存在，创建该目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每个JSON文件
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            shapes = data["shapes"]
            h, w = data["imageHeight"], data["imageWidth"]
            name = data["imagePath"].replace("jpg", "png")

            # 初始化单通道mask
            mask = np.zeros((h, w), dtype=np.uint8)

            # 遍历每个形状并填充相应区域
            for shape in shapes:
                points = np.array(shape["points"], dtype=np.int32)
                label = label_map[shape["label"]] + 1
                temp.append(label)
                # print(max(temp))

                # 创建一个临时mask用于当前形状
                temp_mask = np.zeros_like(mask)
                cv2.fillPoly(temp_mask, [points], label)

                # 只填充mask中未被覆盖的部分
                mask = np.where(mask == 0, temp_mask, mask)

        print(mask.max())

        # 将 mask 存为图像文件（PNG格式常用于标签图）
        cv2.imwrite(os.path.join(output_dir, name), mask)


# json2label(
#     "data/STS24_2D_XRAY/Train-Labeled/Jsons",
#     "data/STS24_2D_XRAY/Train-Labeled/Masks",
# )

# json2label2(
#     "data/STS24_2D_XRAY/Train-Labeled/Jsons",
#     "data/STS24_2D_XRAY/Train-Labeled/Masks2",
# )


# resize图像
# input_dir：原始图像所在的文件夹
# output_dir：resize后图像的输出路径
def resize_images(input_dir, output_dir, target_size=(512, 1024)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_paths = glob(os.path.join(input_dir, "*.png"))
    for image_path in image_paths:
        img = cv2.imread(image_path)
        # print(img.max())
        if img is None:
            print(f"Failed to load image {image_path}")
            continue
        # resized_img = cv2.resize(img, (target_size[1], target_size[0]))
        resized_img = cv2.resize(
            img, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST
        )  # resize mask
        print(resized_img.max(), img.max())

        save_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, resized_img)
        print(f"Image saved to {save_path}")


# resize mask
# input_dir：原始mask所在的文件夹
# output_dir：resize后mask的输出路径
def resize_npy_files(input_dir, output_dir, target_size=(512, 1024)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    npy_paths = glob(os.path.join(input_dir, "*.npy"))
    for npy_path in npy_paths:
        array = np.load(npy_path)
        # print(array.shape)
        if array.shape[0] != 52:
            print(f"Unexpected shape {array.shape} in {npy_path}")
            continue
        resized_array = np.zeros(
            (52, target_size[0], target_size[1]), dtype=array.dtype
        )
        for i in range(52):
            resized_array[i] = cv2.resize(array[i], (target_size[1], target_size[0]))

        save_path = os.path.join(output_dir, os.path.basename(npy_path))
        # print(resized_array.shape)
        np.save(save_path, resized_array)
        print(f"Numpy array saved to {save_path}")


# resize_npy_files(
#     "data/STS24_2D_XRAY/Train-Labeled/Masks",
#     "data/STS24_2D_XRAY/Train-Labeled/Resized_Masks",
#     target_size=(320, 640),
# )

# resize_images(
#     "data/STS24_2D_XRAY/Train-Labeled/Masks2",
#     "data/STS24_2D_XRAY/Train-Labeled/Resized_Masks2",
#     target_size=(320, 640),
# )
