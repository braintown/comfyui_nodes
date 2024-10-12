import os
from io import BytesIO
import torch
import numpy as np
from PIL import Image, ImageSequence, ImageOps
import requests
from typing import List

API_KEY = os.environ.get("IDEOGRAM_KEY", None)

MIN_COLOR_WEIGHT_FLOAT = 0.05  # weight >= 0.05, required by Ideogram
ROUNDING_MULTIPLIER = 100  # 保留百分位
MIN_COLOR_WEIGHT_INT = int(MIN_COLOR_WEIGHT_FLOAT * ROUNDING_MULTIPLIER)

RESOLUTION_DEFAULT = "NotSelected"

RESOLUTIONS = [
    RESOLUTION_DEFAULT,
    "RESOLUTION_1024_640",
    "RESOLUTION_1024_768",
    "RESOLUTION_1024_832",
    "RESOLUTION_1024_896",
    "RESOLUTION_1024_960",
    "RESOLUTION_1024_1024",
    "RESOLUTION_1088_768",
    "RESOLUTION_1088_832",
    "RESOLUTION_1088_896",
    "RESOLUTION_1088_960",
    "RESOLUTION_1120_896",
    "RESOLUTION_1152_704",
    "RESOLUTION_1152_768",
    "RESOLUTION_1152_832",
    "RESOLUTION_1152_864",
    "RESOLUTION_1152_896",
    "RESOLUTION_1216_704",
    "RESOLUTION_1216_768",
    "RESOLUTION_1216_832",
    "RESOLUTION_1232_768",
    "RESOLUTION_1248_832",
    "RESOLUTION_1280_704",
    "RESOLUTION_1280_720",
    "RESOLUTION_1280_768",
    "RESOLUTION_1280_800",
    "RESOLUTION_1312_736",
    "RESOLUTION_1344_640",
    "RESOLUTION_1344_704",
    "RESOLUTION_1344_768",
    "RESOLUTION_1408_576",
    "RESOLUTION_1408_640",
    "RESOLUTION_1408_704",
    "RESOLUTION_1472_576",
    "RESOLUTION_1472_640",
    "RESOLUTION_1472_704",
    "RESOLUTION_1536_512",
    "RESOLUTION_1536_576",
    "RESOLUTION_1536_640",
    "RESOLUTION_512_1536",
    "RESOLUTION_576_1408",
    "RESOLUTION_576_1472",
    "RESOLUTION_576_1536",
    "RESOLUTION_640_1024",
    "RESOLUTION_640_1344",
    "RESOLUTION_640_1408",
    "RESOLUTION_640_1472",
    "RESOLUTION_640_1536",
    "RESOLUTION_704_1152",
    "RESOLUTION_704_1216",
    "RESOLUTION_704_1280",
    "RESOLUTION_704_1344",
    "RESOLUTION_704_1408",
    "RESOLUTION_704_1472",
    "RESOLUTION_720_1280",
    "RESOLUTION_736_1312",
    "RESOLUTION_768_1024",
    "RESOLUTION_768_1088",
    "RESOLUTION_768_1152",
    "RESOLUTION_768_1216",
    "RESOLUTION_768_1232",
    "RESOLUTION_768_1280",
    "RESOLUTION_768_1344",
    "RESOLUTION_832_960",
    "RESOLUTION_832_1024",
    "RESOLUTION_832_1088",
    "RESOLUTION_832_1152",
    "RESOLUTION_832_1216",
    "RESOLUTION_832_1248",
    "RESOLUTION_864_1152",
    "RESOLUTION_896_960",
    "RESOLUTION_896_1024",
    "RESOLUTION_896_1088",
    "RESOLUTION_896_1120",
    "RESOLUTION_896_1152",
    "RESOLUTION_960_832",
    "RESOLUTION_960_896",
    "RESOLUTION_960_1024",
    "RESOLUTION_960_1088",
]

RESOLUTION_MAPPING = {
    v.replace("RESOLUTION_", "").replace("_", "x"): v
    for v in RESOLUTIONS
}

ASPECT_RATIO_DEFAULT = "NotSelected"

ASPECT_RATIOS = [
    ASPECT_RATIO_DEFAULT,
    "ASPECT_10_16",
    "ASPECT_16_10",
    "ASPECT_9_16",
    "ASPECT_16_9",
    "ASPECT_3_2",
    "ASPECT_2_3",
    "ASPECT_4_3",
    "ASPECT_3_4",
    "ASPECT_4_5",
    "ASPECT_5_4",
    "ASPECT_1_1",
    "ASPECT_1_3",
    "ASPECT_3_1"
]

ASPECT_RATIO_MAPPING = {
    v.replace("ASPECT_", "").replace("_", ":"): v
    for v in ASPECT_RATIOS
}


def load_image(image_source):
    if image_source.startswith('http'):
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content))
        file_name = image_source.split('/')[-1]
    else:
        img = Image.open(image_source)
        file_name = os.path.basename(image_source)
    return img, file_name


def pil2tensor(img):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return output_image, output_mask


def check_color_hex(hex_str: str):
    if hex_str.startswith("#"):
        hex_str = hex_str[1:]
        assert len(hex_str) == 6, "Color must be 6 characters long"
        r, g, b = hex_str[:2], hex_str[2:4], hex_str[4:]
        try:
            int(r, 16)
            int(g, 16)
            int(b, 16)
        except ValueError as e:
            raise Exception(f"Color must be valid hex, error: {e}")
    else:
        raise Exception("Color must start with #")


def rectify_weights(weights: List[float]) -> List[float]:
    # 计算剩余元素的总和
    total_weight = sum(weights)

    # 对剩余的元素进行归一化处理
    normalized_weight = [i / total_weight for i in weights]

    # 去除小数点后两位的数字，并计算这些被去除值的总和
    truncated_weight_int = [int(i * ROUNDING_MULTIPLIER) for i in normalized_weight]
    total_weight_gap = ROUNDING_MULTIPLIER - sum(truncated_weight_int)
    # 找到剩余元素中的最大值
    min_index = truncated_weight_int.index(min(truncated_weight_int))
    # 将被去除值的总和补到最小的数字上
    truncated_weight_int[min_index] += total_weight_gap

    # 补充低于 MIN_COLOR_WEIGHT_INT 的权重
    for (i, val) in enumerate(truncated_weight_int):
        if val < MIN_COLOR_WEIGHT_INT:
            diff = MIN_COLOR_WEIGHT_INT - val
            truncated_weight_int[i] = MIN_COLOR_WEIGHT_INT
            max_index = truncated_weight_int.index(max(truncated_weight_int))
            truncated_weight_int[max_index] -= diff

    for val in truncated_weight_int:
        assert val >= MIN_COLOR_WEIGHT_INT, "Impossible"

    rectified_weights = [val / ROUNDING_MULTIPLIER for val in truncated_weight_int]
    return rectified_weights