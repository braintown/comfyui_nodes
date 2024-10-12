import base64
import os
from io import BytesIO
import torch
import numpy as np
from PIL import Image, ImageSequence, ImageOps
import requests
from typing import List

API_KEY = os.environ.get("mj_secret", None)


def tensor_to_base64(tensor):
    """
    将 PyTorch Tensor 转换为 Base64 编码的字符串
    """
    # Debugging: Print tensor shape and dtype
    print(f"tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

    # Ensure tensor is in range [0, 1]
    tensor = tensor.clamp(0, 1)

    # Ensure tensor is in shape (H, W, C)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # 如果 tensor 是 (1, H, W, C)，则去掉第一个维度
    elif tensor.dim() != 3:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    # Ensure tensor is of type float32
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    # PIL expects (H, W, C) format
    pil_image = Image.fromarray((tensor.mul(255).byte().cpu().numpy()).astype(np.uint8))
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

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