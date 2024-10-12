from .ideogram_utils import *
import requests
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import json


class IdeogramDescribe:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DESCRIPTION",)
    FUNCTION = "describe"
    CATEGORY = "Ideogram/describe"

    def describe(self, image: torch.Tensor, api_key: str):
        if len(api_key) == 0 or api_key is None:
            if API_KEY is None:
                raise Exception("Must configure the API key in env_var `IDEOGRAM_KEY` or on the node.")
            api_key = API_KEY

        describle_url = "https://api.ideogram.ai/describe"
        # 检查输入图像的形状和数据类型
        if image.ndim != 4:
            raise ValueError("Input image tensor must be 4-dimensional (batch_size, channels, height, width)")
        else:
            image = image.squeeze(0)

        # 确保输入图像的数据类型为float32
        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        pil_image = Image.fromarray((image.mul(255).byte().cpu().numpy()).astype(np.uint8))
        image_bytes = BytesIO()
        pil_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)  # 确保字节流的指针在开头位置

        # 准备要发送的文件对象
        files = [
            ('image_file', ('image.png', image_bytes, 'application/octet-stream'))
        ]

        headers = {
            "Api-Key": api_key
        }

        response = requests.post(describle_url, files=files, headers=headers)
        response.raise_for_status()
        response_description = response.json()["descriptions"]
        text = response_description[0]["text"]
        return (text,)


NODE_CLASS_MAPPINGS = {
    "IdeogramDescribe": IdeogramDescribe
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramDescribe": "IdeogramDescribe"
}
