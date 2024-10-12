from .ideogram_utils import *
import requests
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import json


class IdeogramUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "realistic, high quality",
                    "multiline": True
                }),
                "image": ("IMAGE",),
                "resemblance": ("INT", {"default": 50, "min": 1, "max": 100}),
                "detail": ("INT", {"default": 50, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 1234}),
                "magic_prompt_option": (["AUTO", "ON", "OFF"],)
            },
            "optional": {
                "api_key": ("STRING",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "Upscale"
    CATEGORY = "Ideogram/upscale"

    def Upscale(self,
                prompt: str,
                image: torch.Tensor,
                resemblance: int,
                detail: int,
                seed: int,
                magic_prompt_option: str,
                api_key: str
                ):
        if len(api_key) == 0 or api_key is None:
            if API_KEY is None:
                raise Exception("Must configure the API key in env_var `IDEOGRAM_KEY` or on the node.")
            api_key = API_KEY

        upscale_url = "https://api.ideogram.ai/upscale"

        payload = {
            'image_request': json.dumps({
                "prompt": prompt,
                "resemblance": resemblance,
                "detail": detail,
                "magic_prompt_option": magic_prompt_option,
                "seed": seed
            })
        }
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

        response = requests.post(upscale_url, files=files, data=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()["data"][0]
        is_image_safe = response_data["is_image_safe"]
        if not is_image_safe:
            raise Exception("Ideogram reports the generated image is not safe. Image not available.")
        img_url = response_data["url"]
        img, _name = load_image(img_url)
        img_out, mask_out = pil2tensor(img)
        return img_out, mask_out


NODE_CLASS_MAPPINGS = {
    "IdeogramUpscale": IdeogramUpscale
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramUpscale": "IdeogramUpscale"
}
