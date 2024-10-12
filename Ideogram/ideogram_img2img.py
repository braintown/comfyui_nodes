from .ideogram_utils import *
import requests
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import json


class IdeogramImg2Img:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "(shadow), silver grey Cadillac xt5 SUV car on the edge of the mountain,sunlight,afternoon,shot from front right side,realistic film photo",
                    "multiline": True
                }),
                "image": ("IMAGE",),
                "aspect_ratio": (list(ASPECT_RATIO_MAPPING.keys()),),
                "resolution": (list(RESOLUTION_MAPPING.keys()),),
                "seed": ("INT", {"default": 1234}),
                "image_weight": ("INT", {"default": 50, "min": 1, "max": 100}),
                "model": (["V_2", "V_2_TURBO", "V_1", "V_1_TURBO"],),
                "magic_prompt_option": (["AUTO", "ON", "OFF"],),
                "negative_prompt": ("STRING", {"default": ""}),
                "color_palette_hex1": ("STRING",),
                "color_palette_weight1": ("FLOAT", {"default": 10, "min": 0, "max": 10}),
                "color_palette_hex2": ("STRING",),
                "color_palette_weight2": ("FLOAT", {"default": 10, "min": 0, "max": 10}),
                "color_palette_hex3": ("STRING",),
                "color_palette_weight3": ("FLOAT", {"default": 10, "min": 0, "max": 10}),
                "color_palette_hex4": ("STRING",),
                "color_palette_weight4": ("FLOAT", {"default": 10, "min": 0, "max": 10}),
            },
            "optional": {
                "api_key": ("STRING",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "Img2img"
    CATEGORY = "Ideogram/img2img"

    def Img2img(self,
                prompt: str,
                image: torch.Tensor,
                aspect_ratio: str,
                resolution: str,
                seed: int,
                image_weight: int,
                model: str,
                magic_prompt_option: str,
                negative_prompt: str,
                color_palette_hex1: str, color_palette_weight1: float,
                color_palette_hex2: str, color_palette_weight2: float,
                color_palette_hex3: str, color_palette_weight3: float,
                color_palette_hex4: str, color_palette_weight4: float,
                api_key: str):
        if len(api_key) == 0 or api_key is None:
            if API_KEY is None:
                raise Exception("Must configure the API key in env_var `IDEOGRAM_KEY` or on the node.")
            api_key = API_KEY

        img2img_generate_url = "https://api.ideogram.ai/remix"
        weights = [color_palette_weight1, color_palette_weight2, color_palette_weight3, color_palette_weight4]
        colors = [color_palette_hex1, color_palette_hex2, color_palette_hex3, color_palette_hex4]
        for c in colors:
            check_color_hex(c)
        resolution = RESOLUTION_MAPPING[resolution]
        aspect_ratio = ASPECT_RATIO_MAPPING[aspect_ratio]
        image_weight = int(image_weight)

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

        if resolution == RESOLUTION_DEFAULT and aspect_ratio == ASPECT_RATIO_DEFAULT:
            raise Exception("Must select one of aspect ratio and resolution")
        if resolution != RESOLUTION_DEFAULT and aspect_ratio != ASPECT_RATIO_DEFAULT:
            raise Exception("Should not select both aspect ratio and resolution")

        # 移除权重为0的元素及其对应的颜色
        weights_colors = [(w, c) for w, c in zip(weights, colors) if w != 0.0]

        if not weights_colors:
            raise ValueError("All weights are zero, cannot proceed with image generation.")

        weights, colors = zip(*weights_colors)
        rectified_weights = rectify_weights(weights)
        color_palette = [
            {"color_hex": color, "color_weight": color_weight}
            for color, color_weight in zip(colors, rectified_weights)
        ]

        image_request = {
            "prompt": prompt,
            "model": model,
            "magic_prompt_option": magic_prompt_option,
            "negative_prompt": negative_prompt,
            "image_weight": image_weight,
            "seed": seed,
            "color_palette": {
                "members": color_palette
            },
        }

        if resolution != RESOLUTION_DEFAULT:
            image_request["resolution"] = resolution

        if aspect_ratio != ASPECT_RATIO_DEFAULT:
            image_request["aspect_ratio"] = aspect_ratio

        payload = {
            'image_request': json.dumps(image_request)
        }

        headers = {
            "Api-Key": api_key
        }

        response = requests.post(img2img_generate_url, data=payload, files=files, headers=headers)
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
    "IdeogramImg2Img": IdeogramImg2Img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramImg2Img": "IdeogramImg2Img"
}
