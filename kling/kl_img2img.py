import requests
import torch
import time
import jwt
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageSequence, ImageOps
import json
import os

ak = ""  # 填写access key
sk = ""  # 填写secret key


def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800,  # 有效时间，此处示例代表当前时间+1800s(30min)
        "nbf": int(time.time()) - 5  # 开始生效的时间，此处示例代表当前时间-5秒
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token


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
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        output_images.append(image)

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)

    else:
        output_image = output_images[0]

    return output_image


class Klingimg2img:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["kling-v1"],),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "batch": ("INT", {"default": 1, "min": 1, "max": 9}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"],)

            },
            "optional": {
                "image": ("IMAGE",),
                "image_fidelity": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "img2img"
    CATEGORY = "kling_image"

    def img2img(self,
             model: str,
             prompt: str,
             negative_prompt: str,
             batch: int,
             aspect_ratio: str,
             image: torch.Tensor,
             image_fidelity: float,
             ):
        token = encode_jwt_token(ak, sk)  # 在每次生成视频时生成新token
        image_base64 = tensor_to_base64(image)
        if prompt == "" or prompt is None:
            raise Exception("Prompt cannot be empty")
        payload = {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "n": batch,
            "aspect_ratio": aspect_ratio,
            "image": image_base64,
            "image_fidelity": image_fidelity
        }
        payload_str = json.dumps(payload)

        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # print("payload", payload)
        url = "https://api.klingai.com/v1/images/generations"

        response = requests.request("POST", url, headers=headers, data=payload_str)
        response.raise_for_status()
        response_data = response.json()["data"]
        print("response_data",response_data)
        task_id = response_data["task_id"]
        print(f"task_id: {task_id}")
        image_list_url = f"https://api.klingai.com/v1/images/generations/{task_id}"
        result = json.loads(requests.get(image_list_url, headers=headers).text)["data"]["task_status"]
        print("Initial Task Status:", result)
        while result != "succeed":
            time.sleep(5)
            response_image= requests.get(image_list_url, headers=headers)
            result = json.loads(response_image.text)["data"]["task_status"]
            print("Updated Task Status:", result)
        img_urls = json.loads(response_image.text)["data"]["task_result"]["images"]

        tensors = []
        for idx, img_url in enumerate(img_urls):
            finally_url = img_url["url"]
            tensors.append(finally_url)
            # print("tensors",tensors)

        for idx, imgs_url in enumerate(tensors):
            print(f"{idx}: {imgs_url}")
            img, _name = load_image(imgs_url)
            image_tensor = pil2tensor(img)
            tensors[idx] = image_tensor
        concatenated_tensor = torch.cat(tensors, dim=0)
        if concatenated_tensor.dtype != torch.float32:
            concatenated_tensor = concatenated_tensor.float()
        if concatenated_tensor.dim() != 4:
            raise ValueError(f"Final output tensor has unsupported shape: {concatenated_tensor.shape}")

        return (concatenated_tensor,)


NODE_CLASS_MAPPINGS = {
    "Klingimg2img": Klingimg2img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Klingimg2img": "Klingimg2img"
}
