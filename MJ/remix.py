from .utils import *
import requests

import torch


class MjImg2Img:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "dimensions": (["PORTRAIT", "SQUARE", "LANDSCAPE"],),
                "botType": (["MID_JOURNEY", "NIJI_JOURNEY"],),
            },

        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "image2image"
    CATEGORY = "Mj/image2image"

    def image2image(self,
                   image1: torch.Tensor,
                   image2: torch.Tensor,
                   api_key: str,
                   dimensions: str,
                   botType: str
                   ):

        if len(api_key) == 0 or api_key is None:
            if API_KEY is None:
                raise Exception("Must configure the API key in env_var `IDEOGRAM_KEY` or on the node.")
            mj_secret = API_KEY
        else:
            mj_secret = api_key

        img_url = "https://mj-backend.gempoll.com/mj/submit/blend"

        headers = {
            "Mj-Api-Secret": mj_secret
        }

        image1_base64 = tensor_to_base64(image1)
        image2_base64 = tensor_to_base64(image2)
        data_uri1 = f"data:image/jpeg;base64,{image1_base64}"
        data_uri2 = f"data:image/jpeg;base64,{image2_base64}"
        data = {
            "botType": botType,
            "base64Array": [data_uri1, data_uri2],
            "dimensions": dimensions
        }

        response = requests.post(img_url, json=data, headers=headers)
        response.raise_for_status()

        result_id = response.json()["result"]

        describe_result_url = "https://mj-backend.gempoll.com/mj/task/list-by-condition"

        data_result = {
            "ids": [result_id]
        }

        while True:
            response_result = requests.post(describe_result_url, json=data_result, headers=headers)
            response_result.raise_for_status()

            result_image_url = response_result.json()
            print(result_image_url[0]["progress"])
            if result_image_url[0]["progress"] == "100%":
                break
        print('-----------开始获取单个图片-------------')
        img_url = result_image_url[0]["imageUrl"]
        payload = {"url": img_url}

        slice_up_url = "https://ai-dev.gempoll.com/v5/process-image"
        response_slice_up = requests.post(slice_up_url, json=payload, headers=headers)
        response_slice_up.raise_for_status()
        img_out_list = []
        print('-----------开始转tensor-------------')
        for img_url_info in response_slice_up.json()['urls']:
            img, _name = load_image(img_url_info)
            img_out, mask_out = pil2tensor(img)
            img_out_list.append(img_out)
        print('-----------处理图片加标记-------------')
        for idx, image_tensor in enumerate(img_out_list):
            # Ensure tensor is in shape (1, C, H, W)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # 添加批量维度

            img_out_list[idx] = image_tensor

        concatenated_tensor = torch.cat(img_out_list, dim=0)

        if concatenated_tensor.dtype != torch.float32:
            concatenated_tensor = concatenated_tensor.float()
        if concatenated_tensor.dim() != 4:
            raise ValueError(f"Final output tensor has unsupported shape: {concatenated_tensor.shape}")

        return (concatenated_tensor,)  # 返回一个 Tensor 的元组


NODE_CLASS_MAPPINGS = {
    "MjImg2Img": MjImg2Img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MjImg2Img": "MjImg2Img"
}
