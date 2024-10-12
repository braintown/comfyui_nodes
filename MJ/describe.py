import requests
import torch

from .utils import API_KEY, tensor_to_base64


class MjDescribe:

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
    FUNCTION = "Describe"
    CATEGORY = "Mj/describe"

    def Describe(self, image: torch.Tensor, api_key: str):
        if len(api_key) == 0 or api_key is None:
            if API_KEY is None:
                raise Exception("Must configure the API key in env_var `IDEOGRAM_KEY` or on the node.")
            mj_secret = API_KEY
        else:
            mj_secret = api_key

        describe_url = "https://mj-backend.gempoll.com/mj/submit/describe"

        headers = {
            "Mj-Api-Secret": mj_secret
        }

        image_base64 = tensor_to_base64(image)
        data_uri = f"data:image/jpeg;base64,{image_base64}"
        data = {
            "botType": "MID_JOURNEY",
            "base64": data_uri
        }

        response = requests.post(describe_url, json=data, headers=headers)
        response.raise_for_status()

        result_id = response.json()["result"]

        describe_result_url = "https://mj-backend.gempoll.com/mj/task/list-by-condition"

        data_result = {
            "ids": [result_id]
        }

        while True:
            response_result = requests.post(describe_result_url, json=data_result, headers=headers)
            response_result.raise_for_status()
            print(f"text:{response_result.json()}")
            text = response_result.json()[0]["promptEn"]

            if text is not None:
                break

        return (text,)


NODE_CLASS_MAPPINGS = {
    "MjDescribe": MjDescribe
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MjDescribe": "MjDescribe"
}
