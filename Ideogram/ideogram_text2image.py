from .ideogram_utils import *
import requests

class IdeogramTxt2Img:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "(shadow), silver grey Cadillac xt5 SUV car on the edge of the mountain,sunlight,afternoon,shot from front right side,realistic film photo",
                    "multiline": True
                }),
                "aspect_ratio": (list(ASPECT_RATIO_MAPPING.keys()),),
                "resolution": (list(RESOLUTION_MAPPING.keys()),),
                "seed": ("INT", {"default": 1234}),
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
    FUNCTION = "text2image"
    CATEGORY = "Ideogram/txt2img"

    def text2image(self,
                   prompt: str,
                   aspect_ratio: str,
                   resolution: str,
                   model: str,
                   magic_prompt_option: str,
                   negative_prompt: str,
                   seed: int,
                   color_palette_hex1: str, color_palette_weight1: float,
                   color_palette_hex2: str, color_palette_weight2: float,
                   color_palette_hex3: str, color_palette_weight3: float,
                   color_palette_hex4: str, color_palette_weight4: float,
                   api_key: str):
        if len(api_key) == 0 or api_key is None:
            if API_KEY is None:
                raise Exception("Must configure the API key in env_var `IDEOGRAM_KEY` or on the node.")
            api_key = API_KEY

        txt2img_generate_url = "https://api.ideogram.ai/generate"
        weights = [color_palette_weight1, color_palette_weight2, color_palette_weight3, color_palette_weight4]
        colors = [color_palette_hex1, color_palette_hex2, color_palette_hex3, color_palette_hex4]
        for c in colors:
            check_color_hex(c)
        resolution = RESOLUTION_MAPPING[resolution]
        aspect_ratio = ASPECT_RATIO_MAPPING[aspect_ratio]

        if resolution == RESOLUTION_DEFAULT and aspect_ratio == ASPECT_RATIO_DEFAULT:
            raise Exception("Must select one of aspect ratio and resolution")
        if resolution != RESOLUTION_DEFAULT and aspect_ratio != ASPECT_RATIO_DEFAULT:
            raise Exception("Should not select both aspect ratio and resolution")

        # 移除权重为0的元素及其对应的颜色
        weights_colors = [(w, c) for w, c in zip(weights, colors) if w != 0.0]
        assert len(weights_colors) > 0, "All weights are zero, cannot proceed with image generation."
        weights, colors = zip(*weights_colors)
        rectified_weights = rectify_weights(weights)
        color_palette = [
            {"color_hex": color, "color_weight": color_weight}
            for color, color_weight in zip(colors, rectified_weights)
        ]

        payload = {
            "image_request": {
                "prompt": prompt,
                "model": model,
                "magic_prompt_option": magic_prompt_option,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "color_palette": {
                    "members": color_palette
                },
            }
        }

        if resolution != RESOLUTION_DEFAULT:
            payload["image_request"]["resolution"] = resolution

        if aspect_ratio != ASPECT_RATIO_DEFAULT:
            payload["image_request"]["aspect_ratio"] = aspect_ratio

        headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }

        print("payload", payload)


        response = requests.post(txt2img_generate_url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()["data"][0]
        is_image_safe = response_data["is_image_safe"]
        assert is_image_safe, "Ideogram reports the generated image is not safe. Image not available."
        img_url = response_data["url"]
        img, _name = load_image(img_url)
        img_out, mask_out = pil2tensor(img)
        return img_out, mask_out


NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img"
}
