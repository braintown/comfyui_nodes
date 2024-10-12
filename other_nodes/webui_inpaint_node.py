import requests
import time
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import folder_paths



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


def base64_to_tensor(base64_str):
    """
    将 Base64 编码的字符串转换为 PyTorch Tensor
    """
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def get_image(task_id):
    """
    根据任务 ID 获取图像
    """
    url_new = f"http://10.7.100.40:11007/agent-scheduler/v1/task/{task_id}/results"

    while True:
        try:
            response_new = requests.get(url_new)
            response_new.raise_for_status()
            response_json = response_new.json()

            if response_json['success']:
                image_tensors = []
                for item in response_json['data'][:-1]:
                    if 'image' in item:
                        image_base64 = item['image'].split(',')[1]
                        image_tensor = base64_to_tensor(image_base64)

                        # Debugging: Print tensor shape and dtype
                        print(f"Retrieved image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")

                        # Ensure tensor is in shape (1, C, H, W)
                        if image_tensor.dim() == 3:
                            image_tensor = image_tensor.unsqueeze(0)  # 如果 tensor 是 (C, H, W)，则添加一个维度
                        elif image_tensor.dim() != 4:
                            raise ValueError(f"Unsupported tensor shape: {image_tensor.shape}")

                        # Ensure tensor is of type float32
                        if image_tensor.dtype != torch.float32:
                            image_tensor = image_tensor.float()

                        image_tensors.append(image_tensor)
                if image_tensors:
                    return image_tensors
                else:
                    print("Task is not completed yet. Retrying in 5 seconds...")
                    time.sleep(5)
            else:
                print("Task is not completed yet. Retrying in 5 seconds...")
                time.sleep(5)
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(5)


class webui_inpaint:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Automatic", "Anything-V3.0-vae.pt", "anytwam11Mixedmodel_anytwam11二次元.vae.pt"],),
                # "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "original_image": ("IMAGE",),
                "mask": ("IMAGE",),
                "positive": ("STRING", {"default": "(shadow), silver grey Cadillac xt5 SUV car on the edge of the mountain,sunlight,afternoon,shot from front right side,realistic film photo", "multiline": True}),
                "negative": ("STRING", {"default": "oversaturated, grayscale, blurry, color cast, faded, old, washed out, low quality, lackluster", "multiline": True}),
                # "n_iter": ("INT", {"default": 1, "min": 1, "max": 20}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 200}),
                "sampler_name": (["Euler", "Euler a", "DPM++ 2M", "DPM++ SDE", "DPM++ 3M SDE", "DDIM"],),
                # "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                # "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "scheduler": (["Automatic", "Uniform", "Karras", "Exponential", "KL Optimal"],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0}),
                "denoising_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 4294967295}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("inpaint_image",)
    FUNCTION = "webui_inpaint"
    CATEGORY = "gempoll/webui_inpaint"

    def webui_inpaint(self, ckpt_name, original_image, mask, positive, negative, batch_size, vae_name,
                      sampler_name, scheduler, steps, cfg_scale, denoising_strength, seed):
        # Debugging: Ensure input types and shapes
        # print(
        #     f"original_image type: {type(original_image)}, shape: {original_image.shape}, dtype: {original_image.dtype}")
        # print(f"mask type: {type(mask)}, shape: {mask.shape}, dtype: {mask.dtype}")

        url = "http://10.7.100.40:11007/agent-scheduler/v1/queue/img2img"

        # 将 Tensor 转换为 base64 字符串
        original_image_base64 = tensor_to_base64(original_image)
        mask_base64 = tensor_to_base64(mask)
        w, h = original_image.shape[2], original_image.shape[1]
        payload = {
            "prompt": positive,
            "negative_prompt": negative,
            "seed": seed,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "batch_size": batch_size,
            "n_iter": 1,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": w,
            "height": h,
            "denoising_strength": denoising_strength,
            "init_images": [original_image_base64],
            "resize_mode": 0,
            "image_cfg_scale": 0,
            "mask": mask_base64,
            "mask_blur": 4,
            "inpainting_fill": 1,
            "inpaint_full_res": 0,
            "inpaint_full_res_padding": 32,
            "inpainting_mask_invert": 0,
            "initial_noise_multiplier": 1,
            "include_init_images": 1,
            "checkpoint": ckpt_name,
            "vae": vae_name,
        }

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            task_id = response.json()["task_id"]
            image_tensors = get_image(task_id)

            # Debugging: Print output tensor shapes and dtypes
            for idx, image_tensor in enumerate(image_tensors):
                # Ensure tensor is in shape (1, C, H, W)
                if image_tensor.dim() == 3:
                    image_tensor = image_tensor.unsqueeze(0)  # 添加批量维度

                # Convert tensor to numpy array and then back to tensor to ensure correct type and shape
                image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
                image_tensor = torch.from_numpy(image_np).unsqueeze(0)
                print(f"Output image tensor {idx} shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
                image_tensors[idx] = image_tensor

            # Concatenate tensors along the channel dimension (dim=1)
            concatenated_tensor = torch.cat(image_tensors, dim=0)

            # Ensure the final tensor is of type float32 and shape (1, C, H, W)
            if concatenated_tensor.dtype != torch.float32:
                concatenated_tensor = concatenated_tensor.float()
            if concatenated_tensor.dim() != 4:
                raise ValueError(f"Final output tensor has unsupported shape: {concatenated_tensor.shape}")

            return (concatenated_tensor,)  # 返回一个 Tensor 的元组
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return (None,)  # 返回一个包含 None 的元组


NODE_CLASS_MAPPINGS = {
    "webui_inpaint": webui_inpaint
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "webui_inpaint": "webui_inpaint"
}
