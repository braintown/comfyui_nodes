from .kl_img2img import Klingimg2img
from .kl_txt2img import Klingtxt2img

NODE_CLASS_MAPPINGS = {
    "Klingimg2img": Klingimg2img,
    "Klingtxt2img": Klingtxt2img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Klingimg2img": "Klingimg2img",
    "Klingtxt2img": "Klingtxt2img"
}


all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
