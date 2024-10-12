from .ideogram_text2image import IdeogramTxt2Img
from .ideogram_img2img import IdeogramImg2Img
from .ideogram_upscale import IdeogramUpscale
from .ideogram_describe import IdeogramDescribe
from .mode_select import ColorSelect, Seed_select

NODE_CLASS_MAPPINGS = {
    "IdeogramTxt2Img": IdeogramTxt2Img,
    "IdeogramImg2Img": IdeogramImg2Img,
    "IdeogramUpscale": IdeogramUpscale,
    "IdeogramDescribe": IdeogramDescribe,
    "ColorSelect": ColorSelect,
    "Seed_select": Seed_select

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdeogramTxt2Img": "IdeogramTxt2Img",
    "IdeogramImg2Img": "IdeogramImg2Img",
    "IdeogramUpscale": "IdeogramUpscale",
    "IdeogramDescribe": "IdeogramDescribe",
    "ColorSelect": "ColorSelect",
    "Seed_select": "Seed_select"
}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
