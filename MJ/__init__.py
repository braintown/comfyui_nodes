from .describe import MjDescribe
from .remix import MjImg2Img
from .split_text_node import SplitText
from .text2image import MjTxt2Img

NODE_CLASS_MAPPINGS = {
    "MjDescribe": MjDescribe,
    "MjTxt2Img": MjTxt2Img,
    "SplitText": SplitText,
    "MjImg2Img": MjImg2Img
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MjDescribe": "MjDescribe",
    "MjTxt2Img": "MjTxt2Img",
    "SplitText": "SplitText",
    "MjImg2Img": "MjImg2Img"

}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
