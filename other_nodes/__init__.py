from .text_node import Text_Node
from .webui_inpaint_node import webui_inpaint


NODE_CLASS_MAPPINGS = {
    "Text_Node": Text_Node,
    "webui_inpaint": webui_inpaint

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Text_Node": "Text_Node",
    "webui_inpaint": "webui_inpaint"
}

all = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']