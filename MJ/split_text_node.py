class SplitText:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING",),
                "number": ("INT", {"default": 1, "min": 1, "max": 4})
            },

        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("split_text",)
    FUNCTION = "split_text"
    CATEGORY = "Mj/split_text"

    def split_text(self, text: str, number: int):
        segments = text.split('\n\n')  # 忽略分割后的第一个空元素
        text = segments[number - 1]

        return (text,)


NODE_CLASS_MAPPINGS = {
    "SplitText": SplitText
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitText": "SplitText"
}
