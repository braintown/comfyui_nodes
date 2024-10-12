
class ColorSelect:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("COLOR", {"default": "#FFFFFF"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = 'picker'
    CATEGORY = 'gempoll/Ideogram/select'

    def picker(self, color):
        ret = color
        return (ret,)

class Seed_select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": ("BOOLEAN", {"default": True, "label_on": "random", "label_off": "fixed"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "fixed_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    CATEGORY = "gempoll/Ideogram/seed"
    FUNCTION = "get_seed"

    def get_seed(self, mode, seed, fixed_seed):
        return (fixed_seed if not mode else seed,)



NODE_CLASS_MAPPINGS = {
    "ColorSelect": ColorSelect,
    "Seed_select": Seed_select
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorSelect": "ColorSelect",
    "Seed_select": "Seed_select"
}