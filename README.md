# comfyui custom nodes
Some Sora applications, such as MJ, Ideogram, and Kling, are built using ComfyUI custom nodes.

### Ideogram:

ComfyUI nodes for using Ideogram APIs, including:
  * Text to Image
    * with color palette utility
  * `img2img`: Remix - Text with Reference Image to Image
  * Describe: Image to Text
  * Upscale: Upscale Image 
  * utils: additional files for Ideogram nodes


### kling:
* A collection of nodes for using [kling.ai] APIs.
  * `kl_txt2image`: text to image
  * `kl_img2img`: text with reference image to image


### MJ:

* A collection of nodes for using [midjourney.ai] APIs.
  * describe: image to text
  * remix: text with reference image to image
  * text2image: text to image
  * split_text: split text from the output of `Describe` node

  
### other_nodes:

* webui_inpaint_node.py: use fastapi to call webui inpaint
* text_node.py: one node for user's input text



## Usage

### Prerequisite: [Ideogram](https://ideogram.ai/) API Key
1. Set the API key in environment variables `IDEOGRAM_KEY` before ComfyUI starts or set it on the node.
2. Use the node as you like
   * If you want to pick a color instead of inputting a color hex value, use `ColorSelect` node.

### Prerequisite: [Kling](https://kling.ai/) API Key
1. Set the `access key` and `secret key` on the behind of the code before ComfyUI starts or set it on the node.

### Prerequisite: [Midjourney](https://midjourney.com/) API Key
1. Set the API key in environment variables `mj_secret` before ComfyUI starts or set it on the node.


## LICENSE
[MIT License](LICENSE)

## Acknowledgements

* `ColorSelect` is modified from [LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)'s `ColorPicker`
  * LayerStyle is licensed under MIT License.
