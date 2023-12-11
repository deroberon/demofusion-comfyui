import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

my_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(my_dir)
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
sys.path.remove(my_dir)


class Demofusion:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
                "negative": ("STRING", {
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": ""
                }),
                "width": ("INT", {
                    "default": 2048, 
                    "min": 2048, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "height": ("INT", {
                    "default": 2048, 
                    "min": 2048, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "inference_steps": ("INT", {
                    "default": 40, 
                    "min": 1, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
                "seed": ("INT", {
                    "default": 522, 
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "tests"

    def execute(self, positive, negative, width, height, inference_steps, cfg, seed):
        model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        generator = torch.Generator(device='cuda')
        generator = generator.manual_seed(seed)

        images = pipe(str(positive), negative_prompt=str(negative),
              height=height, width=width, view_batch_size=4, stride=64,
              num_inference_steps=inference_steps, guidance_scale=cfg,
              cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
              multi_decoder=True, show_image=False#, lowvram=True
             )
        image=images[len(images)-1]
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Demofusion": Demofusion
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Demofusion": "Demofusion"
}