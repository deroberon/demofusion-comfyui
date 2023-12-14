import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import comfy.model_management
import comfy.sample
import logging as logger

my_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(my_dir)
from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline
sys.path.remove(my_dir)

import folder_paths

class Demofusion:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": ("STRING", {
                    "multiline": False,
                    "default": "stabilityai/stable-diffusion-xl-base-1.0"
                }),
                "positive": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "negative": ("STRING", {
                    "multiline": True, 
                    "default": ""
                }),
                "width": ("INT", {
                    "default": 2048, 
                    "min": 2048, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" 
                }),
                "height": ("INT", {
                    "default": 2048, 
                    "min": 2048, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" 
                }),
                "inference_steps": ("INT", {
                    "default": 40, 
                    "min": 1, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" 
                }),
                "cfg": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "round": 0.001, 
                    "display": "number"}),
                "seed": ("INT", {
                    "default": 522, 
                    "display": "number" 
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "tests"

    def execute(self, ckpt_name, positive, negative, width, height, inference_steps, cfg, seed):
        pipe = DemoFusionSDXLPipeline.from_pretrained(ckpt_name, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        generator = torch.Generator(device='cuda')
        generator = generator.manual_seed(seed)

        images = pipe(str(positive), negative_prompt=str(negative),
              height=height, width=width, view_batch_size=4, stride=64,
              num_inference_steps=inference_steps, guidance_scale=cfg,
              cosine_scale_1=3, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
              multi_decoder=True, show_image=False
             )
        image=images[len(images)-1]
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)


class BatchUnsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "end_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "step_increment": ("INT", {"default": 1, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "normalize": (["disable", "enable"], ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    }}
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    FUNCTION = "batch_unsampler"

    CATEGORY = "tests"
        
    def batch_unsampler(self, model, cfg, sampler_name, steps, end_at_step, step_increment, scheduler, normalize, positive, negative, latent_image):
        normalize = normalize == "enable"
        device = comfy.model_management.get_torch_device()
        latent = latent_image
        latent_image = latent["samples"]

        batch_of_latents = []

        end_at_step = min(end_at_step, steps-1)
        end_at_step = steps - end_at_step
        
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sample.prepare_mask(latent["noise_mask"], noise, device)

        real_model = None
        real_model = model.model

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        positive = comfy.sample.convert_cond(positive)
        negative = comfy.sample.convert_cond(negative)

        models, inference_memory = comfy.sample.get_additional_models(positive, negative, model.model_dtype())
        
        comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sigmas = sampler.sigmas.flip(0) + 0.0001

        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, x0, x, total_steps):
            logger.warning(f'batch_unsampler: step {step} of {total_steps}')
            batch_of_latents.append(x)
            pbar.update_absolute(step + 1, total_steps)

        #for i in range(0, end_at_step + 1, step_increment):
        # The range() call makes this loop run from step=0
        # to step=(end_at_step - 1). Since the sigmas are reversed
        # above, we are really "unsampling" (adding noise rather than
        # removing noise). Optionally, we step by more than
        # one step at a time using step_increment.

        # Each loop, we sample from 0 all the way to the current step.
        # I'd like to be able to sample just from the current step to the
        # next step, but it doesn't seem to work.
        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, start_step=0, last_step=end_at_step, callback=callback)

        if normalize:
            #technically doesn't normalize because unsampling is not guaranteed to end at a std given by the schedule
            samples -= samples.mean()
            samples /= samples.std()

        #batch_of_latents.append(samples)
        # This happens now in the callback above.

        # samples = samples.cpu()
        # Not sure whether the above is needed... I guess I'll find out.
        
        comfy.sample.cleanup_additional_models(models)

        if len(batch_of_latents) > 0:
            # Concatenate the latents into a batch and do it the Comfy
            # way by jamming the batch into a dictionary as "samples".
            batch_of_latents = torch.cat(batch_of_latents)
            batch_of_latents = {'samples': batch_of_latents}
        else:
            # If no latents were unsampled then just return the
            # input latent.
            batch_of_latents = latent_image

        return (batch_of_latents,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Demofusion": Demofusion,
    "Batch Unsampler": BatchUnsampler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Demofusion": "Demofusion",
    "Batch Unsampler": "Batch Unsampler"
}