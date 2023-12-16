import math
import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import comfy.model_management
import comfy.sample
import comfy.utils
import comfy.samplers
import logging as logger

my_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(my_dir)
## Have to change the original name to contain the word "StableDiffusion" because of:
## https://github.com/huggingface/diffusers/blob/2d94c7838e273c40920ffd6d24d724357add7f2d/src/diffusers/loaders/single_file.py#L207C15-L207C30
from pipeline_demofusion_sdxl import DemoFusionSDXLStableDiffusionPipeline
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
        pipe = DemoFusionSDXLStableDiffusionPipeline.from_pretrained(ckpt_name, torch_dtype=torch.float16)
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

class DemofusionFromSingleFile:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_name": ("STRING", {
                    "multiline": False,
                    "default": "https://huggingface.co/Lykon/DreamShaper/blob/main/DreamShaperXL1.0Alpha2_fixedVae_half_00001_.safetensors"
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

    def execute(self, file_name, positive, negative, width, height, inference_steps, cfg, seed):
        pipe = DemoFusionSDXLStableDiffusionPipeline.from_single_file(file_name, torch_dtype=torch.float16, use_safetensors=True)
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


def generate_noised_latents(x, sigmas):
    """
    Generate all noised latents for a given initial latent image and sigmas in parallel.

    :param x: Original latent image as a PyTorch tensor.
    :param sigmas: Array of sigma values for each timestep as a PyTorch tensor.
    :return: A tensor containing all noised latents for each timestep.
    """
    # Ensure that x and sigmas are on the same device (e.g., CPU or CUDA)
    device = x.device
    sigmas = sigmas[1:].to(device) # ignore the first sigma
    batch_size = x.shape[0]
    num_sigmas = len(sigmas)

    # Expand x and sigmas to match each other in the first dimension
    # x_expanded shape will be:
    # [batch_size * num_sigmas, channels, height, width]
    x_expanded = x.repeat(num_sigmas, 1, 1, 1)
    sigmas_expanded = sigmas.repeat_interleave(batch_size)

    logger.warning(f"sigmas: {sigmas.view(-1)}")

    # Create a noise tensor with the same shape as x_expanded
    noise = torch.randn_like(x_expanded)

    logger.warning(f"noise: {noise.shape}")
    logger.warning(f"x:     {x.shape}")

    # Multiply noise by sigmas, reshaped for broadcasting
    noised_latents = x_expanded + noise * sigmas_expanded.view(-1, 1, 1, 1)

    logger.warning(f"noised_latents: {x.shape}")

    return noised_latents


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
    FUNCTION = "unsampler"

    CATEGORY = "tests"

    
    def unsampler(self, model, cfg, sampler_name, steps, end_at_step, step_increment, scheduler, normalize, positive, negative, latent_image):
        """
        Generate a batch of latents representing each z[i] in the
        progressively noised sequence of latents stemming from the
        source latent_image, using the model's noising schedule (sigma)
        in reverse and applying normal noise at each step in the manner
        prescribed by the original latent diffusion paper.
        """
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

        real_model = model.model

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        positive = comfy.sample.convert_cond(positive)
        negative = comfy.sample.convert_cond(negative)

        models, inference_memory = comfy.sample.get_additional_models(positive, negative, model.model_dtype())
        
        comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)

        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        # Flip the sigmas (the sampling schedule) in reverse so that the sampler
        # will instead "unsample" the latent, adding noise rather than
        # removing noise. I think we add a small epsilon value to prevent
        # division by zero, but I'm not really sure. I got this code from
        # BlenderNeko's noise node.
        sigmas = sampler.sigmas.flip(0)

        z = generate_noised_latents(latent_image, sigmas)

        logger.warning(f"latent_image.shape={latent_image.shape}")
        logger.warning(f"z.shape={z.shape}")

        out = {"samples": z}
        
        comfy.sample.cleanup_additional_models(models)

        return (out,)

def get_blending_schedule(steps, alpha_1=3.0, step_size=1):
    """
    Define a tensor representing the constant c1 from the DemoFusion paper.
    """
    # Create a tensor for t ranging from steps to 0.
    # In the DemoFusion paper, they use the original stable diffusion
    # terminology where de-noising goes from T to 0, but in our case,
    # we go from 0 to steps so we have to reverse the time step
    # indices.
    t = torch.arange(0, steps + step_size, step_size)
    t = t.flip([0])

    # Calculate c1 using the code borrowed from the author's repository.
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (steps - t) / steps))

    c1 = cosine_factor ** alpha_1
    return c1

def batched_ksampler(model, seed, cfg, sampler_name, scheduler, step_increment, positive, negative, latent_image_batch, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, c1=None, alpha_1=3.0):
    z_primes = latent_image_batch["samples"]
    steps = z_primes.shape[0] # batch size

    # Get the blending parameter from the DemoFusion paper.
    if c1 is None:
        c1 = get_blending_schedule(steps, step_size=step_increment, alpha_1=alpha_1)

    # Move the blending schedule tensor to the same device as our
    # latents.
    c1 = c1.to(z_primes.device)

    if disable_noise:
        noise = torch.zeros(z_primes.size(), dtype=z_primes.dtype, layout=z_primes.layout, device="cpu")
    else:
        batch_inds = latent_image_batch["batch_index"] if "batch_index" in latent_image_batch else None
        noise = comfy.sample.prepare_noise(z_primes, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent_image_batch:
        noise_mask = latent_image_batch["noise_mask"]

    pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    # Assume that the first image in the batch is the most noised,
    # representing z_prime[T] from the DemoFusion paper. You may
    # need to reverse the batch of latents to ensure that the first
    # latent in the batch is the most noised.

    # Set up a tensor to receive the samples from the de-noising process.
    # It has the same shape as z_primes and the first element is
    # the first element of z_primes.
    z_out = torch.zeros_like(z_primes)
    z_out = z_out.to(z_primes.device)
    z_out[0] = z_primes[0]
    z_i = z_primes[0].unsqueeze(0)

    # The paper suggests that we de-noise the image step by step, blending
    # in samples from the noised z_hat tensor along the way according to 
    # a blending schedule given by the c1 tensor. Each successively de-noised
    # sample z[i] is given by this formula:
    #
    # z[i] = denoise(z[i-1]) * (1 - c1[i]) + z_prime[i-1] * c1
    #
    # Thus we have to do the following:
    # a) latent_image contains the z_primes for all steps.
    # b) Iterate through all the steps, denoising from z_prime[0] initially.
    # c) Blend the denoised z[i] with z_prime[i] to get a new z[i].

    for i in range(1, steps, step_increment):
        # Grab the i-th z_prime and i-th noise tensor from their batches.
        # Unsqueezing replaces the batch dimension with 1, so it transforms
        # [i, channel, width, height] into [1, channel, width, height]
        z_prime_i = z_primes[i].unsqueeze(0)
        noise_i = noise[i].unsqueeze(0)

        # The paper tells us to de-noise z[i-1] from step
        # T to T-1; in ComfyUI lingo, that means going from
        # step i-1 to step i because we iterate in the reverse
        # direction.
        z_start_step = i - 1
        z_last_step = i
        z_i_minus_1 = z_out[i-1]

        logger.warning(f'Denoising z[{i}] from {i-1} to {i} (steps={steps})')

        # De-noise z[i-1] from step i-1 to step i. Recall that since we
        # start this loop from i=1, z[i-1] is initialized with z_prime[0].
        # After we have the de-noised latent, we will mix it with z_prime[i]
        # according to the paper's cosine blending function. The blended
        # latent will then become z[i] and we will head to the next iteration.
        samples_i = comfy.sample.sample(model, noise_i, steps, cfg, sampler_name, scheduler, positive, negative, z_i_minus_1,
                                    denoise=denoise, disable_noise=disable_noise, start_step=z_start_step, last_step=z_last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

        # Move samples to the same device as z_prime_i so that we can
        # work with them both to mix below.
        samples_i = samples_i.to(z_prime_i.device)

        # Find z_hat (as per the paper) by applying the c1 blending schedule
        # to the samples and the prior z_prime latent. The paper suggests 
        # following this formula, which will mix in a declining fraction of
        # z_prime as de-noising continues:
        #
        # z[i] = denoise(z[i-1]) * (1 - c1[i]) + z_prime[i-1] * c1

        c1_i = c1[i]
        logger.warning(f'mixing in {c1_i} of the z_prime latent and {1 - c1_i} of the samples')
        z_i = c1_i * z_prime_i + (1 - c1_i) * samples_i

        # Append this new latent onto a list; we will concatenate later
        # to create a batch tensor for output from the node.
        logger.warning(f'z_i has shape {z_i.shape}')

        z_out[i] = z_i
        #z_out[i] = samples_i

    logger.warning(f'dimension at output is {z_out.shape}')
    out = latent_image_batch.copy()
    out["samples"] = z_out
    return (out, )


class IterativeMixingKSampler:
    """
    Take a batch of latents, z_prime, and progressively de-noise them
    step by step from z_prime[0] to z_prime[steps], mixing in a weighted
    fraction of z_prime[i] at each step so that de-noising is guided by
    the z_prime latents. This batch sampler assumes that the number of steps
    is just the length of z_prime, so there is no steps parameter. The parameter
    latent_image_batch should come from the Batch Unsampler node.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "step_increment": ("INT", {"default": 1, "min": 1, "max": 10000}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image_batch": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "alpha_1": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0}),
                    "reverse_batch": ("BOOLEAN", {"default": True})
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "test"

    def sample(self, model, seed, cfg, sampler_name, scheduler, step_increment, positive, negative, latent_image_batch, denoise=1.0, alpha_1=3.0, reverse_batch=True):
        # If desired, reverse the latent batch before batch sampling.
        # This is important if the supplied batch is from least-to-most noisy,
        # as this node expects the reverse.
        if reverse_batch:
            latent_image_batch["samples"] =\
                torch.flip(latent_image_batch["samples"], [0])
        return batched_ksampler(model, seed, cfg, sampler_name, scheduler, step_increment, positive, negative, latent_image_batch, denoise=denoise, alpha_1=alpha_1)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Demofusion": Demofusion,
    "Demofusion From Single File" : DemofusionFromSingleFile,
    "Batch Unsampler": BatchUnsampler,
    "Iterative Mixing KSampler": IterativeMixingKSampler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Demofusion": "Demofusion",
    "Demofusion From Single File": "Demofusion From Single File", 
    "Batch Unsampler": "Batch Unsampler",
    "Iterative Mixing KSampler": "Iterative Mixing KSampler"
}