import os    
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import numpy as np
import torch
import tqdm


seed = None


def create_model():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe


def prompt_model(prompt, pipe, pos=False, neg=False):
    negative_prompt = None
    if neg:
        negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
    style_comp_prompt = ""
    if pos:
        style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    image = pipe(prompt=(prompt + style_comp_prompt), negative_prompt=negative_prompt).images[0]
    return image


def lyrics_to_images(lines, pipe, out_dir, params):
    print("Creating images from lyrics")
    if params["seed_consistency"]:
        global seed
        seed = np.random.randint(np.iinfo(np.int32).max)
        print("Seed:", seed)
    for i, line in tqdm.tqdm(list(enumerate(lines))):
        image = prompt_model(line, pipe, params["style_prompt"], params["negative_prompt"])
        image.save(os.path.join(out_dir, f"img_{i:03d}.png"))
