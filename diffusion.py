import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import tqdm

seed = 42

def create_model():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe


def prompt_model(prompt, pipe):
    negative_prompt = "nsfw, text, poorly Rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, Images cut out at the top, left, right, bottom. bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features"
    style_comp_prompt = "Surrealism, trending on artstation, matte, elegant, illustration, digital paint, epic composition, beautiful, the most beautiful image ever seen,"
    image = pipe(prompt=(prompt + style_comp_prompt), negative_prompt=negative_prompt, seed=seed).images[0]
    return image


def lyrics_to_images(lines, pipe, out_dir):
    print("Creating images from lyrics")
    for i, line in tqdm.tqdm(list(enumerate(lines))):
        image = prompt_model(line, pipe)
        image.save(os.path.join(out_dir, f"img_{i:03d}.png"))
