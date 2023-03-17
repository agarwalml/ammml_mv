import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import tqdm


def create_model():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe


def prompt_model(prompt, pipe):
    image = pipe(prompt).images[0]
    return image


def lyrics_to_images(lines, pipe, out_dir):
    for i, line in tqdm.tqdm(list(enumerate(lines))):
        image = prompt_model(line, pipe)
        image.save(os.path.join(out_dir, f"img_{i:03d}.png"))
