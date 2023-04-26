import os    
from giffusion.generate import run
from giffusion.utils import  (
    get_audio_key_frame_information,
    get_video_frame_information,
    load_video_frames,
    to_pil_image,
)
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

def lyrics_to_video(lines, pipe, out_dir, params, lyrics, no_lyrics):
    print("Creating video from lyrics")
    if params["seed_consistency"]:
        global seed
        seed = np.random.randint(np.iinfo(np.int32).max)
        print("Seed:", seed)
    t = 0
    for i, line in tqdm.tqdm(list(enumerate(lines))):
       
        if start > t:
            dur = start - t
            t = start
            text_prompt_input = "0: black screen\n" + str(dur) + ": " + line
            negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"

            run(pipe, text_prompt_input, negative_prompt, fps = 10, seed=seed, model_name="stable-diffusion-2-1-base")
        dur = end - start
        t = end
        out_lines.append(f"file out/img_{i:03d}.png\n")
        out_lines.append(f"duration {dur}\n")
    start, end = no_lyrics[-1]
    if end > t:
        dur = end - t
        t = end
        out_lines.append(f"file black.png\n")
        out_lines.append(f"duration {dur}\n")
        out_lines.append(f"file black.png\n")
    else:
        out_lines.append(f"file out/img_{i:03d}.png\n")
