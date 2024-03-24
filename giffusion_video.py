# import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from pydub import AudioSegment
import torch
from giffusion.generate import run
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
# from giffusion.app import load_pipeline


FR = 10
MELANCHOLY = "Style: melancholy, calm, moody, pensive"
SERENE = "Style: serene, chill, bright, meditative"
TENSE = "Style: tense, energetic, moody, aggresive"
EUPHORIC = "Style: euphoric, energetic, bright, happy"
EMOTIONS = {
    "melancholy": MELANCHOLY,
    "serene": SERENE,
    "tense": TENSE,
    "euphoric": EUPHORIC,
    "default": "",
}
# STYLE = "Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful artwork"
STYLE = "Trending on artstation, digital art scene, matte style, diversity, uplifting, detailed, high quality, high resolution, professional"
# STYLE = "Trending on artstation, anime style, detailed, digital art, studio ghibli"
NEGATIVE = "NSFW, text, writing, watermark, signature, nudity, nude, canvas painting, pencil, photo, furniture, frame, border, harmful bias, depressing, ugly, whitewashed, sexualized, negative stereotype, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, poorly rendered eyes, low resolution, bad crop, bad composition, deformed body parts, blurry image, disfigured, unnatural, visual artifacts, bad anatomy, deformed facial features, low quality, amateur"


def mux_lyrics_emotions(lyrics, emotions, length):
    i = 0
    j = 0
    t = 0
    mux = []
    while i < len(lyrics) or j < len(emotions):
        lyric = None
        if i < len(lyrics):
            lyric_start, lyric_end, lyric = lyrics[i]
        emotion = None
        if j < len(emotions):
            emotion_start, emotion_end, emotion = emotions[j]
        mux_lyric = None
        mux_emotion = None
        if lyric is not None and t >= lyric_start and t < lyric_end:
            mux_lyric = lyric
        if emotion is not None and t >= emotion_start and t < emotion_end:
            mux_emotion = emotion
        mux_start = t
        t = min(filter(lambda x: x > t, [lyric_start, lyric_end, emotion_start, emotion_end]))
        mux_end = t
        mux.append((mux_start, mux_end, mux_lyric, mux_emotion))
        if t >= lyric_end:
            i += 1
        if t >= emotion_end:
            j += 1
    mux_length = round(length * FR)
    if mux_length > t:
        mux.append((t, mux_length, None, None))
    return mux


# def build_prompts(mux, default="Abstract art of music"):
#     prompts = []
#     for start, end, lyric, emotion in mux:
#         if lyric is None:
#             lyric = default
#         if emotion is None:
#             emotion = "default"
#         prompts.append(f"{round(start * FR)}: {STYLE}. {lyric} {EMOTIONS[emotion]}")
#     prompts = "\n".join(prompts)
#     return prompts


def build_prompts(proto_prompts):
    prompts = []
    for start, end, line in proto_prompts:
        # if lyric is None:
        #     lyric = default
        # if emotion is None:
        #     emotion = "default"
        prompts.append(f"{round(start * FR)}: {STYLE}. {line}")
        # prompts.append(f"{round(start * FR)}: {STYLE}. {lyric} {EMOTIONS[emotion]}")
    prompts = "\n".join(prompts)
    return prompts


def get_pipe():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to("cuda")
    return pipe


# def create_video(lyrics, emotions, title, length, seed=42):
def create_video(proto_prompts, seed=42):
    # # Convert lyrics to prompts from llm.py?
    # lyrics_to_prompts(lyrics)
    # print("Generating video...")
    # black = np.zeros((512, 512, 3), dtype=np.uint8)
    # img = Image.fromarray(black, "RGB")
    # img.save("black.png")
    # out_lines = []
    # t = 0

    # lyrics: list of (start, end, lyric) pairs
    # emotions = mer.extract_emotions("audio.mp3") # list of (start, end, emotion) pairs
    # mux = mux_lyrics_emotions(lyrics, emotions, length)
    # print(mux)
    prompts = build_prompts(proto_prompts)
    print(prompts)
    # pipe = load_pipeline("stabilityai/stable-diffusion-2-1-base", "StableDiffusionPipeline", False, None)[0]
    pipe = get_pipe()
    run(pipe=pipe, text_prompt_inputs=prompts, negative_prompt_inputs=NEGATIVE, fps=FR, audio_input="/home/ec2-user/main/ammml_mv/results/audio.mp3", seed=seed, model_name="stable-diffusion-2-1-base")


    # for i, (start, end, line) in enumerate(lyrics):
    #     pipe = load_pipeline("stabilityai/stable-diffusion-2-1-base", "DiffusionPipeline")
    #     if start > t and i == 0:
    #         dur = start - t
    #         t = start
    #         # In this case, diffuse from a black to first lyric prompt
    #         # Assuming lyrics have been converted to prompts already
    #         negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
    #         style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
    #         line = line + style_comp_prompt
    #         text_prompt_input = "0: black screen\n" + str(int(dur*10)) + ": " + line
            
    #         # get audio clip from start to t
    #         song = AudioSegment.from_mp3("audio.mp3")
    #         clip = song[start*1000:t*1000]
    #         clip.export("temp.mp3", format="mp3")

    #         # generate giffusion output from black to first lyric prompt

            
    #         seed = np.random.randint(np.iinfo(np.int32).max)
    #         run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, audio_input="temp.mp3", seed=seed, model_name="stable-diffusion-2-1-base")
    #     elif start > t:
    #         if do_transition:
    #             dur = start - t
    #             t = start
    #             old_line = lyrics[i-1][2]
    #             # In this case, diffuse from a black to first lyric prompt
    #             # Assuming lyrics have been converted to prompts already
    #             negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
    #             style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
    #             line = line + style_comp_prompt
    #             text_prompt_input = "0: " + old_line + "\n" + str(int(dur*10)) + ": " + line
                
    #             # get audio clip from start to t
    #             song = AudioSegment.from_mp3("audio.mp3")
    #             clip = song[start*1000:t*1000]
    #             clip.export("temp.mp3", format="mp3")

    #             # generate giffusion output from black to first lyric prompt

    #             # need to create proper pipe and generate random seed

    #             # pipe = create_model()
                
    #             # seed = np.random.randint(np.iinfo(np.int32).max)
    #             run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, audio_input="temp.mp3", seed=seed, model_name="stable-diffusion-2-1-base")
    #         else:
    #             dur = start - t
    #             t = start
    #             # old_line = lyrics[i-1][2]
    #             # In this case, diffuse from a black to first lyric prompt
    #             # Assuming lyrics have been converted to prompts already
    #             negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
    #             style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
    #             line = line + style_comp_prompt
                
                
    #             # get audio clip from start to t
    #             song = AudioSegment.from_mp3("audio.mp3")
    #             clip = song[start*1000:t*1000]
    #             clip.export("temp.mp3", format="mp3")

    #             frames = str(int(dur*10))
    #             emotions = get_emotions("temp.mp3", )
    #             text_prompt_input = []
    #             for i in range(str(round(dur*10))):
    #                 text_prompt_input.append(f"{i}: " + line + emotions[i] + "\n")
    #             text_prompt_input = "0: " + old_line + "\n" + str(int(dur*10)) + ": " + line
    #             # need to create proper pipe and generate random seed

    #             # pipe = create_model()
                
    #             # seed = np.random.randint(np.iinfo(np.int32).max)
    #             run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, audio_input="temp.mp3", seed=seed, model_name="stable-diffusion-2-1-base")
    #     dur = end - start
    #     t = end
    #     # get timestamps for positive and negative music features (some quantity) from start to end
    #     pos_feature_timestamps = get_pos_feature_timestamps( "audio.mp3", start, end) 
    #     neg_feature_timestamps = get_neg_feature_timestamps( "audio.mp3", start, end)

    #     # generate curve interpolation parameters from positive and negative feature timestamps
    #     curve_str = "0:(0.0),"
    #     param_val = 0.0
    #     for j in range(start*10, end*10):
            
    #         if j in pos_feature_timestamps:
    #             if(param_val < 1.0):
    #                 param_val += 0.1
    #             # interpolate curve to positive feature
    #             curve_str += str(j-start*10) + ":(" + str(param_val) + "),"
    #         elif j in neg_feature_timestamps:
    #             if(param_val > 0.0):
    #                 param_val -= 0.1
    #             # interpolate curve to negative feature
    #             curve_str += str(j-start*10) + ":(" + str(param_val) + "),"
    #     curve_str += str(end*10-start*10) + ":(" + str(0.0) + ")"

    #     # generate giffusion output starting from lyric prompt to music feature inspired lyric prompt and back
    #     negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
    #     style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
    #     line = line + style_comp_prompt
    #     MUSIC_FEATURE_PROMPT = "Large" # COME UP WITH SOMETHING BETTER
    #     text_prompt_input = "0: " + line + "\n" + str(int(dur*10)) + ": " + MUSIC_FEATURE_PROMPT + line

    #     # need to use proper pipe and generate random seed
    #     # seed = np.random.randint(np.iinfo(np.int32).max)
    #     run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, seed=seed, model_name="stable-diffusion-2-1-base", interpolation_type="curve", interpolation_args=curve_str)
    #     # out_lines.append(f"file out/img_{i:03d}.png\n")
    #     # out_lines.append(f"duration {dur}\n")
    # start, end = no_lyrics[-1]
    # if end > t:
    #     dur = end - t
    #     t = end
    #     # In this case, diffuse from lyric prompt to black screen
    #     # Assuming lyrics have been converted to prompts already
    #     negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
    #     style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
    #     line = line + style_comp_prompt
    #     text_prompt_input = "0: " + line + "\n" + str(int(dur*10)) + ": black screen"            
    #     # get audio clip from start to t
    #     song = AudioSegment.from_mp3("audio.mp3")
    #     clip = song[t*1000:end*1000]
    #     clip.export("temp.mp3", format="mp3")

    #     # generate giffusion output from lyric prompt to black screen

    #     # need to use proper pipe and generate random seed

    #     # pipe = create_model()
        
    #     # seed = np.random.randint(np.iinfo(np.int32).max)
    #     run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, audio_input="temp.mp3", seed=seed, model_name="stable-diffusion-2-1-base")
