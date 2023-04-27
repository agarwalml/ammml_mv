# import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from pydub import AudioSegment
from giffusion.generate import run
from giffusion.app import load_pipeline



def create_video(lyrics, no_lyrics):
    # # Convert lyrics to prompts from llm.py?
    # lyrics_to_prompts(lyrics)
    print("Creating video")
    # black = np.zeros((512, 512, 3), dtype=np.uint8)
    # img = Image.fromarray(black, "RGB")
    # img.save("black.png")
    out_lines = []
    t = 0
    for i, (start, end, line) in enumerate(lyrics):
        pipe = load_pipeline("stabilityai/stable-diffusion-2-1-base", "DiffusionPipeline")
        if start > t and i == 0:
            dur = start - t
            t = start
            # In this case, diffuse from a black to first lyric prompt
            # Assuming lyrics have been converted to prompts already
            negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
            style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
            line = line + style_comp_prompt
            text_prompt_input = "0: black screen\n" + str(int(dur*10)) + ": " + line
            
            # get audio clip from start to t
            song = AudioSegment.from_mp3("audio.mp3")
            clip = song[start*1000:t*1000]
            clip.export("temp.mp3", format="mp3")

            # generate giffusion output from black to first lyric prompt

            
            seed = np.random.randint(np.iinfo(np.int32).max)
            run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, audio_input="temp.mp3", seed=seed, model_name="stable-diffusion-2-1-base")
        elif start > t:
            dur = start - t
            t = start
            old_line = lyrics[i-1][2]
            # In this case, diffuse from a black to first lyric prompt
            # Assuming lyrics have been converted to prompts already
            negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
            style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
            line = line + style_comp_prompt
            text_prompt_input = "0: " + old_line + "\n" + str(int(dur*10)) + ": " + line
            
            # get audio clip from start to t
            song = AudioSegment.from_mp3("audio.mp3")
            clip = song[start*1000:t*1000]
            clip.export("temp.mp3", format="mp3")

            # generate giffusion output from black to first lyric prompt

            # need to create proper pipe and generate random seed

            # pipe = create_model()
            
            # seed = np.random.randint(np.iinfo(np.int32).max)
            run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, audio_input="temp.mp3", seed=seed, model_name="stable-diffusion-2-1-base")

        dur = end - start
        t = end
        # get timestamps for positive and negative music features (some quantity) from start to end
        pos_feature_timestamps = get_pos_feature_timestamps( "audio.mp3", start, end) 
        neg_feature_timestamps = get_neg_feature_timestamps( "audio.mp3", start, end)

        # generate curve interpolation parameters from positive and negative feature timestamps
        curve_str = "0:(0.0),"
        param_val = 0.0
        for j in range(start*10, end*10):
            
            if j in pos_feature_timestamps:
                if(param_val < 1.0):
                    param_val += 0.1
                # interpolate curve to positive feature
                curve_str += str(j-start*10) + ":(" + str(param_val) + "),"
            elif j in neg_feature_timestamps:
                if(param_val > 0.0):
                    param_val -= 0.1
                # interpolate curve to negative feature
                curve_str += str(j-start*10) + ":(" + str(param_val) + "),"
        curve_str += str(end*10-start*10) + ":(" + str(0.0) + ")"

        # generate giffusion output starting from lyric prompt to music feature inspired lyric prompt and back
        negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
        style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
        line = line + style_comp_prompt
        MUSIC_FEATURE_PROMPT = "Large" # COME UP WITH SOMETHING BETTER
        text_prompt_input = "0: " + line + "\n" + str(int(dur*10)) + ": " + MUSIC_FEATURE_PROMPT + line

        # need to use proper pipe and generate random seed
        # seed = np.random.randint(np.iinfo(np.int32).max)
        run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, seed=seed, model_name="stable-diffusion-2-1-base", interpolation_type="curve", interpolation_args=curve_str)
        # out_lines.append(f"file out/img_{i:03d}.png\n")
        # out_lines.append(f"duration {dur}\n")
    start, end = no_lyrics[-1]
    if end > t:
        dur = end - t
        t = end
        # In this case, diffuse from lyric prompt to black screen
        # Assuming lyrics have been converted to prompts already
        negative_prompt = "nsfw, text, watermark, ugly, poorly rendered face, poorly drawn face, poor facial details, poorly drawn hand, poorly rendered hands, low resolution, image cut off, bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features, low quality"
        style_comp_prompt = ". Trending on artstation, matte, elegant, illustration, detailed, digital painting, epic composition, beautiful art"
        line = line + style_comp_prompt
        text_prompt_input = "0: " + line + "\n" + str(int(dur*10)) + ": black screen"            
        # get audio clip from start to t
        song = AudioSegment.from_mp3("audio.mp3")
        clip = song[t*1000:end*1000]
        clip.export("temp.mp3", format="mp3")

        # generate giffusion output from lyric prompt to black screen

        # need to use proper pipe and generate random seed

        # pipe = create_model()
        
        # seed = np.random.randint(np.iinfo(np.int32).max)
        run(pipe=pipe, text_prompt_inputs=text_prompt_input, negative_prompt_inputs=negative_prompt, fps=10, audio_input="temp.mp3", seed=seed, model_name="stable-diffusion-2-1-base")
