import os
import sys

import diffusion
import llm
import music
import transcriber
import video


def main():
    modstring = sys.argv[1]
    url = sys.argv[2]
    out_dir = sys.argv[3]
    params = {}
    params["seed_consistency"] = bool(int(modstring[0]))
    params["style_prompt"] = bool(int(modstring[1]))
    params["negative_prompt"] = bool(int(modstring[2]))
    params["beat_alignment"] = bool(int(modstring[3]))
    params["llm_prompting"] = bool(int(modstring[4]))
    params["music_features"] = bool(int(modstring[5]))
    # params["lyric_features"] = bool(modstring[6])
    print("Running ablation:", params)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    transcriber.get_audio(url)
    # transcriber.get_result()
    lyrics, no_lyrics = transcriber.create_timestamps()
    pcm = music.read_pcm("audio.mp3")
    if params["beat_alignment"]:
        lyrics = music.beat_alignment(lyrics, pcm)
    if params["llm_prompting"]:
        lyrics = llm.lyrics_to_prompts(lyrics)
    # features = None
    # if params["music_features"]:
        # features = music.music_features(pcm)
    os.makedirs("out", exist_ok=True)
    model = diffusion.create_model()
    lines = [line for start, end, line in lyrics]
    # lines = list(map(lambda x: x[0][2] + ", " + x[1], zip(lyrics, features)))
    diffusion.lyrics_to_images(lines, model, "out", params)
    # paths = list(sorted(glob.glob("out/img_*.png")))
    video.create_video(lyrics, no_lyrics)
    print("Done!")


if __name__ == "__main__":
    main()
