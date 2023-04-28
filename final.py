import os
import sys

# import diffusion
import llm
import music
import transcriber
# import video
import giffusion_video
from MultimodalMusicEmotion.utils import extract_emotions, extract_emotions_debug


debug = True


def main():
    # modstring = sys.argv[1]
    title = sys.argv[1]
    url = sys.argv[2]
    out_dir = sys.argv[3]
    model_path = sys.argv[4]
    seed = int(sys.argv[5])
    print("Automatic Multimodal Music Video Generation")
    # params = {}
    # params["seed_consistency"] = bool(int(modstring[0]))
    # params["style_prompt"] = bool(int(modstring[1]))
    # params["negative_prompt"] = bool(int(modstring[2]))
    # params["beat_alignment"] = bool(int(modstring[3]))
    # params["llm_prompting"] = bool(int(modstring[4]))
    # params["music_features"] = bool(int(modstring[5]))
    # params["lyric_features"] = bool(modstring[6])
    # print("Running ablation:", params)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    print("Downloading...")
    transcriber.get_audio(url)
    # transcriber.get_result()
    print("Transcribing...")
    lyrics, no_lyrics = transcriber.create_timestamps()
    pcm = music.read_pcm("audio.mp3")
    # if params["beat_alignment"]:
    print("Beat aligning...")
    lyrics = music.beat_alignment(lyrics, pcm)
    # if params["llm_prompting"]:
    print("LLM prompting...")
    lyrics = llm.lyrics_to_prompts(lyrics)
    # features = None
    # if params["music_features"]:
        # features = music.music_features(pcm)
    # os.makedirs("out", exist_ok=True)
    # model = diffusion.create_model()
    # lines = [line for start, end, line in lyrics]
    # lines = list(map(lambda x: x[0][2] + ", " + x[1], zip(lyrics, features)))
    # diffusion.lyrics_to_images(lines, model, "out", params)
    # paths = list(sorted(glob.glob("out/img_*.png")))
    # video.create_video(lyrics, no_lyrics)
    print("Extracting emotions...")
    if debug:
        csv_path = sys.argv[6]
        emotions = extract_emotions_debug(csv_path, model_path)
    else:
        emotions = extract_emotions("audio.mp3", model_path)
    print("Generating video...")
    giffusion_video.create_video(lyrics, emotions, title, seed)
    print("Done!")


if __name__ == "__main__":
    main()
