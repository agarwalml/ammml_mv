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


EMOTIONS = {
    "melancholy": ("low", "low"),
    "serene": ("low", "high"),
    "tense": ("high", "low"),
    "euphoric": ("high", "high"),
    "default": ("neutral", "neutral"),
}
FR = 10


def mux_lyrics_emotions(lyrics, emotions, length, default="Abstract art of music"):
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
        mux_lyric = default
        mux_emotion = EMOTIONS["default"]
        if lyric is not None and t >= lyric_start and t < lyric_end:
            mux_lyric = lyric
        if emotion is not None and t >= emotion_start and t < emotion_end:
            mux_emotion = EMOTIONS[emotion]
        mux_start = t
        t = min(filter(lambda x: x > t, [lyric_start, lyric_end, emotion_start, emotion_end]))
        mux_end = t
        mux.append((mux_start, mux_end, mux_lyric, mux_emotion))
        if t >= lyric_end:
            i += 1
        if t >= emotion_end:
            j += 1
    mux_length = round(length * FR)
    if mux_length > t + 1:
        mux_start = t
        mux_end = mux_length
        mux.append((mux_start, mux_end - 1, default, EMOTIONS["default"]))
    mux.append((mux_end - 1, mux_end, default, EMOTIONS["default"]))
    return mux


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
    if url:
        print("Downloading...")
        transcriber.get_audio(url)
    # transcriber.get_result()
    print("Transcribing...")
    lyrics, no_lyrics, length = transcriber.create_timestamps()
    pcm = music.read_pcm("audio.mp3")
    # if params["beat_alignment"]:
    print("Beat aligning...")
    lyrics = music.beat_alignment(lyrics, pcm)
    # if params["llm_prompting"]:
    print("Extracting emotions...")
    if debug:
        csv_path = sys.argv[6]
        emotions = extract_emotions_debug(csv_path, model_path)
    else:
        emotions = extract_emotions("audio.mp3", model_path)
    mux = mux_lyrics_emotions(lyrics, emotions, length, default=title)
    print(mux)
    print("LLM prompting...")
    proto_prompts = llm.mux_to_prompts(mux)
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
    print("Generating video...")
    giffusion_video.create_video(proto_prompts, seed)
    print("Done!")


if __name__ == "__main__":
    main()
