import os
import sys

# import diffusion
import llm
import music
import transcriber
# import video
# import giffusion_video
from MultimodalMusicEmotion.utils import extract_emotions, extract_emotions_debug


debug = False


EMOTIONS = {
    "melancholy": ("low", "low"),
    "serene": ("low", "high"),
    "tense": ("high", "low"),
    "euphoric": ("high", "high"),
    "default": ("neutral", "neutral"),
}
FR = 10


def mux_lyrics_emotions(lyrics, emotions, length, default="Abstract art of music"):
    print("Lyrics: ")
    print(lyrics)

    emotions_valence = []
    for t1, t2, emotion in emotions:
        if emotion is not None:
            emotions_valence.append((t1, t2, EMOTIONS[emotion]))
        else:
            emotions_valence.append((t1, t2, EMOTIONS["default"]))

    if emotions_valence[-1][1] < length:
        emotions_valence.append((emotions_valence[-1][1], length, EMOTIONS["default"]))

    print("Emotions Valence: ")
    print(emotions_valence)

    # make lyrics a contiguous list
    if lyrics[0][0] > 0:
        lyrics.insert(0, (0, lyrics[0][0], default))
    # for any gap between lyrics, fill with default
    for i in range(1, len(lyrics)):
        if lyrics[i][0] > lyrics[i-1][1]:
            lyrics.insert(i, (lyrics[i-1][1], lyrics[i][0], default))
    if lyrics[-1][1] < length:
        lyrics.append((lyrics[-1][1], length, default))
    
    print("Lyrics: ")
    print(lyrics)

    print("Final Length: ", length)
    
    # Make lyrics a continuo

    mux = []
    i = 0
    j = 0
    prev_end = 0

    while i < len(lyrics) or j < len(emotions_valence):
        if i < len(lyrics):
            lyric_start, lyric_end, lyric = lyrics[i]
        else:
            lyric_start = float('inf')
            lyric_end = float('inf')

        if j < len(emotions_valence):
            emotion_start, emotion_end, emotion_val = emotions_valence[j]
        else:
            emotion_start = float('inf')
            emotion_end = float('inf')

        # Find next interval start
        next_start = min(lyric_start, emotion_start, length)

        # Fill in gap with default values
        if next_start > prev_end:
            mux.append((prev_end, next_start, default, default))
            # print(mux)

        # Find intersection interval
        intersection_start = max(lyric_start, emotion_start)
        intersection_end = min(lyric_end, emotion_end)

        if intersection_start <= intersection_end:
            # There is an intersection
            mux.append((intersection_start, intersection_end, lyric, emotion_val))
            # print(mux)

        # Update pointers
        if lyric_end <= emotion_end:
            i += 1
        if lyric_end >= emotion_end:
            j += 1

        prev_end = max(lyric_end, emotion_end)
    
    while i < len(lyrics):
        lyric_start, lyric_end, lyric = lyrics[i]
        mux.append((lyric_start, lyric_end, lyric, EMOTIONS["default"]))
        i += 1

    while j < len(emotions_valence):
        emotion_start, emotion_end, emotion_val = emotions_valence[j]
        mux.append((emotion_start, emotion_end, default, emotion_val))
        j += 1

    # default till end
    if prev_end < length:
        mux.append((prev_end, length, default, EMOTIONS["default"]))

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
    # os.makedirs(out_dir, exist_ok=True)
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
    # proto_prompts = llm.mux_to_prompts(mux)
    proto_prompts = llm.mux_to_narrative_prompts(mux)
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
    
    # proto_prompts = [(0, 0.18575963718820862, ' A solitary flower blooms in the middle of a barren landscape'), (0.18575963718820862, 0.5, ' Two figures stand side by side in a snow-covered field, breath visible in the cold air'), (0.5, 1.0, ' The same figures, now sitting apart on a frozen bench, their breath less visible, a silent tension between them'), (1.0, 3.0, " Amidst a snowstorm, one figure reaches for the other's hand, but the gesture is not reciprocated"), (3.0, 11.3081179138322, ' The figures sit back to back under a thin veil of snow, their distance growing colder'), (11.3081179138322, 15.116190476190477, ' A dream catcher hangs frozen from a branch, symbolizing unattainable dreams'), (15.116190476190477, 18.94748299319728, ' The once harmonious pair is now visibly divided, an invisible force tearing them apart'), (18.94748299319728, 19.71374149659864, ' Wilted flowers lay on a snowy patch, symbolizing the fading of their love'), (19.71374149659864, 23.26639455782313, ' A small wooden house is engulfed in flames, watched from a distance by a solitary figure'), (23.26639455782313, 26.5, ' A figure looks back at a burning house, their expression a mix of reluctance and sorrow'), (26.5, 26.563628117913833, ' The same figure walks away with increased pace, fighting the urge to look back, their face contorted with emotion'), (26.563628117913833, 28.328344671201815, ' The figure pauses, the truth weighing heavily on their lips'), (28.328344671201815, 32.64725623582766, " Tears well up in the figure's eyes, but a moment of realization halts their fall"), (32.64725623582766, 36.19990929705215, ' A hand holding fresh flowers, a personal act of kindness and self-reassurance'), (36.19990929705215, 40.03120181405896, ' Someone etches their own name into the beach sand, claiming their story'), (40.03120181405896, 44.0, ' A figure, alone, engaged in deep conversation with their reflection in a mirror'), (44.0, 44.6055328798186, ' The same scene, but the figure now laughs freely, enjoying the solitude'), (44.6055328798186, 47.64734693877551, ' Words of self-love and affirmations are spoken into the mirror, incomprehensible but empowering'), (47.64734693877551, 51.96625850340136, ' A lone dancer twirls under a cascade of lights, their joy unabated by solitude'), (51.96625850340136, 56.540589569161, ' A single figure intertwines their fingers, finding comfort in their own embrace'), (56.540589569161, 58.32852607709751, ' A radiant smile reflects a newfound self-acceptance and love'), (58.32852607709751, 58.32852607709751, ' The smile persists, stronger, fueled by self-love and independence'), (58.32852607709751, 58.32852607709751, ' The figure now stands tall and confident, surrounded by an aura of self-appreciation'), (58.32852607709751, 61.881179138322, ' A declaration of self-love, more profound than any external affection'), (61.881179138322, 62.902857142857144, ' A reminder that external validation pales in comparison to inner strength and love'), (62.902857142857144, 65.45705215419501, ' With a confident stride, the figure embodies self-love, untouchable and radiant'), (65.45705215419501, 66.96634920634921, ' The assertion of self-worth repeats, echoing into the surroundings'), (66.96634920634921, 69.0097052154195, " The figure's journey of self-love reaches a peak, radiant and full of life"), (69.0097052154195, 72.56235827664399, ' Fingers painting nails cherry red, a vibrant act of self-care'), (72.56235827664399, 76.39365079365079, ' The red nails contrast with leftover roses, a symbol of moving on'), (76.39365079365079, 80.20172335600907, ' A determined face in the mirror, no traces of regret or sorrow'), (80.20172335600907, 84.52063492063492, ' All memories of past words are discarded, leaving room for self-embrace'), (84.52063492063492, 87.56244897959184, " A fleeting look back, but the figure's step is light and joyful"), (87.56244897959184, 88.3287074829932, ' The hesitation fades, replaced by an assured forward motion'), (88.3287074829932, 89.35038548752834, ' The choice not to engage in conflict, opting for peace instead'), (89.35038548752834, 93.66929705215419, ' Tears transform into a smile, a turning point from sorrow to strength'), (93.66929705215419, 97.47736961451247, ' A bouquet of flowers bought for oneself, symbolizing self-love and independence'), (97.47736961451247, 101.54086167800453, ' A solitary name in the sand stands proud against the crashing waves'), (101.54086167800453, 105.34893424036281, ' Hours spent in meaningful solitude, embracing the comfort of one’s own company'), (105.34893424036281, 108.66938775510204, ' Whispered self-affirmations fill the room, a language of love understood by one'), (108.66938775510204, 112.7328798185941, ' The rhythm of self-embrace as feet move in harmony to the music'), (112.7328798185941, 117.56263038548752, " A reassuring grip on one's own hand, a symbol of self-support"), (117.56263038548752, 118.32888888888888, " This gesture of holding one's hand is repeated, signifying independence and strength"), (118.32888888888888, 119.35056689342403, ' The same affirmations, now a mantra, envelop the figure in warmth and understanding'), (119.35056689342403, 122.39238095238095, ' The ultimate realization that self-love surpasses any love once sought from another'), (122.39238095238095, 123.92489795918367, ' A bold defiance against the notion that others could love us more than we love ourselves'), (123.92489795918367, 130.2871655328798, ' Embracing self-love, the figure glows with happiness and confidence'), (130.2871655328798, 131.54104308390023, ' The repetition of this belief reinforces the figure’s transformation and growth'), (131.54104308390023, 136.64943310657597, ' Surrounded by an aura of self-love, the figure is content, fulfilled, and radiant'), (136.64943310657597, 138.66956916099772, ' A final, affirming nod to oneself, a promise of continued self-love and care'), (138.66956916099772, 140.20208616780045, ' A glance back not in sorrow but in acceptance of moving forward'), (140.20208616780045, 142.24544217687074, ' Choosing peace over conflict, embracing the path of healing and self-love'), (142.24544217687074, 145.28725623582767, ' A brief moment of vulnerability turns into an epiphany of self-worth'), (145.28725623582767, 147.30739229024942, ' The purchase of flowers becomes an emblem of self-appreciation and care'), (147.30739229024942, 161.54122448979592, ' The act of self-love encompassing simple joys and personal victories'), (161.54122448979592, 179.6063492063492, ' A declaration of independence, dancing alone, hand in hand with oneself, a testament to self-love'), (179.6063492063492, 183.41442176870748, ' A final affirmation of the ability to love oneself deeply and utterly'), (183.41442176870748, 201.5, ' Flowers bloom vividly, now symbols of vibrant self-love and renewal'), (201.5, 201.576, ' The narrative closes with a single flower, signifying the enduring presence of hope and growth amidst change')]


    print("Generating video...")
    giffusion_video.create_video(proto_prompts, seed)
    print("Done!")


if __name__ == "__main__":
    main()
