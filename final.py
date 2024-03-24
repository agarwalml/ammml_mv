import os
import sys

# import diffusion
import llm
import music
import transcriber
# import video
import giffusion_video
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
    # # params = {}
    # # params["seed_consistency"] = bool(int(modstring[0]))
    # # params["style_prompt"] = bool(int(modstring[1]))
    # # params["negative_prompt"] = bool(int(modstring[2]))
    # # params["beat_alignment"] = bool(int(modstring[3]))
    # # params["llm_prompting"] = bool(int(modstring[4]))
    # # params["music_features"] = bool(int(modstring[5]))
    # # params["lyric_features"] = bool(modstring[6])
    # # print("Running ablation:", params)
    # # os.makedirs(out_dir, exist_ok=True)
    # os.chdir(out_dir)
    # if url:
    #     print("Downloading...")
    #     transcriber.get_audio(url)
    # # transcriber.get_result()
    # print("Transcribing...")
    # lyrics, no_lyrics, length = transcriber.create_timestamps()
    # pcm = music.read_pcm("audio.mp3")
    # # if params["beat_alignment"]:
    # print("Beat aligning...")
    # lyrics = music.beat_alignment(lyrics, pcm)
    # # if params["llm_prompting"]:
    # print("Extracting emotions...")
    # if debug:
    #     csv_path = sys.argv[6]
    #     emotions = extract_emotions_debug(csv_path, model_path)
    # else:
    #     emotions = extract_emotions("audio.mp3", model_path)
    # mux = mux_lyrics_emotions(lyrics, emotions, length, default=title)
    # print(mux)
    # print("LLM prompting...")
    # proto_prompts = llm.mux_to_prompts(mux)
    # # proto_prompts = llm.mux_to_narrative_prompts(mux)
    # # features = None
    # # if params["music_features"]:
    #     # features = music.music_features(pcm)
    # # os.makedirs("out", exist_ok=True)
    # # model = diffusion.create_model()
    # # lines = [line for start, end, line in lyrics]
    # # lines = list(map(lambda x: x[0][2] + ", " + x[1], zip(lyrics, features)))
    # # diffusion.lyrics_to_images(lines, model, "out", params)
    # # paths = list(sorted(glob.glob("out/img_*.png")))
    # # video.create_video(lyrics, no_lyrics)
    proto_prompts = [(0, 0.18575963718820862, 'A serene garden filled with an array of colorful flowers gently swaying in the mild afternoon breeze.'), (0.18575963718820862, 0.5, 'Two serene figures wrapped in warm layers stand side by side, breathing in the crisp, chilly air of a tranquil, sunlit winter morning.'), (0.5, 11.3081179138322, 'Two individuals stand side by side, wrapped in thick blankets, their breath visible in the crisp, chilly air, exuding a sense of companionship and subtle melancholy.'), (11.3081179138322, 15.116190476190477, '"A solitary figure stands at the edge of a misty pier, gazing into the horizon, lost in the intangible dreams that money cannot buy."'), (15.116190476190477, 18.94748299319728, 'A deserted dirt road stretches away under a fading sunset, bordered by withering fields and an old, rusted car abandoned to the side, symbolizing a journey abruptly ended.'), (18.94748299319728, 19.5, "A solitary daisy stands in a sunlit field, its petals drooping slightly as if weary from the day's heat."), (19.5, 19.71374149659864, 'A gentle breeze teases the soft petals of a vibrant bouquet, basking in the warm embrace of the golden afternoon sun.'), (19.71374149659864, 23.26639455782313, 'A serene twilight envelops the remains of a once-loved home, its silhouette standing stark against the calm hues of the evening sky, a testament to loss and the quiet hope of rebirth.'), (23.26639455782313, 26.563628117913833, 'A lone figure stands at the edge of a tranquil lake at sunset, casting a long, wistful shadow as they gaze towards the horizon, embodying a serene farewell.'), (26.563628117913833, 28.328344671201815, 'A solemn figure stares out a rain-streaked window, reflecting on the truth amidst the quiet storm.'), (28.328344671201815, 32.64725623582766, '"Amid a field of lush, vibrant flowers, tears well up in someone\'s eyes, which then sparkle with a sudden glimmer of joy as a fond memory surfaces, halting the sorrow in its tracks."'), (32.64725623582766, 36.19990929705215, "A content individual stands peacefully in a sunlit flower shop, selecting vibrant blooms with a gentle smile, surrounded by the tranquil beauty of nature's colors."), (36.19990929705215, 40.03120181405896, '"A serene hand gracefully scribing a name onto the smooth, sun-kissed beach, with gentle waves whispering promises of eternity in the background."'), (40.03120181405896, 42.0, 'A serene figure sits by a sunlit window, quietly engrossed in a heartfelt conversation with their own reflection, basking in the warmth of self-compassion and understanding.'), (42.0, 44.6055328798186, 'A solitary figure sits under a canopy of twinkling stars, engaged in thoughtful conversation with the universe, their face illuminated by the soft glow of a warm lantern.'), (44.6055328798186, 47.64734693877551, 'A person leans forward, eyes sparkling with curiosity, as they whisper secrets of an ancient world, igniting an air of excitement and wonder in the dim candlelit room.'), (47.64734693877551, 51.96625850340136, 'A joyful woman twirls freely under the vibrant lights of a dance floor, her laughter mingling with the music, embodying liberation and happiness.'), (51.96625850340136, 56.540589569161, 'A solitary figure stands atop a windswept hill at sunset, firmly grasping their own hand in a powerful symbol of self-reliance and inner strength, bathed in the warm embrace of the golden light.'), (56.540589569161, 58.32852607709751, 'A radiant individual stands atop a blooming hill at sunrise, arms spread wide, embracing the world with a look of profound self-love and boundless joy.'), (58.32852607709751, 58.32852607709751, 'A joyful person stands atop a scenic hill at sunset, arms wide open in self-embrace, basking in the warm, glowing light of self-love and freedom.'), (58.32852607709751, 58.32852607709751, '"A radiant person embraces themself in a sunlit meadow, exuding joy and self-love with a beaming smile that lights up the serene landscape."'), (58.32852607709751, 61.881179138322, 'A triumphant athlete outpaces their closest rival, casting a beaming smile backward with sparkling eyes that say, "I\'m better than you can ever imagine."'), (61.881179138322, 62.902857142857144, 'A radiant couple shares a tender, heartfelt embrace under the golden glow of a setting sun, encapsulating a moment of pure, unadulterated affection that renders the phrase "Can\'t love me better" profoundly true.'), (62.902857142857144, 65.45705215419501, 'A radiant individual joyously embracing themselves under a warm, golden sunset, symbolizing self-love and liberation.'), (65.45705215419501, 66.96634920634921, '"A joyful couple shares a tender, impassioned embrace under a cascade of golden sunset light, symbolizing an unbreakable bond of love that no one could surpass."'), (66.96634920634921, 69.0097052154195, 'A radiant individual blissfully embraces themselves in a sunlit field, an expression of serene self-love illuminating their features.'), (69.0097052154195, 72.56235827664399, 'A radiant smile illuminates her face as she extends her hands, anticipating the vibrant cherry red that will soon adorn her perfectly shaped nails.'), (72.56235827664399, 76.39365079365079, 'A couple tenderly reunites, joyously matching the vibrant roses scattered around, symbols of their unbreakable bond and enduring love.'), (76.39365079365079, 80.20172335600907, 'Beneath a vibrant, fiery sunset, an unyielding figure stands at a cliff\'s edge, gazing into the horizon, their face a mask of serene defiance, embodying the spirit of "no remorse, no regret."'), (80.20172335600907, 84.52063492063492, 'A couple stands beneath a vibrant sunset, lost in an embrace, symbolizing a love that transcends memory and words.'), (84.52063492063492, 87.56244897959184, 'Under a golden sunset, a deeply enamored couple shares a tender, lingering embrace on a serene beach, their silhouettes glowing with a radiant warmth, as the world seems to pause in reverence of their profound reluctance to part.'), (87.56244897959184, 88.3287074829932, "Under the golden hues of a setting sun, two silhouetted lovers embrace tenderly on a secluded beach, unwilling to part from each other's arms."), (88.3287074829932, 89.35038548752834, 'Under a radiant sunset, two silhouettes standing close, hands almost touching, share a tense but tender moment of reconciliation, their faces a canvas of mixed regret and relief, surrounded by an aura of peaceful defiance against the backdrop of a world expecting conflict.'), (89.35038548752834, 93.66929705215419, 'Tears welled in their eyes, sparkling with a mix of sorrow and resilience, as they paused, remembering the inner strength that always carried them through.'), (93.66929705215419, 97.47736961451247, 'A jubilant woman beams radiantly, clutching a vibrant bouquet of flowers she’s just chosen for herself, her eyes sparkling with delight amidst a bustling flower market.'), (97.47736961451247, 101.54086167800453, 'A joyful couple laugh under the radiant sunshine as they lovingly etch their intertwined names into the smooth, golden sand, the waves gently teasing the edges of their heartfelt masterpiece.'), (101.54086167800453, 105.34893424036281, 'Sitting alone on a sun-dappled bench amidst vibrant autumn leaves, a thoughtful person is deeply engaged in an animated conversation with themselves, their expressions ranging from smiles of enlightenment to gestures of passionate emphasis.'), (105.34893424036281, 108.66938775510204, '"A curious child gazes upwards, eyes wide with wonder and lips parted in astonishment, as a kaleidoscope of balloons drifts into the azure sky, each one adorned with mystically undecipherable symbols."'), (108.66938775510204, 112.7328798185941, 'A radiant individual lost in the joy of dancing under a cascade of shimmering lights, their face lit up with an infectious, unbridled enthusiasm.'), (112.7328798185941, 117.56263038548752, 'A lone figure stands silhouetted against the crimson hues of sunset, tenderly clasping their own hand in a gesture of self-reliance and tranquil assurance.'), (117.56263038548752, 118.32888888888888, 'Grasping her own sun-kissed hand with a serene smile, she exudes independence against the vibrant backdrop of a blossoming spring meadow.'), (118.32888888888888, 118.83972789115646, 'A curious child, eyes wide with wonder and lips parted in awe, eagerly whispers secrets into the ear of an ancient, smiling tree under the enchanting glow of a golden summer sunset.'), (118.83972789115646, 119.35056689342403, 'A lonely soul gazes into a shattered mirror, searching through the fragmented reflections for the spark of self-love that seems just out of reach, yet the glimmer of hope in their eyes suggests a journey towards healing just beginning.'), (119.35056689342403, 120.37224489795918, 'A lone figure stands at the edge of a barren field, looking away from a dilapidated house, under a stormy sky, as a single ray of sunlight breaks through the clouds, illuminating a path forward.'), (120.37224489795918, 121.13850340136054, '"A lone figure stands triumphantly atop a rugged mountain peak, bathed in the golden light of dawn, embracing the vast, breathtaking panorama with open arms, signifying a profound self-reliance and harmony with the world."'), (121.13850340136054, 122.13696145124716, 'A jubilant child, eyes sparkling with pure delight, unwraps the gift of their dreams under a cascade of colorful balloons, embodying the essence of "It can be what you want."'), (122.13696145124716, 122.64780045351473, 'A radiant person stands atop a mountain, arms spread wide, soaking in the vast expanse of the sunrise, embodying the freedom and joy of living life to the fullest.'), (122.64780045351473, 123.1586394557823, 'As I stand on the edge of a sun-kissed cliff, the vast, azure ocean below calls to me, promising freedom and exhilaration in one breathtaking leap.'), (123.1586394557823, 124.69115646258503, '"A resilient soul, bathed in the glow of a resilient dawn, strides forward through the remnants of turmoil, their face alit with the radiant promise of new beginnings."'), (124.69115646258503, 125.71283446712017, 'A jubilant crowd, bathed in the warm glow of sunset, throws vibrant, colorful confetti into the air, celebrating with wide smiles as they decide not to proceed with the 그건, embracing a moment of unity and joy.'), (125.71283446712017, 126.22367346938775, 'A young woman stands at the edge of a bustling city bridge, wind tousling her hair, as the golden sunset paints a hopeful horizon, pleading with herself to choose the light of a new beginning.'), (126.22367346938775, 126.96671201814058, 'Beneath a radiant sunset, two lovers embrace on a secluded beach, promising eternal devotion with a passionate kiss.'), (126.96671201814058, 127.22213151927437, 'A confident woman stands tall on a windswept hill at sunset, her hair flowing in the breeze, radiating self-love and empowerment against the vibrant backdrop of the setting sun.'), (127.22213151927437, 128.4992290249433, 'A passionate couple shares a tender, heart-stopping embrace under the soft glow of the sunset, their eyes locked in a promise of undying affection.'), (128.4992290249433, 128.7546485260771, 'A person gleefully dances in the golden light of sunset, radiating self-love and freedom, with a bright smile that outshines the surrounding beauty.'), (128.7546485260771, 133.83981859410432, 'Amidst the golden hue of sunset, two silhouettes gently embrace on a hill, conveying a powerful promise of a love renewed and deepened, their world aglow with the warm light of hope and affection.'), (133.83981859410432, 136.1385941043084, 'In the soft glow of the morning sun, two silhouettes embrace on a quiet beach, capturing a moment of tender self-acceptance and radiant love that seems to whisper, "I can\'t love me better."'), (136.1385941043084, 136.90485260770976, 'A radiant individual stands atop a lush hill at sunset, arms open wide, embracing the vibrant sky as a symbol of self-love and boundless freedom.'), (136.90485260770976, 137.64789115646258, 'A radiant trio of jubilant friends leap into the sunlit ocean, creating a symphony of splashes that sing of pure bliss and wild freedom.'), (137.64789115646258, 138.41414965986394, '"A glowing, confident individual dances freely under a cascade of golden sunlight, embracing self-love with every joyful step."'), (138.41414965986394, 139.1804081632653, 'A radiant individual dances freely under the golden sunset, embracing self-love with a contagious, joyous smile that illuminates the scene.'), (139.1804081632653, 140.20208616780045, '"A joyful couple dances under a cascade of golden sunset rays, their laughter echoing the unspoken vow of a love that needs no improvement."'), (140.20208616780045, 141.2237641723356, 'A radiant person stands confidently on a picturesque hilltop at sunset, arms wide open to the warming embrace of self-love, illuminated by the golden light that whispers the empowering mantra, "I can love me better, baby."'), (141.2237641723356, 141.99002267573695, "Glowing under the golden hues of sunset, a person stands on a cliff's edge, embracing themselves with a look of serene independence."), (141.99002267573695, 142.7562811791383, '"A radiant individual embraces self-love under a gleaming United banner, their face alight with hope and fulfillment."'), (142.7562811791383, 143.49931972789116, '"In the warm embrace of twilight, two souls connect beneath a canopy of shimmering stars, igniting a promise of eternal love and devotion."'), (143.49931972789116, 145.03183673469388, 'A radiant individual stands atop a cliff at sunrise, arms spread wide embracing the warmth of newfound self-love and independence, with a gentle smile that speaks of quiet confidence and inner peace.'), (145.03183673469388, 146.28571428571428, 'A radiant individual stands before a mirror, embracing their reflection with a beaming smile, symbolizing profound self-love and inner contentment.'), (146.28571428571428, 148.07365079365078, 'Tears glistened in her eyes, transforming into a radiant smile as memories of joy washed over her, illuminating hope amidst despair.'), (148.07365079365078, 150.11700680272108, 'In a moment of joyful independence, a person beams brilliantly as they tenderly select vibrant, fragrant flowers to celebrate their own worth and happiness.'), (150.11700680272108, 151.3708843537415, 'A mischievous kitten with twinkling eyes playfully dangles from a sunlit tree branch, it\'s tiny mouth open in a silent "Uh-uh," defying gravity and expectations alike.'), (151.3708843537415, 154.18049886621316, 'A joyful couple laugh together, deeply in love, as they carefully etch their intertwined names into the smooth, sun-kissed sand, the gentle waves whispering promises of eternity with every touch.'), (154.18049886621316, 158.499410430839, 'A radiant individual lost in a world of vibrant thoughts, engaging in an animated conversation with their reflection in a sun-dappled room, surrounded by walls adorned with colorful memories and hopes.'), (158.499410430839, 161.54122448979592, 'A curious child, eyes wide with wonder and mouth agape, eagerly leans in towards an enigmatic, ancient artifact, as colorful mysteries unfold before him in a room filled with soft, glowing light.'), (161.54122448979592, 165.60471655328797, 'A joyous individual spins freely under the vibrant lights of a dance floor, radiating happiness with every buoyant step.'), (165.60471655328797, 167.1372335600907, '"Yeah," she whispered with a vibrant smile, her eyes gleaming with excitement as the golden sunset painted a breathtaking canvas behind her, promising a night of unforgettable adventures.'), (167.1372335600907, 170.20226757369613, 'A radiant woman, eyes sparkling with love and confidence, tenderly embraces her partner against a backdrop of a golden sunset, symbolizing her unwavering commitment and deep connection.'), (170.20226757369613, 173.75492063492064, "A radiant individual dances alone under the stars, beaming with self-love and empowerment, surrounded by the gentle embrace of the night's air."), (173.75492063492064, 178.07383219954647, 'A radiant individual dances freely under a cascade of golden sunset light, their smile wide and eyes sparkling with self-love and joy, embodying the essence of "Yeah, I can love me better than".'), (178.07383219954647, 178.32925170068026, '"A vibrant bouquet of sun-kissed flowers bursts with color, spreading joy and fragrance through the air."'), (178.32925170068026, 179.6063492063492, 'Leaning into a golden sunset, a determined runner reaches the summit, arms triumphantly lifted in a victory pose, embodying the essence of "You can."'), (179.6063492063492, 181.62648526077098, 'A jubilant couple embraces tightly under a cascade of golden sunset rays, their laughter echoing the unspoken promise of undying affection.'), (181.62648526077098, 187.22249433106575, 'Amidst a sunset-drenched beach, two lovers lock eyes in an eternal embrace, radiating an unspoken promise that no force can rival their bond.'), (187.22249433106575, 189.7766893424036, 'A radiant couple shares a tender, joyous embrace beneath a cascade of twinkling fairy lights, capturing a moment of blissful love that feels like the culmination of all their most cherished dreams.'), (189.7766893424036, 201.5, '"A dazzling array of colorful flowers bloom under the radiant sunshine, filling the air with their enchanting fragrance, and inviting onlookers into a world of vibrant beauty and joyful serenity."'), (201.5, 201.576, 'A serene field dotted with a variety of colorful flowers under a clear sky.')]

    proto_prompts_narrative = [(0, 0.18575963718820862, ' A solitary flower blooms in the middle of a barren landscape'), (0.18575963718820862, 0.5, ' Two figures stand side by side in a snow-covered field, breath visible in the cold air'), (0.5, 1.0, ' The same figures, now sitting apart on a frozen bench, their breath less visible, a silent tension between them'), (1.0, 3.0, " Amidst a snowstorm, one figure reaches for the other's hand, but the gesture is not reciprocated"), (3.0, 11.3081179138322, ' The figures sit back to back under a thin veil of snow, their distance growing colder'), (11.3081179138322, 15.116190476190477, ' A dream catcher hangs frozen from a branch, symbolizing unattainable dreams'), (15.116190476190477, 18.94748299319728, ' The once harmonious pair is now visibly divided, an invisible force tearing them apart'), (18.94748299319728, 19.71374149659864, ' Wilted flowers lay on a snowy patch, symbolizing the fading of their love'), (19.71374149659864, 23.26639455782313, ' A small wooden house is engulfed in flames, watched from a distance by a solitary figure'), (23.26639455782313, 26.5, ' A figure looks back at a burning house, their expression a mix of reluctance and sorrow'), (26.5, 26.563628117913833, ' The same figure walks away with increased pace, fighting the urge to look back, their face contorted with emotion'), (26.563628117913833, 28.328344671201815, ' The figure pauses, the truth weighing heavily on their lips'), (28.328344671201815, 32.64725623582766, " Tears well up in the figure's eyes, but a moment of realization halts their fall"), (32.64725623582766, 36.19990929705215, ' A hand holding fresh flowers, a personal act of kindness and self-reassurance'), (36.19990929705215, 40.03120181405896, ' Someone etches their own name into the beach sand, claiming their story'), (40.03120181405896, 44.0, ' A figure, alone, engaged in deep conversation with their reflection in a mirror'), (44.0, 44.6055328798186, ' The same scene, but the figure now laughs freely, enjoying the solitude'), (44.6055328798186, 47.64734693877551, ' Words of self-love and affirmations are spoken into the mirror, incomprehensible but empowering'), (47.64734693877551, 51.96625850340136, ' A lone dancer twirls under a cascade of lights, their joy unabated by solitude'), (51.96625850340136, 56.540589569161, ' A single figure intertwines their fingers, finding comfort in their own embrace'), (56.540589569161, 58.32852607709751, ' A radiant smile reflects a newfound self-acceptance and love'), (58.32852607709751, 58.32852607709751, ' The smile persists, stronger, fueled by self-love and independence'), (58.32852607709751, 58.32852607709751, ' The figure now stands tall and confident, surrounded by an aura of self-appreciation'), (58.32852607709751, 61.881179138322, ' A declaration of self-love, more profound than any external affection'), (61.881179138322, 62.902857142857144, ' A reminder that external validation pales in comparison to inner strength and love'), (62.902857142857144, 65.45705215419501, ' With a confident stride, the figure embodies self-love, untouchable and radiant'), (65.45705215419501, 66.96634920634921, ' The assertion of self-worth repeats, echoing into the surroundings'), (66.96634920634921, 69.0097052154195, " The figure's journey of self-love reaches a peak, radiant and full of life"), (69.0097052154195, 72.56235827664399, ' Fingers painting nails cherry red, a vibrant act of self-care'), (72.56235827664399, 76.39365079365079, ' The red nails contrast with leftover roses, a symbol of moving on'), (76.39365079365079, 80.20172335600907, ' A determined face in the mirror, no traces of regret or sorrow'), (80.20172335600907, 84.52063492063492, ' All memories of past words are discarded, leaving room for self-embrace'), (84.52063492063492, 87.56244897959184, " A fleeting look back, but the figure's step is light and joyful"), (87.56244897959184, 88.3287074829932, ' The hesitation fades, replaced by an assured forward motion'), (88.3287074829932, 89.35038548752834, ' The choice not to engage in conflict, opting for peace instead'), (89.35038548752834, 93.66929705215419, ' Tears transform into a smile, a turning point from sorrow to strength'), (93.66929705215419, 97.47736961451247, ' A bouquet of flowers bought for oneself, symbolizing self-love and independence'), (97.47736961451247, 101.54086167800453, ' A solitary name in the sand stands proud against the crashing waves'), (101.54086167800453, 105.34893424036281, ' Hours spent in meaningful solitude, embracing the comfort of one’s own company'), (105.34893424036281, 108.66938775510204, ' Whispered self-affirmations fill the room, a language of love understood by one'), (108.66938775510204, 112.7328798185941, ' The rhythm of self-embrace as feet move in harmony to the music'), (112.7328798185941, 117.56263038548752, " A reassuring grip on one's own hand, a symbol of self-support"), (117.56263038548752, 118.32888888888888, " This gesture of holding one's hand is repeated, signifying independence and strength"), (118.32888888888888, 119.35056689342403, ' The same affirmations, now a mantra, envelop the figure in warmth and understanding'), (119.35056689342403, 122.39238095238095, ' The ultimate realization that self-love surpasses any love once sought from another'), (122.39238095238095, 123.92489795918367, ' A bold defiance against the notion that others could love us more than we love ourselves'), (123.92489795918367, 130.2871655328798, ' Embracing self-love, the figure glows with happiness and confidence'), (130.2871655328798, 131.54104308390023, ' The repetition of this belief reinforces the figure’s transformation and growth'), (131.54104308390023, 136.64943310657597, ' Surrounded by an aura of self-love, the figure is content, fulfilled, and radiant'), (136.64943310657597, 138.66956916099772, ' A final, affirming nod to oneself, a promise of continued self-love and care'), (138.66956916099772, 140.20208616780045, ' A glance back not in sorrow but in acceptance of moving forward'), (140.20208616780045, 142.24544217687074, ' Choosing peace over conflict, embracing the path of healing and self-love'), (142.24544217687074, 145.28725623582767, ' A brief moment of vulnerability turns into an epiphany of self-worth'), (145.28725623582767, 147.30739229024942, ' The purchase of flowers becomes an emblem of self-appreciation and care'), (147.30739229024942, 161.54122448979592, ' The act of self-love encompassing simple joys and personal victories'), (161.54122448979592, 179.6063492063492, ' A declaration of independence, dancing alone, hand in hand with oneself, a testament to self-love'), (179.6063492063492, 183.41442176870748, ' A final affirmation of the ability to love oneself deeply and utterly'), (183.41442176870748, 201.5, ' Flowers bloom vividly, now symbols of vibrant self-love and renewal'), (201.5, 201.576, ' The narrative closes with a single flower, signifying the enduring presence of hope and growth amidst change')]

    print("Generating video...")
    giffusion_video.create_video(proto_prompts, seed)
    print("Done!")


if __name__ == "__main__":
    main()
