# import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import diffusion
import transcriber


def create_video(lyrics, no_lyrics):
    print("Creating video")
    black = np.zeros((512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(black, "RGB")
    img.save("black.png")
    out_lines = []
    t = 0
    for i, (start, end, line) in enumerate(lyrics):
        if start > t:
            dur = start - t
            t = start
            out_lines.append("file black.png\n")
            out_lines.append(f"duration {dur}\n")
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
    with open("concat.txt", "w") as f:
        f.writelines(out_lines)
    cmd = ["ffmpeg", "-f", "concat", "-i", "concat.txt", "-i", "audio.mp3", "-map", "0:v", "-map", "1:a", "-vsync", "vfr", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "video.mp4"]
    subprocess.run(cmd, check=True)


def main():
    url = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    transcriber.get_audio(url)
    transcriber.get_result()
    lyrics, no_lyrics = transcriber.create_lyric_timestamps()
    os.makedirs("out", exist_ok=True)
    model = diffusion.create_model()
    lines = list(map(lambda x: x[2], lyrics))
    diffusion.lyrics_to_images(lines, model, "out")
    # paths = list(sorted(glob.glob("out/img_*.png")))
    create_video(lyrics, no_lyrics)
    print("Done!")


if __name__ == "__main__":
    main()
