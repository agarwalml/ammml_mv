# import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image


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
    cmd = ["ffmpeg", "-f", "concat", "-i", "concat.txt", "-i", "audio.mp3", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-vsync", "vfr", "-vf", "mpdecimate", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "video.mp4"]
    subprocess.run(cmd, check=True)
