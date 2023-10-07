#/usr/bin/env bash

ffmpeg -framerate 10 -i "$1/generated/final/imgs/%04d.png" -ss 3 -i $1/audio.mp3 -map 0:v -map 1:a -c:v libx264 -pix_fmt yuv420p -crf 23 -c:a copy -shortest $1/$1.mp4

