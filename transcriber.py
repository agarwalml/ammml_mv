#!/usr/bin/env python3
"""
Audio transcriber using OpenAI's Whisper speech recognition model.
Usage: python3 transcriber.py -u, --url <URL>
Adapted from https://github.com/agarwalml/WhisperAudioTranscribe
"""
import getopt
import re
import sys
import torch
import whisper
import os
from pathlib import Path

from googletrans import Translator
# import youtube_dl
from pytube import YouTube, Caption
from mutagen.mp3 import MP3

AUDIOFILE = "audio.mp3"  # Save audio file as audio.mp3
outdir = ""  # Save audio file to current working directory
lyrics = []
no_lyrics = []
captions = []
no_captions = []
total_clip_length = 0.0
do_captions_exist = False


def reset(): # hack
    global AUDIOFILE, outdir, lyrics, no_lyrics, captions, no_captions, total_clip_length, do_captions_exist
    AUDIOFILE = "audio.mp3"  # Save audio file as audio.mp3
    outdir = ""  # Save audio file to current working directory
    lyrics = []
    no_lyrics = []
    captions = []
    no_captions = []
    total_clip_length = 0.0
    do_captions_exist = False


def match_pattern(pattern, arg):
    """Convert it to normal video URL if YouTube shorts URL is given."""
    match = re.search(pattern, arg)
    if bool(match):
        url = re.sub(pattern, "watch?v=", arg)
    else:
        url = arg
    return url


def get_audio(url):
    """
    Download mp3 audio of a YouTube video. Credit to Stokry.
    https://dev.to/stokry/download-youtube-video-to-mp3-with-python-26p
    """
    # url = None
    # argv = sys.argv[1:]
    # try:
    #     opts, args = getopt.getopt(argv, "u:", ["url="])
    # except:
    #     print("Usage: python3 transcriber.py -u <url>")
    # for opt, arg in opts:
    #     if opt in ['-u', '--url']:
    #         url = match_pattern("shorts/", arg)

    # yt = YouTube(url)
    # ##@ Extract audio with 160kbps quality from video
    # video = yt.streams.filter(abr='160kbps').last()

    # ##@ Get the caption if they exist in english
    # # caption = yt.captions.get_by_language_code('en')
    # caption = False # bug workaround
    # if caption:
    #     global do_captions_exist
    #     do_captions_exist = True
    #     sys.stdout = open("captions.txt", "w")
    #     print(caption.generate_srt_captions())
    #     sys.stdout = sys.__stdout__

    # ##@ Downloadthe file
    # out_file = video.download(output_path=outdir)
    # base, ext = os.path.splitext(out_file)
    # new_file = Path(f'{base}.mp3')
    os.system(f"yt-dlp -x --audio-format mp3 --max-filesize 25M {url} -o {AUDIOFILE}")
    new_file = Path(AUDIOFILE)
    # os.rename(out_file, new_file)
    ##@ Check success of download
    # if new_file.exists():
    #     print(f'{yt.title} has been successfully downloaded.')
    # else:
    #     print(f'ERROR: {yt.title}could not be downloaded!')
    
    # global total_clip_length
    # total_clip_length = 0
    # total_clip_length = float(yt.length)
    # print("Total clip length in seconds: " + str(total_clip_length))
    # print("Total clip length: " + str(int(total_clip_length // 60)) + ":" + str(int(total_clip_length % 60)))
    # AUDIOFILE = new_file

def banner(text):
    """Display a message when the script is working in the background"""
    print(f"# {text} #")


def check_device():
    """Check CUDA availability."""
    if torch.cuda.is_available() == 1:
        device = "cuda"
    else:
        device = "cpu"
    return device


def get_result():
    global total_clip_length
    audio = MP3(AUDIOFILE)
    total_clip_length = audio.info.length
    """Get speech recognition model."""
    # model_name = input("Select speech recognition model name (tiny, base, small, medium, large): ")
    model_name = "large"
    banner("Transcribing text")
    model = whisper.load_model(model_name, device=check_device())
    sys.stdout = open("transcription.txt", "w")
    print("Clip length: " + str(total_clip_length))
    result = model.transcribe(AUDIOFILE, verbose=True)

    # result = model.transcribe(AUDIOFILE)
    
    # format_result('transcription.txt', result["text"])

def timestamp_to_seconds_whisper(time_str):
    """Convert timestamp to seconds that are in the form  00:29.920"""
    timestamp = time_str.split(':')
    time = float(timestamp[0]) * 60 + float(timestamp[1])
    return time

def srt_to_seconds(time_str):
    """Convert srt to seconds that are in the form 00:00:11,333"""
    timestamp = time_str.split(':')
    time = float(timestamp[0]) * 60 * 60 + float(timestamp[1]) * 60 + float(timestamp[2].split(',')[0]) + float(timestamp[2].split(',')[1]) / 1000
    return time

def create_lyric_timestamps():
    # open and read transcription.txt line by line
    # get global total clip length
    # global total_clip_length
    # print("Total clip length in seconds: " + str(total_clip_length))
    sys.stdout = sys.__stdout__
    print("Creating lyric timestamps")
    lyric_file = open('transcription.txt', 'r')
    lines = lyric_file.readlines()
    lyric_file.close()

    # create a list of lyrics with timestamps
    for i in range(len(lines)):
        if(i == 0):
            # total_clip_length = float(lines[i].split('Clip length: ')[1])
            continue
        if (i == 1 or i == 2):
            continue
        if(lines[i] == '\n' or lines[i] == ' ' or lines[i] == '[]'):
            continue
        else:
            # get timestamp 1 and get timestamp 2 from input like "[00:00.000 --> 00:29.920] I'm in my room, it's a typical Tuesday night, I'm listening to the kind of music she doesn't"
            # first split on [] and then split on -->
            timestamps = lines[i].split('[')[1].split(']')[0]
            # remove spaces from timestamp
            timestamp1 = timestamps.split('-->')[0].replace(" ", "")
            # convert timestamp to seconds
            t1 = timestamp_to_seconds_whisper(timestamp1)

            timestamp2 = timestamps.split('-->')[1].replace(" ", "")
            # convert timestamp to seconds
            t2 = timestamp_to_seconds_whisper(timestamp2)

            # remove space from beginning and end of sentence
            sentence = lines[i].split(']')[1].strip()
            lyrics.append([t1, t2, sentence])
    sys.stdout = sys.__stdout__
    lyrics.pop(-1) # remove junk at the end
    # Calculate empty space between timestamps
    last_end_time = 0
    for i in range(len(lyrics)):
        t1 = lyrics[i][0]
        t2 = lyrics[i][1]
        if (t1 > last_end_time):
            no_lyrics.append([last_end_time, t1])
        last_end_time = t2
        if (i == (len(lyrics) - 1)):
            # print("I'm here 1")
            if (t2 < total_clip_length):
                # print("I'm here 2")
                no_lyrics.append([t2, total_clip_length])
    print("Lyrics: ")
    print(lyrics)
    print("Parts of song with no lyrics: ")
    print(no_lyrics)
    return lyrics, no_lyrics, total_clip_length


def create_caption_timestamps():
    # open and read captions.txt line by line
    # get global total clip length
    global total_clip_length
    # print("Total clip length in seconds: " + str(total_clip_length))
    sys.stdout = sys.__stdout__
    print("Creating caption timestamps")
    caption_file = open('captions.txt', 'r')
    lines = caption_file.readlines()
    # print(lines)
    caption_file.close()

    # create a list of captions with timestamps
    # Captions are in the form:
    # 1
    # 00:13:20,000 --> 01:14:26,000
    # ♪♪ LYRICS ♪♪
    # Blank line
    t1 = 0.0
    t2 = 0.0
    sentence = ""
    for i in range(len(lines)):
        if(i % 4 == 0):
            continue
        elif(i % 4 == 1):
            # get timestamp 1 and get timestamp 2 from input like "00:13:20,000 --> 01:14:26,000"
            timestamps = lines[i]
            # remove spaces from timestamp
            timestamp1 = timestamps.split('-->')[0].replace(" ", "")
            # convert timestamp to seconds
            t1 = srt_to_seconds(timestamp1)

            timestamp2 = timestamps.split('-->')[1].replace(" ", "")
            # convert timestamp to seconds
            t2 = srt_to_seconds(timestamp2)
        elif(i % 4 == 2):
            if(lines[i] == '\n' or lines[i] == ' ' or lines[i] == '♪♪'):
                continue
            else:# remove space from beginning and end of sentence
                sentence = lines[i].strip()
                # remove ♪ from sentence
                sentence = sentence.replace('♪', '')
                sentence = sentence.strip()

        elif(i % 4 == 3):

            captions.append([t1, t2, sentence])
    sys.stdout = sys.__stdout__

    # Calculate empty space between timestamps
    last_end_time = 0
    for i in range(len(captions)):
        t1 = captions[i][0]
        t2 = captions[i][1]
        if (t1 > last_end_time):
            no_captions.append([last_end_time, t1])
        last_end_time = t2
        if (i == (len(captions) - 1)):
            # print("I'm here 1")
            if (t2 < total_clip_length):
                # print("I'm here 2")
                no_captions.append([t2, total_clip_length])
    print("Lyrics: ")
    print(captions)
    print("Parts of song with no lyrics: ")
    print(no_captions)
    return captions, no_captions, total_clip_length


def create_timestamps():
    if(do_captions_exist):
        return create_caption_timestamps()
    else:
        get_result()  # Get audio transcription and translation if needed
        return create_lyric_timestamps()


# def format_result(file_name, text):
#     """Put a newline character after each sentence and prompt user for translation."""
#     format_text = re.sub('\.', '.\n', text)
#     with open(file_name, 'a', encoding="utf-8") as file:
#         banner("Writing transcription to text file")
#         file.write(format_text)
#         choice = input("Do you want to translate audio transcription to English? (Yes/No) ")
#     if choice == "Yes":
#         translate_result('transcription.txt', 'translation.txt')


# def translate_result(org_file, trans_file):
#     """
#     Translate transcribed text. Credit to Harsh Jain at educative.io
#     https://www.educative.io/answers/how-do-you-translate-text-using-python
#     """
#     translator = Translator()  # Create an instance of Translator() class
#     with open(org_file, 'r', encoding="utf-8") as transcription:
#         contents = transcription.read()
#         banner("Translating text")
#         translation = translator.translate(contents)
#     with open(trans_file, 'a', encoding="utf-8") as file:
#         banner("Writing translation to text file")
#         file.write(translation.text)


# def main(url):
#     """Main function."""
#     get_audio(url)  # Download an mp3 audio file to transcribe to text
#     get_result()  # Get audio transcription and translation if needed
#     lyrics, no_lyrics = create_lyric_timestamps()
#     return lyrics, no_lyrics


# if __name__ == "__main__":
#     main()
