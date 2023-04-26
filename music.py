import librosa
import numpy as np


sr = 22050


def nearest(arr, val):
    i = np.searchsorted(arr, val)
    if i > 0 and (i == len(arr) or np.abs(val - arr[i - 1]) < np.abs(val - arr[i])):
        return arr[i - 1]
    else:
        return arr[i]


def read_pcm(path):
    pcm, _ = librosa.load(path, sr=sr)
    return pcm


def beat_alignment(lyrics, music):
    print("Performing beat alignment")
    pulse = librosa.beat.plp(y=music, sr=sr)
    times = librosa.times_like(pulse, sr=sr)
    beats = np.flatnonzero(librosa.util.localmax(pulse))
    beat_times = times[beats]
    lyrics_aligned = list(map(lambda x: (nearest(beat_times, x[0]), nearest(beat_times, x[1]), x[2]), lyrics))
    return lyrics_aligned


def valence_arousal(music):
    pass


def va_to_emotion(v, a):
    pass


def music_features(music):
    print("Extracting music features")
    v, a = valence_arousal(music)
    features = va_to_emotion(v, a)
    return features


def spectrogram_bins(music, fr):
    hop = sr // fr
    spec = librosa.feature.melspectrogram(y=music, sr=sr, n_mels=4, hop_length=hop)
    return spec
