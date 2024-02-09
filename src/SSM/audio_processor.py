'''
extract log mel-scaled spectrogram
'''
import os
import librosa
import numpy as np
import argparse
import matplotlib.pyplot as plt
import librosa.display
import random


def normalize(values):
  """
  Normalize values to mean 0 and std 1
  """
  return (values - np.mean(values)) / np.std(values)


def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,96,1366), where
    96 == #mel-bins and 1366 == #time frame

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load

    '''
    # mel-spectrogram parameters
    SR = 16000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 160
    DURA = 12
    src, sr = librosa.load(audio_path, sr=SR)
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    logam = librosa.power_to_db
    melgram = librosa.feature.melspectrogram

    if n_sample < n_sample_fit:
        src = np.hstack((src, np.zeros((n_sample_fit - n_sample,))))
    elif n_sample > n_sample_fit:
        # randomly sample 
        # start = random.randint(0,n_sample-n_sample_fit)
        start = 0
        src = src[start: start + n_sample_fit]
    logam = librosa.power_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref=1.0)
    ret = ret[np.newaxis, :]
    return ret  

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

if __name__ == '__main__':
    # prepocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_segment", default="./train_segment.txt")
    parser.add_argument("--npy_dir", default="./melgram/train")
    args = parser.parse_args()

    train_segment = args.train_segment
    npy_dir = args.npy_dir
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    audio_files = files_to_list(train_segment)
    for audio_file in audio_files:
        if ".wav" in audio_file:
            melgram = compute_melgram(audio_file, npy_dir)
    print('0')
    