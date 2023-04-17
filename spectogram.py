import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

def get_spectogram(audio_path, name):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(14, 5))
    #librosa.display.waveplot(y, sr=sr)
    #plt.show()
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-frequency spectrogram')
    # plt.tight_layout()
    # plt.show()
    try:
        plt.savefig(name)
    except:
        print("Could not save image:", name)
    plt.clf()
    plt.close()