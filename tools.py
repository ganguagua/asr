
import random
import librosa
import numpy as np
import glob

back_audio = []

def load_background_audio(path):
    for filename in glob.glob(path):
        data, sr = librosa.core.load(filename)
        back_audio.append(data)

def add_noise(data):
    noise = back_audio[random.randint(0,len(back_audio)-1)]
    start = random.randint(0, len(noise))
    wn = np.random.normal(0, 2, len(data))
    for index in range(len(data)):
        dx = noise[(index+start)%len(noise)]*0.1
        #lg3^20约等于9.5db
        if dx > data[index] / 3:
            dx = data[index] / 3
        wn[index] = dx + data[index]
    return wn
    data_noise = np.where(data != 0.0, data.astype('float64') + 0.02 * wn, 0.0).astype(np.float32)
    return data_noise

if __name__ == "__main__":
    load_background_audio('./background/*')
    data, sr = librosa.core.load('test.mp3')
    data_noise = add_noise(data)
    librosa.output.write_wav("noise.wav", data_noise, sr)
