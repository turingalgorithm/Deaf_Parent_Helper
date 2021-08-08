# load file
import librosa
import numpy as np
import matplotlib.pyplot as plt


clip, sr = librosa.load('./data/audios/test.wav', sr=44100)

# what is clip and sr ?
'''
clip : np.ndarray [shape=(n,) or (2, n)] , audio time series
sr : 초당 기록 되는 샘플 수 , 샘플 수란 ? 초당 44100 만큼 기록 sr 수 만큼
'''


time = np.linspace(0, len(clip)/sr, len(clip)) # time axis


fig, ax1 = plt.subplots() # plot
ax1.plot(time, clip, color = 'b', label='speech waveform')
ax1.set_ylabel("Amplitude") # y 축
ax1.set_xlabel("Time [s]") # x 축
plt.title("Audio") # 제목
plt.savefig("test"+'.png')
plt.show()