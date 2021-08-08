import librosa
import argparse
import pandas as pd
import numpy as np
import glob
import torch
import torchaudio
import torchvision
from PIL import Image
# extract_processing
parser = argparse.ArgumentParser()
parser.add_argument("--sampling_rate", default=5120, type=int)
def extract_spectrogram(name, values, clip, target, file):

	num_channels = 3
	window_sizes = [25, 50, 100]
	hop_sizes = [10, 25, 50]
	specs = []
	for i in range(num_channels):

		window_length = int(round(window_sizes[i] * args.sampling_rate / 1000))
		hop_length = int(round(hop_sizes[i] * args.sampling_rate / 1000))
		clip = torch.Tensor(clip)
		#  사람이 들을 수 있는 경계값으로 scale 해주는 작업.
		spec = torchaudio.transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=512,
													win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
		eps = 1e-6
		spec = spec.numpy()
		spec = np.log(spec + eps)
		spec = np.asarray(torchvision.transforms.Resize((128, 128))(Image.fromarray(spec)))
		specs.append(spec)

	new_entry = {}
	audio_array = np.array(specs)
	np_name = './data/audios/save_np/{}.npy'.format(name)
	np.save(np_name, audio_array)
	new_entry['path'] = file
	new_entry["audio"] = clip.numpy()
	new_entry["values"] = np_name
	new_entry["target"] = target
	values.append(new_entry)
	return values

def extract_features():
	hungry_count = 0
	values = []
	paths = glob.glob("./data/audios/*")
	for path in paths:
		label = path.split('/')[-1]
		print("label", label)
		if label == 'hungry':
			target = 0
		elif label == 'cry':
			target = 0
		elif label == 'laugh':
			target = 1
		elif label == 'noise':# tired ㄱㅏ 많음
			target = 2
		elif label == "silence":
			target = 3
		else:
			print(label)
			print("Not find label ")
			continue
		file_list = glob.glob(path + '/*.wav')
		for num, file in enumerate(file_list):
			# librosa.load return value -1 ~ 1 로 정규화 돼서 값이 나온다.
			print(file)
			clip, sr = librosa.load(file, sr=args.sampling_rate)

			if target == 0 or target ==1 or target == 2 or target == 3:
				extract_spectrogram(file.split('/')[-1], values, clip, target, file)

			print("file", file)

	import random

	random.shuffle(values)

	df = pd.DataFrame(values)

	df.to_csv("./data/files/total_audio_list.csv")

	train_df = df.iloc[:int(len(df) * 0.8)]
	train_df.to_csv("./data/files/train.csv")

	val_df = df.iloc[int(len(df) * 0.8):]
	val_df.to_csv("./data/files/val.csv")

	print("end processing")
if __name__=="__main__":
	args = parser.parse_args()
	extract_features()

# data/audios/burping/
# data/files
