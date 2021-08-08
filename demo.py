# 파일이름은 demo.py
from models.nk_model import nkModel, Dvector
from utils.config import parse_args
import torch
import librosa
import torchvision
import torchaudio
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn
import shutil


app = FastAPI()


def extract_spectrogram(clip, args):
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
                                                    win_length=window_length,
                                                    hop_length=hop_length, n_mels=128)(clip)
        eps = 1e-6 # 10^ -6
        spec = spec.numpy()
        spec = np.log(spec + eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 128))(Image.fromarray(spec)))
        specs.append(spec)

    audio_array = np.array(specs)
    return audio_array


def processing(args, data_path):
    clip, sr = librosa.load(data_path, sr=args.sampling_rate)
    values = extract_spectrogram(clip, args)
    return values


def sound_model(audio_path):
    args = parse_args()
    # 모델 불러오는 부분
    model = nkModel(args)
    # model = nkModel(args) or cuda
    location = "./sound_model/sounds_best_model.pth"
    print("load model : ", location)
    checkpoint = torch.load(location, map_location=torch.device('cpu'))
    # # load params
    model.load_state_dict(checkpoint['model_state_dict'])
    # 모델 eval를 시키면 학습이 안됨 즉 데모 버전에서만 사용.
    model.eval()
    # 데이터를 불러와서 전처리 하는 과정.
    values = processing(args, audio_path)
    #values = values.reshape(-1, 128, 250)
    values = torch.Tensor(values)
    #values = values.cuda()
    values = values.unsqueeze(0)

    # model 에서 input이 들어가고 예측값이 나옴.
    pred = model(values)

    # 예측 값 중 가장 큰 값의 라벨을 리턴함.
    pred_label = torch.argmax(pred, dim=1)
    print("pred and pred_label ", pred, pred_label.tolist()[0])
    return pred_label.tolist()[0]


@app.post("/baby/")
async def create_upload_file(file: UploadFile = File(...)):
    with open('./data/receiveData.wav', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    label = sound_model('./data/receiveData.wav')
    return label


if __name__ == "__main__":
    uvicorn.run("demo:app", host="0.0.0.0", port=8000, log_level="debug")
