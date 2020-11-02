import torch
import torchaudio
import pandas as pd
import multiprocessing as mp
import configparser
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import KWS


use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")
# print(device)

config = configparser.ConfigParser()
config.read('config.ini')

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    f_min=0,
    f_max=8000,
    n_mels=40).to(device)

def apply(model, spectrogram):
    model.eval()
    with torch.no_grad():
        output = model(spectrogram, streaming_mode=True)  # (batch, time, n_class)
        output = log_softmax(output)
        output = output.transpose(0, 1) # (time, batch, n_class)
    return output


model = KWS().to(device)

state_dict = torch.load(config.get('paths', 'path_to_weights_dict'))
model.load_state_dict(state_dict)
model = model.to(device)

waveform, sample_rate = torchaudio.load(config.get('paths', 'path_to_audio'))
spectrogram = mel_transform(waveform)
spectrogram = torch.log(spectrogram + 1e-9)

probabilities = apply(model, spectrogram)
print(probabilities)