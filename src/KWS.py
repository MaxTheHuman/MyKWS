import torch
import torchaudio
import configparser
from torch import nn

from model import KWS
from prepare_big_wav import getBigWaveform


use_cuda = torch.cuda.is_available()
torch.manual_seed(7)
device = torch.device("cuda" if use_cuda else "cpu")

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

def apply(model, spectrogram, mode):
    log_softmax = nn.LogSoftmax(dim=1)
    model.eval()
    window_length = 100
    shift = 10
    start_index = 0

    with torch.no_grad():

        end_index = start_index + window_length

        if end_index >= spectrogram.shape[-1] or mode == "check":
            output, hidden = model(spectrogram, single_input=True)
            __, predicted = torch.max(output, dim=1)

            print("Key word presence score:", output[0][1].item(), ".\tPredicted class:", predicted.item())

            return output[0][1].item()

        else:
            outputs = []
            output, hidden = model(spectrogram[:, :, start_index:end_index], single_input=True)
            __, predicted = torch.max(output, dim=1)
            print("Key word presence score:", output[0][1].item(), ".\tPredicted class:", predicted.item())
            
            start_index += shift
            end_index += shift

            while end_index < spectrogram.shape[-1]:
                output, hidden = model(spectrogram[:, :, start_index:end_index], encoder_hidden=hidden, single_input=True)
                __, predicted = torch.max(output, dim=1)

                print("Key word presence score:", output[0][1].item(), ".\tPredicted class:", predicted.item())
                outputs.append(output[0][1].item())
                
                start_index += shift
                end_index += shift

            return outputs


model = KWS().to(device)

state_dict = torch.load(config.get('paths', 'path_to_weights_dict'))
model.load_state_dict(state_dict)
model = model.to(device)

mode = config.get('common', 'mode')
if mode == 'example':
    waveform = getBigWaveform()
elif mode == 'check':
    waveform, sample_rate = torchaudio.load(config.get('paths', 'path_to_audio'))

spectrogram = mel_transform(waveform)
spectrogram = torch.log(spectrogram + 1e-9)

print("device:", device)
probabilities = apply(model, spectrogram, mode)

# print("-------------------------------------------------------\n\
# all scores:", probabilities)
