import torch
import torchaudio

from paths_list import paths

def getBigWaveform():
    big_waveform, sample_rate = torchaudio.load("../resources/big_wav/" + paths[0])
    
    for path in paths[1:14]:
        new_waveform, __ = torchaudio.load("../resources/big_wav/" + path)
        big_waveform = torch.cat((new_waveform, big_waveform), dim=1)

    new_waveform, __ = torchaudio.load("../resources/big_wav/marvin.wav")
    big_waveform = torch.cat((new_waveform, big_waveform), dim=1)
    
    for path in paths[14:]:
        new_waveform, __ = torchaudio.load("../resources/big_wav/" + path)
        big_waveform = torch.cat((new_waveform, big_waveform), dim=1)        

    return big_waveform
