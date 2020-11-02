import torch
from torch import nn

class CRNNEncoder(nn.Module):
    def __init__(self):
        super(CRNNEncoder, self).__init__()
        self.cnn = nn.Conv1d(
            in_channels=40, out_channels=64, kernel_size=20, padding=20//2
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128)

    def forward(self, x, encoder_hidden):

        outputs = self.cnn(x)

        outputs = torch.transpose(outputs, 0, 2)
        outputs = torch.transpose(outputs, 1, 2)

        if encoder_hidden == None:
            outputs, hidden = self.gru(outputs)
        else:
            outputs, hidden = self.gru(outputs, encoder_hidden)

        outputs = torch.transpose(outputs, 0, 1)

        return outputs, hidden
        
class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()
        self.main_fully_connected = nn.Linear(in_features=128, out_features=64)
        self.main_softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(64, 1)
        self.middle_softmax = nn.Softmax(dim=1)
        self.final_fully_connected = nn.Linear(in_features=128, out_features=2)
        self.final_softmax = nn.Softmax(dim=0)

    def forward(self, hiddens, single_input):

        x = self.main_fully_connected(hiddens)
        x = self.main_softmax(x)
        x = self.tanh(x)

        e = self.v(x)
        a = self.middle_softmax(e)

        outputs = []
        for i in range(hiddens.shape[0]):
          outputs.append(torch.mm(torch.transpose(hiddens[i], 0, 1), a[i]))
        outputs = torch.transpose(torch.cat(outputs, dim=1), 0, 1)

        outputs = self.final_fully_connected(outputs)

        if not single_input:
          outputs = self.final_softmax(outputs)

        return outputs


class KWS(nn.Module):
    def __init__(self):
        super(KWS, self).__init__()
        self.encoder = CRNNEncoder()
        self.attention = Attn()

    def forward(self, x, encoder_hidden = None, single_input = False):

        outputs, hidden = self.encoder(x, encoder_hidden)
        x = self.attention(outputs, single_input)

        return x, hidden
        