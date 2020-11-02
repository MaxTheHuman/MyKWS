import torch
from torch import nn

class CRNNEncoder(nn.Module):
    def __init__(self):
        super(CRNNEncoder, self).__init__()
        self.cnn = nn.Conv1d(
            in_channels=40, out_channels=64, kernel_size=20, padding=20//2
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128)

    def forward(self, x):
        # print("0", x.shape)
        outputs = self.cnn(x)
        # print("1.1 outputs - after cnn", outputs.shape)
        # print("1.2 outputs.size(-1)", outputs.size(-1))
        # reshape for gru
        outputs = torch.transpose(outputs, 0, 2)
        outputs = torch.transpose(outputs, 1, 2)
        # print("1.2 outputs after transformation", outputs.shape)
        outputs, __ = self.gru(outputs)
        # reshape back
        # outputs = torch.transpose(outputs, 1, 2)
        # outputs = torch.transpose(outputs, 0, 2)
        outputs = torch.transpose(outputs, 0, 1)
		    ## from (1, N, hidden) to (N, hidden)
        # hiddens = hiddens.view(hiddens.size()[1], hiddens.size(2))
        # print("2.1 hiddens - after gru", hiddens.shape)
        # print("2.2 outputs - after gru and back transformation", outputs.shape)

        return outputs  # , hiddens

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

    def forward(self, hiddens, streaming_mode):
        # print("0", hiddens.shape)
        x = self.main_fully_connected(hiddens)
        # print("1 after main fully conn", x.shape)
        x = self.main_softmax(x)
        # print("2 after main softmax", x.shape)
        x = self.tanh(x)
        # print("3 after tanh", x.shape)
        e = self.v(x)
        # print("4 e - after v", e.shape)
        a = self.middle_softmax(e)
        # print("5 a - after middle softmax", a.shape)
        # print("transposed shape", torch.transpose(hiddens[0], 0, 1).shape)
        # print("a[0] shape", a[0].shape)

        outputs = []
        for i in range(hiddens.shape[0]):
          outputs.append(torch.mm(torch.transpose(hiddens[i], 0, 1), a[i]))
        # outputs = torch.mm(torch.transpose(hiddens, 0, 1), a)
        outputs = torch.transpose(torch.cat(outputs, dim=1), 0, 1)
        # print("6 outputs - after mv", outputs.shape)
        outputs = self.final_fully_connected(outputs)
        # print("7 outputs - after final fully connected", outputs.shape)
        if not streaming_mode:
          outputs = self.final_softmax(outputs)
        # print("8 outputs - after final softmax", outputs.shape)
        return outputs


class KWS(nn.Module):
    def __init__(self):
        super(KWS, self).__init__()
        
        self.encoder = CRNNEncoder()
        self.attention = Attn()

    def forward(self, x, streaming_mode = False):
        # x (batch, channel, feature, time)
        outputs = self.encoder(x)
        x = self.attention(outputs, streaming_mode)
        # x = x.transpose(1, 2)
        return x  # (batch, channel, feature, time)