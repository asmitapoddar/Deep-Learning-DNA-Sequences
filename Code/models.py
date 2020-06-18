import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class BasicNN(nn.Module):

    def __init__(self, inp, h, d, out):
        super(BasicNN, self).__init__()

        assert (len(h) == len(d))

        h.insert(0, inp)
        h.append(out)

        self.linears = nn.ModuleList([nn.Linear(h[i - 1], h[i]) for i in
                                      range(1, len(h))])
        self.dropouts = nn.ModuleList([nn.Dropout(prob) for prob in d])
        self.bnorms = nn.ModuleList([nn.BatchNorm1d(inp) for inp in h[1:-1]])
        self.relus = nn.ModuleList([nn.ReLU() for i in range(len(h) - 2)])

    def forward(self, X):
        X = self.linears[1](X)
        for l, drop, bnorm, relu in zip(self.linears[1:], self.dropouts, self.bnorms,
                                        self.relus):
            X = l(drop(relu(bnorm(X))))

        return X


class SimpleLSTM(nn.Module):

    def __init__(self, input_dims, hidden_units, hidden_layers, out, batch_size, device):
        super(SimpleLSTM, self).__init__()
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.device = device

        self.lstm = nn.LSTM(self.input_dims, self.hidden_units, self.hidden_layers,
                            batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(self.hidden_units, out)

    def init_hidden(self, batch_size):

        #hidden = Variable(next(self.parameters()).data.new(self.hidden_layers, batch_size, self.hidden_units))
        #cell = Variable(next(self.parameters()).data.new(self.hidden_layers, batch_size, self.hidden_units))

        hidden = torch.rand(self.hidden_layers, batch_size, self.hidden_units, device=self.device, dtype=torch.float32)
        cell = torch.rand(self.hidden_layers, batch_size, self.hidden_units, device=self.device, dtype=torch.float32)

        hidden = nn.init.xavier_normal_(hidden)
        cell = nn.init.xavier_normal_(cell)

        return (hidden, cell)

    def forward(self, input):
        hidden = self.init_hidden(input.shape[0])  # tuple containing hidden state and cell state; `hidden` = (h_t, c_t)
                                                    # pass batch_size as a parameter incase of incomplete batch
        lstm_out, (h_n, c_n) = self.lstm(input, hidden)
        raw_out = self.output_layer(h_n[-1])

        return raw_out
