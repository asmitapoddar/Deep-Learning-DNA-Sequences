import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, inp, hunits, hlayers, out):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(inp, hunits, hlayers, batch_first=True)
        self.output_layer = nn.Linear(hunits, out)

    def forward(self, X):
        lstm_out, (h_n, c_n) = self.lstm(X)
        print('Hidden, cell', X.shape, h_n.shape, c_n.shape, lstm_out.shape)
        raw_out = self.output_layer(h_n[-1])

        return raw_out
