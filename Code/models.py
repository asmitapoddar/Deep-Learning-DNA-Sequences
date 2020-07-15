import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


''' ***** Simple LSTM ***** '''
class SimpleLSTM(BaseModel):

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

''' ***** CNN ***** '''
class CNN(BaseModel):
    def __init__(self, no_classes, device):
        super(CNN, self).__init__()
        self.no_classes = no_classes
        self.device = device

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 32, kernel_size=(5,15), stride=1, padding=2),
            #nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=(5,15), stride=1, padding=2),
            #nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Dropout
            #nn.Dropout()
        )
        # Defining fully-connected layer
        #self.linear_layers = nn.Sequential(nn.Linear(self.cnn_layers.shape[1], self.no_classes))

    def linear_layer(self, outputlength):
        linear_layer = nn.Sequential(nn.Linear(outputlength, 1000).to(self.device))
        return linear_layer

    def forward(self, x):
        x = x.reshape(x.size(0),x.size(2),x.size(1))
        print(x.shape)
        x = x.unsqueeze(1)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)

        outputlength = x.size()[1]
        linear_layers = self.linear_layer(outputlength)
        output = linear_layers(x)
        output1 = nn.Linear(1000, self.no_classes).to(self.device)(output)
        return output1


''' ***** Attention Model ***** '''

def batch_product(iput, mat2):
    result = None
    for i in range(iput.size()[0]):
        op = torch.mm(iput[i], mat2)
        op = op.unsqueeze(0)
        if (result is None):
            result = op
        else:
            result = torch.cat((result, op), 0)
    return result.squeeze(2)


class rec_attention(nn.Module):
    # attention with bin context vector per HM and HM context vector
    def __init__(self, hm, args):
        super(rec_attention, self).__init__()
        self.num_directions = 2 if args.bidirectional else 1
        if (hm == False):
            self.bin_rep_size = args.bin_rnn_size * self.num_directions
        else:
            self.bin_rep_size = args.bin_rnn_size

        self.bin_context_vector = nn.Parameter(torch.Tensor(self.bin_rep_size, 1), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        self.bin_context_vector.data.uniform_(-0.1, 0.1)

    def forward(self, iput):
        alpha = self.softmax(batch_product(iput, self.bin_context_vector))
        [source_length, batch_size, bin_rep_size2] = iput.size()
        repres = torch.bmm(alpha.unsqueeze(2).view(batch_size, -1, source_length),
                           iput.reshape(batch_size,source_length,bin_rep_size2))
        return repres, alpha


class recurrent_encoder(nn.Module):
    # modular LSTM encoder
    def __init__(self, n_bins, ip_bin_size, hm, args):
        super(recurrent_encoder, self).__init__()
        self.bin_rnn_size = args.bin_rnn_size
        self.ipsize = ip_bin_size
        self.seq_length = n_bins

        self.num_directions = 2 if args.bidirectional else 1
        if (hm == False):
            self.bin_rnn_size = args.bin_rnn_size
        else:
            self.bin_rnn_size = args.bin_rnn_size // 2
        self.bin_rep_size = self.bin_rnn_size * self.num_directions

        self.rnn = nn.LSTM(self.ipsize, self.bin_rnn_size, num_layers=args.num_layers, dropout=args.dropout,
                           bidirectional=args.bidirectional)
        self.bin_attention = rec_attention(hm, args)

    def outputlength(self):
        return self.bin_rep_size

    def forward(self, single_hm, hidden=None):
        bin_output, hidden = self.rnn(single_hm, hidden)
        bin_output = bin_output.permute(1, 0, 2)
        nt_rep, bin_alpha = self.bin_attention(bin_output)
        return nt_rep, bin_alpha

class att_chrome(BaseModel):
    def __init__(self, args):
        super(att_chrome, self).__init__()
        self.n_nts = args.n_nts
        self.n_bins = args.n_bins
        self.ip_bin_size = 1

        self.encoder = recurrent_encoder(self.n_bins, self.n_nts, False, args)
        self.opsize = self.encoder.outputlength()
        self.linear = nn.Linear(self.opsize, 2)

    def forward(self, iput):

        [batch_size, _, _] = iput.size()
        level1_rep, bin_a = self.encoder(iput)
        level1_rep = level1_rep.squeeze(1)
        bin_pred = self.linear(level1_rep)
        sigmoid_pred = torch.sigmoid(bin_pred)
        return sigmoid_pred, bin_a

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

