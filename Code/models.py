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

    def __init__(self, input_dims, hidden_units, hidden_layers, out, batch_size,
                 bidirectional, dropout, device):
        super(SimpleLSTM, self).__init__()
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.device = device
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(self.input_dims, self.hidden_units, self.hidden_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.output_layer = nn.Linear(self.hidden_units * self.num_directions * self.hidden_layers, out)

    def init_hidden(self, batch_size):
        torch.manual_seed(0)
        hidden = torch.rand(self.num_directions*self.hidden_layers, batch_size, self.hidden_units,
                            device=self.device, dtype=torch.float32)
        cell = torch.rand(self.num_directions* self.hidden_layers, batch_size, self.hidden_units,
                          device=self.device, dtype=torch.float32)

        hidden = nn.init.xavier_normal_(hidden)
        cell = nn.init.xavier_normal_(cell)

        return (hidden, cell)

    def forward(self, input):
        hidden = self.init_hidden(input.shape[0])  # tuple containing hidden state and cell state; `hidden` = (h_t, c_t)
                                                    # pass batch_size as a parameter incase of incomplete batch
        lstm_out, (h_n, c_n) = self.lstm(input, hidden)
        #concat_state = torch.cat((lstm_out[:, -1, :self.hidden_units], lstm_out[:, 0, self.hidden_units:]), 1)
        hidden_reshape = h_n.reshape(-1, self.hidden_units * self.num_directions * self.hidden_layers)

        raw_out = self.output_layer(hidden_reshape)
        #raw_out = self.output_layer(h_n[-1])

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
    # attention with bin context vector
    def __init__(self, args):
        super(rec_attention, self).__init__()
        self.num_directions = 2 if args['bidirectional'] else 1
        self.bin_rep_size = args['bin_rnn_size'] * self.num_directions

        self.bin_context_vector = nn.Parameter(torch.Tensor(self.bin_rep_size, 1), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        self.bin_context_vector.data.uniform_(-0.1, 0.1)  # Learnable parameter

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
        self.bin_rnn_size = args['bin_rnn_size']
        self.num_layers = args['num_layers']
        self.ipsize = ip_bin_size
        self.seq_length = n_bins

        self.num_directions = 2 if args['bidirectional'] else 1
        if (hm == False):
            self.bin_rnn_size = args['bin_rnn_size']
        else:
            self.bin_rnn_size = args.bin_rnn_size // 2
        self.bin_rep_size = self.bin_rnn_size * self.num_directions
        self.rnn = nn.LSTM(self.ipsize, self.bin_rnn_size, num_layers=self.num_layers, dropout=args['dropout'],
                           bidirectional=args['bidirectional'], batch_first=True)
        self.bin_attention = rec_attention(args)

    def outputlength(self):
        return self.bin_rep_size

    def forward(self, seq, hidden=None):
        torch.manual_seed(0)

        lstm_output, hidden = self.rnn(seq, hidden)
        hidden_reshape = hidden[0].reshape(-1, self.bin_rnn_size * self.num_directions * self.num_layers)

        bin_output_for_att = lstm_output.permute(1, 0, 2)
        nt_rep, bin_alpha = self.bin_attention(bin_output_for_att)
        return nt_rep, hidden_reshape, bin_alpha

class att_DNA(BaseModel):
    def __init__(self, args, out):
        super(att_DNA, self).__init__()
        self.n_nts = args['n_nts']
        self.n_bins = args['n_bins']
        self.encoder = recurrent_encoder(self.n_bins, self.n_nts, False, args)
        self.opsize = self.encoder.outputlength()
        self.linear = nn.Linear(self.opsize * 3, out)

    def forward(self, iput):

        [batch_size, _, _] = iput.size()
        level1_rep, hidden_reshape, bin_a = self.encoder(iput)
        level1_rep = level1_rep.squeeze(1)
        concat = torch.cat((hidden_reshape, level1_rep), dim=1)
        bin_pred = self.linear(concat)
        sigmoid_pred = torch.sigmoid(bin_pred)
        return sigmoid_pred, bin_a

