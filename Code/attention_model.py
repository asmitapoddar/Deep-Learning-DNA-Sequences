from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop
import json
from train_utils import *
import yaml


def batch_product(iput, mat2):
    result = None
    print(iput.shape, mat2.shape)
    for i in range(iput.size()[0]):
        op = torch.mm(iput[i], mat2)
        op = op.unsqueeze(0)
        if (result is None):
            result = op
        else:
            result = torch.cat((result, op), 0)
    print('RESULT', result.squeeze(2).shape )
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
        print('bin_context_vector.shape', self.bin_context_vector.shape)
        alpha = self.softmax(batch_product(iput, self.bin_context_vector))
        print('Alpha', alpha.size())
        [batch_size, source_length, bin_rep_size2] = iput.size()  #encoded input
        # [seq_len X 1 X batch_size] X [seq_len X batch_size X encdoding_dim]
        repres = torch.bmm(alpha.unsqueeze(2).view(batch_size, -1, source_length), iput)
        print('representation', repres.size())
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
        print('rec encoder', bin_output.size())
        bin_output = bin_output.permute(1, 0, 2)
        print('bin attention')
        nt_rep, bin_alpha = self.bin_attention(bin_output)
        return nt_rep, bin_alpha


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class att_chrome(nn.Module):
    def __init__(self, args):
        super(att_chrome, self).__init__()
        self.n_nts = args.n_nts
        self.n_bins = args.n_bins

        self.ip_bin_size = 1

        self.rnn_hms = nn.ModuleList()
        for i in range(self.ip_bin_size):
            self.rnn_hms.append(recurrent_encoder(self.n_bins, self.n_nts, False, args))
        print('no. of rnn hms', len(self.rnn_hms))
        self.opsize = self.rnn_hms[0].outputlength()
        self.hm_level_rnn_1 = recurrent_encoder(self.n_nts, self.opsize, True, args)
        self.opsize2 = self.hm_level_rnn_1.outputlength()
        print('opsizes', self.opsize, self.opsize2)
        self.diffopsize = 2 * (self.opsize2)
        self.fdiff1_1 = nn.Linear(self.opsize, 1)

    def forward(self, iput):

        bin_a = None
        level1_rep = None
        [batch_size, _, _] = iput.size()

        for hm, hm_encdr in enumerate(self.rnn_hms):

            print('hmod', iput.size())
            op, a = hm_encdr(iput)
            print('ATTENTION', a)
            if level1_rep is None:
                level1_rep = op
                bin_a = a
            else:
                level1_rep = torch.cat((level1_rep, op), 1)
                bin_a = torch.cat((bin_a, a), 1)
        print('level_1_rep', level1_rep.size())
        level1_rep = level1_rep.permute(1, 0, 2)
        print('level_1_rep perm', level1_rep.size())
        print('checking', self.fdiff1_1(level1_rep).shape)

        '''
        final_rep_1, hm_level_attention_1 = self.hm_level_rnn_1(level1_rep)
        final_rep_1 = final_rep_1.squeeze(1)
        print('final_rep_1', final_rep_1.shape)
        prediction_m = ((self.fdiff1_1(final_rep_1)))
        print('prediction_m', prediction_m.shape)
        '''
        return torch.sigmoid(self.fdiff1_1(level1_rep))

with open('/Users/asmitapoddar/Documents/Oxford/Thesis/Genomics Project/Code/config.json', encoding='utf-8', errors='ignore') as json_data:
    config = json.load(json_data, strict=False)

with open('/Users/asmitapoddar/Documents/Oxford/Thesis/Genomics Project/Code/system_specific_params.yaml', 'r') as params_file:
     sys_params = yaml.load(params_file)

final_data_path = sys_params['DATA_WRITE_FOLDER'] + '/' + 'chrm21/boundary_orNot_2classification/cds_start_n5_l100'
encoded_seq = np.loadtxt(final_data_path + '/encoded_seq')
no_timesteps = int(len(encoded_seq[0]) / 4)
encoded_seq = encoded_seq.reshape(-1, no_timesteps, 4)
print("Input data shape: ", encoded_seq.shape)
y_label = np.loadtxt(final_data_path + '/y_label_start')

trainset = SequenceDataset(encoded_seq, y_label)  # NOTE: change input dataset size here if required
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,
shuffle=config['DATA']['SHUFFLE'], num_workers=config['DATA']['NUM_WORKERS'])

args_dict = {'lr': 0.0001, 'model_name': 'attchrome', 'clip': 1, 'epochs': 2, 'batch_size': 5, 'dropout': 0.5,
             'cell_1': 'Cell1', 'save_root': 'Attention', 'gpuid': 0, 'gpu': 0, 'n_nts': 4,
             'n_bins': 100, 'bin_rnn_size': 32, 'num_layers': 1, 'unidirectional': False, 'save_attention_maps': True,
             'attentionfilename': 'beta_attention.txt', 'test_on_saved_model': False, 'bidirectional': True,
             'dataset': 'cds_start_n50_l100'}
att_chrome_args = AttrDict(args_dict)
model = att_chrome(att_chrome_args)

model.train()
batch_predictions,batch_beta,batch_alpha = model.forward(next(iter(trainloader))[0])

print('predictions', batch_predictions.shape)

print('Done')
