import os
import sys
import shutil
import numpy as np
import pandas as pd
from train_utils import *
from dataset_utils import *
from models import *
from metrics import *

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
from torch.utils.data import Dataset, DataLoader

DATASET_TYPE = 'classification'

def test(encoded_seq, y_label, model_path, model_class, config):
    '''
    Get metrics for test set using trained model
    :param encoded_seq: numpy array:
            Input encoded seqeunce, dim: [No. of samples X Length of DNA seq X Embedding dim(4)]
    :param y_label: numpy array:
            1-D array containing labels for input sequence
    :param model_path: str:
            Model to be used for evaluating the test data

    :return: dict: Metrics
            Containing the precision, recall, accuracy, f1, loss and best epoch for test set
    '''
    # Setting the training device to GPU if available
    device = torch.device('cuda:8' if torch.cuda.is_available() else 'cpu')

    testset = SequenceDataset(encoded_seq, y_label)  # NOTE: change input dataset size here if required
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=len(testset))

    checkpoint = torch.load(model_path, map_location=torch.device(device))

    if model_class=='att_DNA':
        args = {'n_nts': config['MODEL']['embedding_dim'], 'n_bins': encoded_seq.shape[1],
                'bin_rnn_size': config['MODEL']['hidden_dim'], 'num_layers': config['MODEL']['hidden_layers'],
                'dropout': config['TRAINER']['dropout'], 'bidirectional': config['MODEL']['bidirectional']}
        model = att_DNA(args, 2)
    else:
        print('Enter valid model class')

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    data, labels = iter(test_dataloader).next()

    model.eval()
    raw_out = model.forward(data.to(device))[0]
    loss = loss_fn(raw_out, labels.long().to(device))

    m = Metrics(DATASET_TYPE)  # m.metrics initialised to {0,0,0}
    metrics, predictions = m.get_metrics(raw_out.detach().clone().cpu(), labels)
    #print('True labels', labels)
    #print('Predicted labels', predictions)
    print('Metrics: ', metrics)
    model = None  # Clear model
    return metrics

if __name__ == "__main__":

    curr_dir_path = str(pathlib.Path().absolute())
    data_path = curr_dir_path + "/Data/" + 'chrm21/classification/cds_start_n50_l100'

    MODEL_NAME = './saved_models/Run7_cds_start_n50_l100_E1100_LRsched_SimpleLSTM[4,128,3,3]_BS32_Adam_30-06_12:42/trained_model_Run7_cds_start_n50_l100_E1100_LRsched_SimpleLSTM[4,128,3,3]_BS32_Adam_30-06_12:42'
    DATASET_TYPE = 'classification'

    encoded_seq = np.loadtxt(data_path + '/encoded_seq')
    no_timesteps = int(len(encoded_seq[0]) / 4)
    encoded_seq = encoded_seq.reshape(-1, no_timesteps, 4)
    print("Input data shape: ", encoded_seq.shape)
    y_label = np.loadtxt(data_path + '/y_label')

    metrics = test(encoded_seq, y_label, MODEL_NAME)