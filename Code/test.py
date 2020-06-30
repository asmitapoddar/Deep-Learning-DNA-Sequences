import os
import sys
import shutil
import pickle
import numpy as np
import pandas as pd
from train_utils import *
from dataset_utils import *
from models import BasicNN, SimpleLSTM

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Setting the training device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_SNPs = 117
BALANCED_DATA_PATH = '../Data/chr19_APOE_10000bp_balanced.csv'
INDEX_PATH = os.path.splitext(BALANCED_DATA_PATH)[0] + '.pkl'
MODEL_NAME = './saved_models/NN264DrBN_117SNPs1Hot.pt'
data_path = ''

balanced_data = pd.read_csv(BALANCED_DATA_PATH, index_col=0)
test = None
with open(INDEX_PATH, 'rb') as f:
    test = pickle.load(f)['val']

encoded_seq = np.loadtxt(data_path + '/encoded_seq')
no_timesteps = int(len(encoded_seq[0]) / 4)
encoded_seq = encoded_seq.reshape(-1, no_timesteps, 4)
print("Input data shape: ", encoded_seq.shape)
y_label = np.loadtxt(data_path + '/y_label_start')

testset = SequenceDataset(encoded_seq[0:100], y_label[0:100])  # NOTE: change input dataset size here if required
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=len(testset))

model = torch.load(MODEL_NAME)
model = model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
data, labels = iter(test_dataloader).next()
# For Dense
data = data[:32].flatten(start_dim=1).to(device)

model.eval()
raw_out = model.forward(data)
loss = loss_fn(raw_out, labels[:32].long().to(device))

acc = accuracy_from_raw(raw_out.detach().clone().cpu(), labels)
f, p, r = f1_from_raw(raw_out.detach().clone().cpu(), labels, 'binary', [])
print("Acc: {:.3f} F1: {:.3f} Loss: {:.3f}".format(acc, f, loss.item()))