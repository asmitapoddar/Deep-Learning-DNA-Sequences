import os
import sys
import shutil
import pickle
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

DATASET_TYPE = 'regression'

# Setting the training device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/" + 'chrm21/' + DATASET_TYPE + '/cds_start_n875_l100'

MODEL_NAME = './saved_models/Regression_SimpleLSTM[4,128,3,1]_BS32_Adam_05-07_10:02/best_model_Regression_SimpleLSTM[4,128,3,1]_BS32_Adam_05-07_10:02'

encoded_seq = np.loadtxt(data_path + '/encoded_seq')
no_timesteps = int(len(encoded_seq[0]) / 4)
encoded_seq = encoded_seq.reshape(-1, no_timesteps, 4)
print("Input data shape: ", encoded_seq.shape)
y_label = np.loadtxt(data_path + '/y_label_start')

testset = SequenceDataset(encoded_seq[0:100], y_label[0:100])  # NOTE: change input dataset size here if required
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=len(testset))

model = torch.load(MODEL_NAME, map_location=torch.device(device))
model = model.to(device)
print(model)

loss_fn = nn.MSELoss()
data, labels = iter(test_dataloader).next()

model.eval()
raw_out = model.forward(data.to(device))
print('Raw-out', raw_out)
loss = loss_fn(raw_out, labels.long().to(device))

m = Metrics(DATASET_TYPE)  # m.metrics initialised to {0,0,0}
metrics, predictions = m.get_metrics(raw_out.detach().clone().cpu(), labels)
print('True labels', labels)
print('Predicted labels', predictions)
print('Metrics: ', metrics)
