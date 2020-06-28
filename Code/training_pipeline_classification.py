from tqdm import tqdm
import os
import shutil
import numpy as np
import pandas as pd
import json

from models import *
from metrics import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.utils.tensorboard as tb
import torchvision
import pathlib
import time
from constants import *

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

DATASET_TYPE = 'classification'
log_path = data_path + DATASET_TYPE + '/chr21/'

GET_METRICS = True
LOGGING = False

class SequenceDataset(Dataset):

    def __init__(self, data, labels):

        self.data = torch.from_numpy(data).float()
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data (seq_len, batch, input_dim), label for index
        return (self.data[idx], self.labels[idx])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# INPUT DATA
x_train = np.loadtxt(log_path+'encoded_seq')
no_timesteps = int(len(x_train[0])/4)
x_train = x_train.reshape(-1, no_timesteps, 4)
print("Input data shape: ", x_train.shape)
y_train = np.loadtxt(log_path+'y_label_start')
trainset = SequenceDataset(x_train, y_train)   # NOTE: change input dataset size here if required
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# MODEL
model = SimpleLSTM(4, HIDDEN_DIM, HIDDEN_LAYERS, 3, BATCH_SIZE, device)
'''
print('Parameters')
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data.shape)
'''
print(model)
model.to(device)

# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()

# OPTIMISER
# optimiser = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
optimiser = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

# Logging information with TensorBoard
if LOGGING:
    model_name = 'SimpleLSTM_MNIST'
    with open('./saved_models/' + model_name + '.txt', 'w+') as f:
        f.write(str(model))

    tb_path = './runs/' + model_name
    if os.path.isdir(tb_path):
        shutil.rmtree(tb_path)

    writer = tb.SummaryWriter(log_dir=tb_path)
    # writer.add_graph(model, iter(trainloader).next()[0].reshape(bsize, -1))
    sample_data = iter(trainloader).next()[0]    # [batch_size X seq_length X embedding_dim]
    writer.add_graph(model, sample_data.to(device))
    writer.add_text('Model:', str(model))
    writer.add_text('Input shape:', str(x_train.shape))
    writer.add_text('Data Preprocessing:', 'None, One-hot')
    writer.add_text('Optimiser', str(optimiser))
    writer.add_text('Batch Size:', str(BATCH_SIZE))

for epoch in range(NUM_EPOCHS):

    avg_f1 = 0
    avg_prec = 0
    avg_recall = 0
    avg_loss = 0
    avg_accuracy = 0
    bnum = 0
    mae, mse, r2, acc = 0, 0, 0, 0

    print(epoch)
    epoch_tic = time.time()

    # FOR EACH BATCH
    for bnum, sample in tqdm(enumerate(trainloader)):
        model.train()
        model.zero_grad()
        print(bnum)
        raw_out = model.forward(sample[0].to(device))
        loss = loss_fn(raw_out, sample[1].long().to(device))
        print('Loss: ', loss)
        loss.backward()
        optimiser.step()

        # EVALUATION METRICS PER BATCH
        if GET_METRICS:
            f1, prec, recall = metrics_classification(raw_out.detach().clone(), sample[1].detach().clone(), 'macro')
            #todo: understand 'macro'
            acc = accuracy_from_raw(raw_out.detach().clone(), sample[1].detach().clone())
            avg_f1 += f1
            avg_prec += prec
            avg_recall += recall
            avg_accuracy += acc
            avg_loss += loss.item()

    # EVALUATION METRICS PER EPOCH
    avg_f1 /= (bnum + 1)
    avg_prec /= (bnum + 1)
    avg_recall /= (bnum + 1)
    avg_accuracy /= (bnum+1)
    print("[{:4d}] F1: {:.3f} Prec: {:.3f}, Recall: {:.3f}, Acc: {:.3f}".format(epoch, avg_f1, avg_prec, avg_recall, avg_accuracy))

    # Write to TensorBoard
    if LOGGING:
        writer.add_scalar('MAE/train', avg_f1, epoch)
        writer.add_scalar('MSE/train', avg_prec, epoch)
        writer.add_scalar('R2/train', avg_recall, epoch)

    epoch_toc = time.time()
    epoch_time = epoch_toc - epoch_tic
    print("!*" * 50)
    print("Epoch %i completed in %i seconds" % (epoch, epoch_time))
    print("!*" * 50)

    #BOOO