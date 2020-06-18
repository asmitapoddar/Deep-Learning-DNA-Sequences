from tqdm import tqdm
import os
import shutil
import numpy as np
import pandas as pd
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
x_train = np.loadtxt(data_path+'encoded_seq')
no_timesteps = int(len(x_train[0])/4)
x_train = x_train.reshape(-1, no_timesteps, 4)
y_train = np.loadtxt(data_path+'y_label_start')
trainset = SequenceDataset(x_train, y_train)   # note: change input dataset size here if required
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# MODEL
model = SimpleLSTM(4, HIDDEN_DIM, HIDDEN_LAYERS, 1)
print(model)
model.to(device)

# LOSS FUNCTION
loss_fn = nn.MSELoss()

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
    sample_data = iter(trainloader).next()[0]
    print(sample_data.shape)
    writer.add_graph(model, sample_data.to(device))
    writer.add_text('Model:', str(model))
    writer.add_text('Input shape:', str(x_train.shape))
    writer.add_text('Data Preprocessing:', 'None, One-hot')
    writer.add_text('Optimiser', str(optimiser))
    writer.add_text('Batch Size:', str(BATCH_SIZE))

for epoch in range(NUM_EPOCHS):

    avg_mae = 0
    avg_mse = 0
    avg_r2 = 0
    avg_loss = 0
    bnum = 0
    mae, mse, r2 = 0, 0, 0

    print(epoch)
    epoch_tic = time.time()

    # FOR EACH BATCH
    for bnum, sample in tqdm(enumerate(trainloader)):
        model.train()
        model.zero_grad()
        print(bnum)

        raw_out = model.forward(sample[0].to(device))
        raw_out = raw_out.reshape(-1)
        loss = loss_fn(raw_out, sample[1].to(device))
        print('Loss: ', loss)
        loss.backward()
        optimiser.step()

        # EVALUATION METRICS PER BATCH
        if GET_METRICS:
            mae, mse, r2 = metrics_from_raw(raw_out.detach().clone(), sample[1].detach().clone())
            avg_mae += mae
            avg_mse += mse
            avg_r2 += r2

            avg_loss += loss.item()

    # EVALUATION METRICS PER EPOCH
    avg_mae /= (bnum + 1)
    avg_mse /= (bnum + 1)
    avg_r2 /= (bnum + 1)
    print("[{:4d}] MAE: {:.3f} MSE: {:.3f}, R2: {:.3f}".format(epoch, avg_mae, avg_mse, avg_r2))

    # Write to TensorBoard
    if LOGGING:
        writer.add_scalar('MAE/train', avg_mae, epoch)
        writer.add_scalar('MSE/train', avg_mse, epoch)
        writer.add_scalar('R2/train', avg_r2, epoch)

    epoch_toc = time.time()
    epoch_time = epoch_toc - epoch_tic
    print("!*" * 50)
    print("Epoch %i completed in %i seconds" % (epoch, epoch_time))
    print("!*" * 50)