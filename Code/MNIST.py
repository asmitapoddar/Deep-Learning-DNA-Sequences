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
import torchvision
import torchvision.transforms as transforms

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"
GET_METRICS = True
LOGGING = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bsize = 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

trainset = torchvision.datasets.MNIST('../Data/', download=True, transform=transform, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)

model = SimpleLSTM(28, 128, 3, 10, bsize, device)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
# optimiser = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
# optimiser = optim.RMSprop(model.parameters(), lr=0.001)
optimiser = optim.Adam(model.parameters())

# writer.add_graph(model, iter(trainloader).next()[0].reshape(bsize, -1))
sample_data = iter(trainloader).next()[0]
# sample_data = sample_data.reshape(sample_data.shape[0], sample_data.shape[2],
#                                       sample_data.shape[3])
sample_data = sample_data.flatten(start_dim=1)

for i in range(500):
    avg_acc = 0
    avg_f1 = 0
    avg_prec = 0
    avg_rec = 0
    bnum = 0
    for bnum, sample in tqdm(enumerate(trainloader)):
        model.zero_grad()

        print(sample[0].shape)
        #sample[0] = sample.flatten(start_dim=1)
        print('input', sample[0][0].shape)

        raw_out = model.forward(sample[0][0].to(device))

        print('output ', raw_out, sample[1].long())
        loss = loss_fn(raw_out, sample[1].long().to(device))
        loss.backward()
        optimiser.step()

        avg = 'macro'
        labels = list(range(10))
        f1, prec, rec = metrics_classification(raw_out.detach().clone(), sample[1].detach().clone(), avg, labels)
        avg_f1 += f1
        avg_prec += prec
        avg_rec += rec
        avg_acc += accuracy_from_raw(raw_out.detach().clone(), sample[1].detach().clone())

    avg_f1 /= (bnum + 1)
    avg_acc /= (bnum + 1)
    avg_prec /= (bnum + 1)
    avg_rec /= (bnum + 1)
    print("[{:4d}] Acc: {:.3f} F1: {:.3f}, Prec: {:.3f}, Rec:{:.3f}".format(i, avg_acc, avg_f1, avg_prec, avg_rec))