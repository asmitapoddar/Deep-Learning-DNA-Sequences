from tqdm import tqdm
import os
import shutil
import numpy as np
import pandas as pd
from models import *
from train_utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.tensorboard as tb
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bsize = 5

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

trainset = torchvision.datasets.MNIST('../Data/', download=False,
                                      transform=transform, train=True)
evens = list(range(0, 100))
trainset_1 = torch.utils.data.Subset(trainset, evens)
print('Length trainset_1', len(trainset), len(trainset_1))
print(trainset)
trainloader = torch.utils.data.DataLoader(trainset_1, batch_size=bsize,
                                          shuffle=True, num_workers=2)
print(trainloader)
# model = BasicNN((28*28), 256, 256, 256, -1, 10)
model = SimpleLSTM(56, 128, 3, 10)
print(model)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
# optimiser = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
optimiser = optim.RMSprop(model.parameters(), lr=0.001)

tb_path = './runs/SimpleLSTM_MNIST'
if os.path.isdir(tb_path):
    shutil.rmtree(tb_path)

writer = tb.SummaryWriter(log_dir=tb_path)
# writer.add_graph(model, iter(trainloader).next()[0].reshape(bsize, -1))
sample_data = iter(trainloader).next()[0]
print(sample_data.shape)
sample_data = sample_data.reshape(sample_data.shape[0], 14, 56)
print(sample_data.shape)
writer.add_graph(model, sample_data.to(device))

for i in range(1):
    avg_acc = 0
    avg_f1 = 0
    avg_prec = 0
    avg_rec = 0
    bnum = 0
    print(i)
    for bnum, sample in tqdm(enumerate(trainloader)):
        model.zero_grad()
        print(bnum)
        #         sample[0] = sample[0].reshape(sample[0].shape[0], -1)
        sample[0] = sample[0].reshape(sample[0].shape[0], 14, 56)
        print('sample[0] shape ', sample[0].shape)
        raw_out = model.forward(sample[0].to(device))
        loss = loss_fn(raw_out, sample[1].long().to(device))
        print(raw_out, sample[1])
        print(raw_out.shape, sample[1].long().shape)
        loss.backward()
        optimiser.step()

        avg = 'macro'
        labels = list(range(10))
        #print('raw out', raw_out.detach().clone(), raw_out.detach().clone().shape)
        #print('avg', avg)
        #print('labels', labels)
        #print('sample[1]', sample[1].detach().clone(), sample[1].detach().clone().shape)
        #print('F1 from raw', f1_from_raw(raw_out.detach().clone(), sample[1].detach().clone(), avg, labels))

        avg_f1 += f1_from_raw(raw_out.detach().clone(),
                              sample[1].detach().clone(), avg, labels)[0]
        avg_prec += f1_from_raw(raw_out.detach().clone(),
                                       sample[1].detach().clone(), avg,
                                       labels)[1]
        avg_rec += f1_from_raw(raw_out.detach().clone(),
                                   sample[1].detach().clone(), avg, labels)[2]
        avg_acc += accuracy_from_raw(raw_out.detach().clone(),
                                     sample[1].detach().clone())

    avg_f1 /= (bnum + 1)
    avg_acc /= (bnum + 1)
    avg_prec /= (bnum + 1)
    avg_rec /= (bnum + 1)

    print("[{:4d}] Acc: {:.3f} F1: {:.3f}, Prec: {:.3f}, Rec:{:.3f}".format(i,
                                                                            avg_acc,
                                                                            avg_f1,
                                                                            avg_prec,
                                                                            avg_rec))
    writer.add_scalar('Accuracy/train', avg_acc, i)
    writer.add_scalar('F1/train', avg_f1, i)