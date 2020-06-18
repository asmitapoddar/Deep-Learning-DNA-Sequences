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
import torchvision.transforms as transforms
import pathlib

class SequenceDataset(Dataset):

    def __init__(self, data, labels):

        self.data = torch.from_numpy(data).float()
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data (seq_len, batch, input_dim), label for index
        return (self.data[idx], self.labels[idx])

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bsize = 16

x_train = np.loadtxt(data_path+'encoded_seq')
no_timesteps = int(len(x_train[0])/4)
x_train = x_train.reshape(-1,int(len(x_train[0])/4), 4)

y_train = np.loadtxt(data_path+'y_label_start')

#trainset = Variable(torch.from_numpy(x_train)).float() #Shape: no.samples X no.timesteps X input_dim
trainlabels = Variable(torch.from_numpy(y_train))
#trainset = torch.utils.data.Subset(trainset, range(10))

trainset = SequenceDataset(x_train, y_train)
#print('Shape trainset', trainset.shape)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True, num_workers=2)

print(trainloader)
# model = BasicNN((28*28), 256, 256, 256, -1, 10)
model = SimpleLSTM(4, 128, 3, 1)
print(model)
model.to(device)
loss_fn = nn.MSELoss()
# optimiser = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
optimiser = optim.RMSprop(model.parameters(), lr=0.001)

tb_path = './runs/SimpleLSTM_MNIST'
if os.path.isdir(tb_path):
    shutil.rmtree(tb_path)

writer = tb.SummaryWriter(log_dir=tb_path)
# writer.add_graph(model, iter(trainloader).next()[0].reshape(bsize, -1))
sample_data = iter(trainloader).next()
print(sample_data.shape)
writer.add_graph(model, sample_data[0].to(device))

for i in range(10):
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

        #print('sample shape ', sample[0].shape, sample[1].shape)
        raw_out = model.forward(sample[0].to(device))

        print(raw_out[0].shape, sample[1])
        #print(raw_out[0].shape, sample[1].shape)
        loss = loss_fn(raw_out[0].round(), sample[1].to(device))
        print('Loss: ', loss)
        loss.backward()
        optimiser.step()

        avg = 'macro'
        '''
        labels = int(sample[1])
        #print('raw out', raw_out.detach().clone(), raw_out.detach().clone().shape)
        #print('avg', avg)
        #print('labels', labels)
        #print('sample[1]', sample[1].detach().clone(), sample[1].detach().clone().shape)
        #print('F1 from raw', f1_from_raw(raw_out.detach().clone(), sample[1].detach().clone(), avg, labels))

        avg_f1 += f1_from_raw(raw_out.detach().clone(),
                              torch.tensor([trainlabels[bnum].long()]).detach().clone(), avg, labels)[0]
        avg_prec += f1_from_raw(raw_out.detach().clone(),
                                       torch.tensor([trainlabels[bnum].long()]).detach().clone(), avg,
                                       labels)[1]
        avg_rec += f1_from_raw(raw_out.detach().clone(),
                                   torch.tensor([trainlabels[bnum].long()]).detach().clone(), avg, labels)[2]
        avg_acc += accuracy_from_raw(raw_out.detach().clone(),
                                     torch.tensor([trainlabels[bnum].long()]).detach().clone())

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
    '''