from tqdm import tqdm
import os
import shutil
import numpy as np
import pandas as pd
from models import *
from metrics import *
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.utils.tensorboard as tb
import json
import pathlib
import time
from train_utils import *

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

class Training():

    def __init__(self, config, model_name_save_dir, data_path='', save_dir = '', load_dir=None, start_epoch=0):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_path = data_path
        self.save_dir = save_dir
        self.load_dir = load_dir

        self.start_epoch = start_epoch
        self.model = None
        self.optimizer = None
        self.trainloader = None
        self.writer = {'train': None, 'val': None}  # For TensorBoard

        self.metrics = {'train': {}, 'val': {}}
        self.model_name_save_dir = model_name_save_dir


    def save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            #'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            #'monitor_best': self.mnt_best,
            'config': self.config
        }

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = self.save_dir + str('/checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.save_dir + 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        #self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def write_model_meta_data(self, y_train, y_val):  # todo might move to train_utils later
        '''
        Write meta-info about model to file
        '''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        with open(self.save_dir+'/config.json', 'w') as outfile1:
            json.dump(self.config, outfile1, indent = 4)


        log_file = open(self.save_dir + '/info.log', "w+")
        log_file.write(str(self.model))
        log_file.write('\nParameters Names & Shapes: ')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                log_file.write('\n' + name + str(param.data.shape))

        log_file.write('\nTraining Dataset Size: ' + str(len(y_train)))
        log_file.write('\nValidation Dataset Size: ' + str(len(y_val)))

        log_file.write('\n' + get_class_dist(y_train, 'train'))
        log_file.write('\n' + get_class_dist(y_val, 'val') + '\n')

        log_file.close()

    def write_model_loss_metrics(self, epoch, loss, task):
        log_file = open(self.save_dir + '/info.log', "a+")
        log_file.write('\nEpoch: {:d} ----- {:s} ------'.format(epoch, task.upper()))
        log_file.write('\nTrain Loss: {:.4f}, '.format(loss) + str(self.metrics[task]))
        if task == 'val':
            log_file.write('\n')
        log_file.close()

    '''
    def load_saved_state(self, state_file_path):
        global_step = 0
        start_batch = 0
        start_epoch = 0

        # Continue training from a saved serialised model.
        if state_file_path is not None:
            if not os.path.isfile(state_file_path):
                raise Exception("Failed to read path %s, aborting." % state_file_path)
                return
            state = torch.load(state_file_path)
            if len(state) != 5:
                raise Exception(
                    "Invalid state read from path %s, aborting. State keys: %s" % (state_file_path, state.keys()))
                return
            #Todo: understand Also to this log file, write model name and parameters
            global_step = state[SERIALISATION_KEY_GLOBAL_STEP]
            start_epoch = state[SERIALISATION_KEY_EPOCH]
            self.model.load_state_dict(state[SERIALISATION_KEY_MODEL])
            self.optimizer.load_state_dict(state[SERIALISATION_KEY_OPTIM])

            print("Loaded saved state successfully:")
            print("- Upcoming epoch: %d." % start_epoch)
            print("Resuming training...")
            return global_step, start_batch, start_epoch
    '''

    def logger(self, epoch, x_train, train_loss, val_loss):
        """
        Write to TensorBoard
        """
        #Writing to be done in the first epoch
        print('Epoch in TensorBoard:', epoch)
        if epoch==0:
            tb_path = './runs/' + self.model_name_save_dir
            print('tb_path', tb_path)
            if os.path.isdir(tb_path):
                shutil.rmtree(tb_path)

            self.writer['train'] = tb.SummaryWriter(log_dir=tb_path+'/train')
            self.writer['val'] = tb.SummaryWriter(log_dir=tb_path + '/val')
            sample_data = iter(self.trainloader).next()[0]  # [batch_size X seq_length X embedding_dim]
            print(sample_data.shape)
            self.writer['train'].add_graph(self.model, sample_data.to(self.device))
            self.writer['train'].add_text('Model:', str(self.model))
            self.writer['train'].add_text('Input shape:', str(x_train.shape))
            self.writer['train'].add_text('Data Preprocessing:', 'None, One-hot')
            self.writer['train'].add_text('Optimiser', str(self.optimizer))
            self.writer['train'].add_text('Batch Size:', str(self.config['DATA']['BATCH_SIZE']))
            self.writer['train'].add_text('Epochs:', str(self.config['TRAINER']['epochs']))

        for measure, value in self.metrics['train'].items():
            self.writer['train'].add_scalar(str('Train/'+measure), value, epoch)
        self.writer['train'].add_scalar('Loss', train_loss, epoch)
        for measure, value in self.metrics['val'].items():
            self.writer['val'].add_scalar(str('Val/'+measure), value, epoch)
        self.writer['val'].add_scalar('Loss', val_loss, epoch)


    def train_one_epoch(self, epoch, x_train, y_train):
        '''
        Train one epoch
        :param epoch: int - epoch number
        :param x_train: Numpy array - training data (sequences)
        :param y_train: Numpy array - training data (labels)
        :return: float - avg_train_loss
        '''

        # INPUT DATA
        trainset = SequenceDataset(x_train[0:10], y_train[0:10])  # NOTE: change input dataset size here if required
        self.trainloader = torch.utils.data.DataLoader(
                        trainset, batch_size=self.config['DATA']['BATCH_SIZE'],
                        shuffle=self.config['DATA']['SHUFFLE'], num_workers=self.config['DATA']['NUM_WORKERS'])

        # MODEL
        self.model = eval(self.config['MODEL_NAME'])(self.config['MODEL']['embedding_dim'], self.config['MODEL']['hidden_dim'],
                                                self.config['MODEL']['hidden_layers'], self.config['MODEL']['output_dim'],
                                                self.config['DATA']['BATCH_SIZE'], self.device)
        self.model.to(self.device)

        # LOSS FUNCTION
        loss_fn = getattr(nn, self.config['LOSS'])()  # For eg: nn.CrossEntropyLoss()

        # OPTIMISER  #todo: read from config file
        # optimiser = optim.SGD(model.parameters(), momentum=0.9, lr=0.001)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.1)

        avg_train_loss = 0
        m = Metrics(self.config['DATASET_TYPE'])   #m.metrics initialised to {0,0,0}
        self.metrics['train'] = m.metrics

        # FOR EACH BATCH
        for bnum, sample in tqdm(enumerate(self.trainloader)):
            self.model.train()
            self.model.zero_grad()
            print('Train batch: ', bnum)
            raw_out = self.model.forward(sample[0].to(self.device))
            loss = loss_fn(raw_out, sample[1].long().to(self.device))
            #print('Loss: ', loss)
            loss.backward()
            self.optimizer.step()

            # EVALUATION METRICS PER BATCH
            metrics_for_batch = m.get_metrics(raw_out.detach().clone(), sample[1].detach().clone(), 'macro')  # todo: understand 'macro'
            for key,value in metrics_for_batch.items():
                self.metrics['train'][key] += value
            avg_train_loss += loss.item()

        # EVALUATION METRICS PER EPOCH
        for measure in m.metrics:
            self.metrics['train'][measure] /= (bnum+1)

        print('Epoch: {:d}, Train Loss: {:.4f}, '.format(epoch, avg_train_loss), self.metrics['train'])
        return avg_train_loss


    def val_one_epoch(self, epoch, x_val, y_val):
        '''
        Validation loop for epoch
        :param epoch: int - epoch number
        :param x_val: Numpy array - validation data (sequences)
        :param y_val: Numpy array - validation data (labels)
        :return: int - avg_val_loss
        '''

        valset = SequenceDataset(x_val, y_val)  # NOTE: change input dataset size here if required todo:
        val_dataloader = torch.utils.data.DataLoader(
                    valset, batch_size=self.config['DATA']['BATCH_SIZE'],
                    shuffle=self.config['DATA']['SHUFFLE'], num_workers=self.config['DATA']['NUM_WORKERS'])

        m = Metrics(self.config['DATASET_TYPE'])  # m.metrics initialised to {0,0,0}
        self.metrics['val'] = m.metrics
        loss_fn =  getattr(nn, self.config['LOSS'])()
        avg_val_loss = 0

        for bnum, sample in enumerate(val_dataloader):
            print('Val batch: ', bnum)
            self.model.eval()
            raw_out = self.model.forward(sample[0].to(self.device))
            loss = loss_fn(raw_out, sample[1].long().to(self.device))

            # EVALUATION METRICS PER BATCH
            metrics_for_batch = m.get_metrics(raw_out.detach().clone(), sample[1].detach().clone(), 'macro')
            for key, value in metrics_for_batch.items():
                self.metrics['val'][key] += value
            avg_val_loss += loss.item()

        for measure in m.metrics:
            self.metrics['val'][measure] /= (bnum+1)

        print('Epoch: {:d}, Valid Loss: {:.4f}, '.format(epoch, avg_val_loss), self.metrics['val'])
        return avg_val_loss


    def training_pipeline(self):
        #Todo:For loading state, self.start epoch would change

        encoded_seq = np.loadtxt(self.data_path + '/encoded_seq')
        no_timesteps = int(len(encoded_seq[0]) / 4)
        encoded_seq = encoded_seq.reshape(-1, no_timesteps, 4)
        print("Input data shape: ", encoded_seq.shape)
        y_label = np.loadtxt(self.data_path + '/y_label_start')

        if self.config['VALIDATION']:
            train_idx, val_idx = create_train_val_split(self.config['DATA']['VALIDATION_SPLIT'], n_samples=len(encoded_seq))

            # Create train/validation split --
            x_train = encoded_seq[np.ix_(train_idx)] #replace `train_idx` by `np.arange(len(encoded_seq))` to use whole dataset
            y_train = y_label[np.ix_(train_idx)]
            x_val = encoded_seq[np.ix_(val_idx)]
            y_val = y_label[np.ix_(val_idx)]
        else:
            x_train = encoded_seq
            y_train = y_label

        print(get_class_dist(y_train, 'train'))
        print(get_class_dist(y_val, 'val'))

        for epoch in range(self.start_epoch, self.config['TRAINER']['epochs']):
            print("Training Epoch %i -------------------" % epoch)

            epoch_tic = time.time()
            train_loss = self.train_one_epoch(epoch, x_train, y_train)
            if self.config['VALIDATION']:
                val_loss = self.val_one_epoch(epoch, x_val, y_val)
            epoch_toc = time.time()
            epoch_time = epoch_toc - epoch_tic
            print("******************* Epoch %i completed in %i seconds ********************" % (epoch, epoch_time))

            # Writing to be done in the first epoch
            if self.config['TRAINER']["save_model_to_dir"]:
                if epoch==0:
                    self.write_model_meta_data(y_train, y_val)

                # SAVE TRAINING DETAILS TO INFO LOG
                self.write_model_loss_metrics(epoch, train_loss, 'train')
                if self.config['VALIDATION']:
                    self.write_model_loss_metrics(epoch, val_loss, 'val')

                # SAVE TO CHECKPOINT TO DIRECTORY
                if epoch % self.config['TRAINER']['save_period'] == 0:
                    self.save_checkpoint(epoch)

            # TENSORBOARD LOGGING
            if self.config['TRAINER']['tensorboard']:
                if not self.config['VALIDATION']:
                    val_loss = 0.0
                self.logger(epoch, x_train, train_loss, val_loss)
                # write to runs folder (create a model file name, and write the various training runs in it

        # SAVE MODEL TO DIRECTORY
        if self.config['TRAINER']["save_model_to_dir"]:
            print('Saving model at ', self.save_dir)
            torch.save(self.model, self.save_dir+'/trained_model_'+self.model_name_save_dir)

        if self.config['TRAINER']['tensorboard']:
            self.writer['train'].close()
            self.writer['train'].close()

if __name__ == "__main__":

    chrm =  "chrm21/"

    # Get config file
    with open(curr_dir_path + "/config.json", encoding='utf-8', errors='ignore') as json_data:
        config = json.load(json_data, strict=False)

    final_data_path = data_path+chrm+config['DATASET_TYPE']+config["DATA"]["DATA_DIR"]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('_%d-%m_%H:%M')
    saved_model_folder = string_metadata(config) + timestamp
    save_dir_path = curr_dir_path + config['TRAINER']['save_dir'] + '/'+ saved_model_folder

    obj = Training(config,  saved_model_folder, final_data_path, save_dir_path)
    obj.training_pipeline()
