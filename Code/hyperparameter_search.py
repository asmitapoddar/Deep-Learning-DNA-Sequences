import tqdm
import itertools
import random
import os
import shutil
import numpy as np
import pandas as pd
from models import *
from metrics import *
import time
import datetime
import json
import pathlib
import time
from train_utils import *
from train import *

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"
NO_MODELS = 2
FILE_NAME = 'boundary60_orNot_2classification.csv'

# Get config file
with open(curr_dir_path + "/config.json", encoding='utf-8', errors='ignore') as json_data:
    config = json.load(json_data, strict=False)
#Get System-specific Params file
with open('system_specific_params.yaml', 'r') as params_file:
    sys_params = yaml.load(params_file)

h_batch_size = [32, 64, 128, 256]
h_hidden_dims = [8, 16, 32, 64, 128]
h_num_layers = [1, 2, 3]
h_lr = [0.1, 0.01, 0.001, 0.0001]
h_dropout = [0.0, 0.3, 0.5]
list_hyperparam = [h_batch_size, h_hidden_dims, h_num_layers, h_lr, h_dropout]
cartesian_prod = list(itertools.product(*list_hyperparam))
random.shuffle(cartesian_prod)

chrm =  "all/"

hyperparameter_df = pd.DataFrame(columns = ['Model Name','Batch Size','Hidden dim','No. Layers', 'LR', 'Dropout',
                                            'Val Loss', 'Val Accuracy', 'Val F1', 'Val Prec', 'Val Recall',
                                            'Train Loss', 'Train Accuracy', 'Train F1', 'Train Prec', 'Train Recall'])
full_hyperparam_file_write_path = sys_params['HYPER_BASE_FOLDER'] + '/' + FILE_NAME
hyperparameter_df.to_csv(full_hyperparam_file_write_path, mode='a', header=True)

for i in range(0,NO_MODELS):
    print('Building Model {} ...'.format(i))
    hyper = cartesian_prod[i]
    bs = hyper[0]; hd = hyper[1]; nl = hyper[2]; lr = hyper[3]; dropout = hyper[4]

    #Create required config file
    config['DATA']['BATCH_SIZE'] = bs
    config['MODEL']['hidden_dim'] = hd
    config['MODEL']['hidden_layers'] = hyper[2]
    config['OPTIMIZER']['lr'] = nl
    config['TRAINER']['dropout'] = dropout

    # Set paths
    final_data_path = sys_params['DATA_WRITE_FOLDER'] + '/' + chrm + config['DATASET_TYPE'] + \
                      config["DATA"]["DATA_DIR"]  # config["DATA"]["DATA_DIR"] = 60
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('_%d-%m_%H:%M')
    model_name_save_dir = string_metadata(config) + timestamp

    save_dir_path = sys_params['HYPER_BASE_FOLDER'] + '/saved_models/' + config[
        'DATASET_TYPE'] + '/' + config["DATA"]["DATA_DIR"] + '/' + model_name_save_dir
    tb_path = sys_params['HYPER_BASE_FOLDER'] + '/runs/' + config['DATASET_TYPE'] + \
              '/' + config["DATA"]["DATA_DIR"] + '/' + model_name_save_dir

    config['TRAINER']['save_dir'] = save_dir_path
    config['TRAINER']['tb_path'] = tb_path

    # Train model
    print('Start Training Model: {}...'.format(model_name_save_dir))
    obj = Training(config, model_name_save_dir, final_data_path, save_dir_path, tb_path)
    obj.training_pipeline()

    #Write hyper-parameters and results to csv file
    print('Writing hyper-parameters and results file "{}"'.format(full_hyperparam_file_write_path))
    train_met = obj.best_metrics['train']
    val_met = obj.best_metrics['val']
    data_dict = {'Model Name': model_name_save_dir, 'Batch Size': bs,'Hidden dim': hd,'No. Layers': nl,
                 'LR':lr, 'Dropout':dropout, 'Val Loss': val_met['loss'], 'Val Accuracy': val_met['acc'],
                 'Val F1': val_met['f1'], 'Val Prec': val_met['prec'], 'Val Recall': val_met['recall'],
                'Train Loss': train_met['loss'], 'Train Accuracy': train_met['acc'],
                 'Train F1': train_met['f1'], 'Train Prec': train_met['prec'], 'Train Recall':
                train_met['recall']}
    pd.DataFrame(data_dict, index=[0]).to_csv(full_hyperparam_file_write_path, mode='a', header=False)
