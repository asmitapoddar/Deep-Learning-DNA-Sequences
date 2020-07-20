import tqdm
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

h_batch_size = [32, 64, 128]
h_hidden_dims = [8, 16, 32, 64, 128]
h_num_layers = [1, 2, 3]
h_lr = [0.001, 0.0001]
h_dropout = [0.0, 0.3, 0.5]

h_batch_size = [32]
h_hidden_dims = [8]
h_num_layers = [1]
h_lr = [0.0011]
h_dropout = [0.0]

chrm =  "all/"

# Get config file
with open(curr_dir_path + "/config.json", encoding='utf-8', errors='ignore') as json_data:
    config = json.load(json_data, strict=False)

#Get System-specific Params file
with open('system_specific_params.yaml', 'r') as params_file:
    sys_params = yaml.load(params_file)

hyperparameter_df = pd.DataFrame(columns = ['Model Name','Batch Size','Hidden dim','No. Layers', 'LR', 'Dropout',
                                            'Val Loss', 'Val Accuracy', 'Val F1', 'Val Prec', 'Val Recall',
                                            'Train Loss', 'Train Accuracy', 'Train F1', 'Train Prec', 'Train Recall'])
hyperparameter_df.to_csv('my_csv.csv', mode='a', header=True)

for bs in h_batch_size:
    for hd in h_hidden_dims:
        for nl in h_num_layers:
            for lr in h_lr:
                for dropout in h_dropout:
                    #Create required config file
                    config['DATA']['BATCH_SIZE'] = bs
                    config['MODEL']['hidden_dim'] = hd
                    config['MODEL']['hidden_layers'] = nl
                    config['OPTIMIZER']['lr'] = lr
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
                    obj = Training(config, model_name_save_dir, final_data_path, save_dir_path, tb_path)
                    obj.training_pipeline()

                    #Write hyper-parameters and results to csv file
                    train_met = obj.best_metrics['train']
                    val_met = obj.best_metrics['val']
                    data_dict = {'Model Name': model_name_save_dir, 'Batch Size': bs,'Hidden dim': hd,'No. Layers': nl,
                                 'LR':lr, 'Dropout':dropout, 'Val Loss': val_met['loss'], 'Val Accuracy': val_met['acc'],
                                 'Val F1': val_met['f1'], 'Val Prec': val_met['prec'], 'Val Recall': val_met['recall'],
                                'Train Loss': train_met['loss'], 'Train Accuracy': train_met['acc'],
                                 'Train F1': train_met['f1'], 'Train Prec': train_met['prec'], 'Train Recall':
                                train_met['recall']}
                    pd.DataFrame(data_dict).to_csv('my_csv.csv', mode='a', header=False)
