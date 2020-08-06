import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

import torch
from test_model import *

chrm = 'all'
dataset_type = 'boundaryCertainPoint_orNot_2classification'
DATA_LEN = 10  #Note: change
n_nt = 5
MODEL_DIR = 'Len10_bound_half _balanced_AttLSTM[4,64,2,2]_BS32_Adam_04-08_15:50' #Note: change
MODEL_CLASS = 'att_DNA'
unpermuted_accuracy = 0.5882

#Get System-specific Params file
with open('system_specific_params.yaml', 'r') as params_file:
    sys_params = yaml.load(params_file)
with open(sys_params['LOGS_BASE_FOLDER']+ '/final_half/'+MODEL_DIR + "/config.json", encoding='utf-8', errors='ignore') as json_data:
    config = json.load(json_data, strict=False)

final_data_path = sys_params['DATA_WRITE_FOLDER']+'/'+chrm+'/'+dataset_type+'/'+str(DATA_LEN)+'_bound_half'
encoded_seq = np.loadtxt(final_data_path + '/encoded_seq_sub')
no_timesteps = int(len(encoded_seq[0]) / 4)
encoded_seq = encoded_seq.reshape(-1, no_timesteps, 4)
print("Input data shape: ", encoded_seq.shape)
y_label = np.loadtxt(final_data_path + '/y_label_start_sub')
model_name = sys_params['LOGS_BASE_FOLDER']+ '/final_half/'+MODEL_DIR+'/best_checkpoint.pth'

np.random.seed(0)
all_metrics = []
noise_vector = np.random.normal(0, 1, [encoded_seq.shape[0], n_nt , 4])
ranges = []
for i in range(0,no_timesteps,n_nt ):
    print('Permuting sequence in range [{},{}]...'.format(i, i+n_nt))
    perm_encoded_seq = np.copy(encoded_seq)
    ranges.append((i, i+n_nt))
    perm_encoded_seq[:,i:i+n_nt,:] = noise_vector  #add noise to input
    print('Running sequence through model...')
    metrics = test(perm_encoded_seq, y_label, model_name, MODEL_CLASS, config)  # feed data to model - get accuracy
    all_metrics.append(metrics['acc'])   # store metric vales

# plot - make sequence plot
print(ranges)
print(all_metrics)
x = list(range(0,len(all_metrics)))
plt.plot(x, all_metrics, label='Accuracy:\npermuted seq')
plt.plot(x,[unpermuted_accuracy]*len(all_metrics),color='r', label='Accuracy:\noriginal seq')
plt.axvline(x=int(len(ranges)/2)-0.5, linestyle=':', color='green', label='Boundary position')

plt.xticks(ticks=x, labels=ranges)  #fontsize=10
plt.xlabel("DNA Sequence Positions")
plt.ylabel("Accuracy")
plt.title('Sequence Perturbation Test')
plt.legend(loc='lower right', bbox_to_anchor=(1, 0.4))  #, bbox_to_anchor=(1, 0.5) , fontsize='small'
print('Saving figure at: {}'.format(sys_params['LOGS_BASE_FOLDER']+ '/final_half/' + 'perm{}.png'.format(DATA_LEN)))
plt.savefig(sys_params['LOGS_BASE_FOLDER']+ '/final_half/' + 'perm{}.png'.format(DATA_LEN))
plt.show()