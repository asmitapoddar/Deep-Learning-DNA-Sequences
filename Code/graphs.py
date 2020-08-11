import numpy as np
import pandas as pd
import yaml
import pathlib
import logomaker
import matplotlib.pyplot as plt
from collections import Counter

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/chrm21/regression/cds_start_n875_l100"
graphs_path = curr_dir_path + "/Graphs/"
print(graphs_path)

with open('system_specific_params.yaml', 'r') as params_file:
    sys_params = yaml.load(params_file)
final_data_path = sys_params['DATA_WRITE_FOLDER']+'/all/boundaryCertainPoint_orNot_2classification/end/ok'

def exon_postion():
    y_train = np.loadtxt(data_path + '/y_label_start')
    print(str(y_train.shape[0]))
    y_train_without_0 = y_train[y_train != 0]
    #ax = plt.subplot(1, 2, 2)
    #plt.hist(y_train, bins=350)
    plt.hist(y_train, bins = 100)
    plt.xlabel('Exon Boundary relative to beginning of seq (intronic region)')
    plt.ylabel('# Sequences')
    plt.title('Exon Start Boundaries')
    plt.text(20, 70, 'Total num. sequences = ' +str(y_train.shape[0]))
    plt.savefig(graphs_path+'regression-cds_start_n875_l100.png')  #todo: rename properly automatically
    plt.show()

def accuracy_vs_length():
    lengths = [10,20,30,40,50,60,70,80,90,100]
    accuracy_64 = [54.6,54.5,56.2,54.3,53.84,53.8,53,52.8,52.2,52.14]
    accuracy_128 = [53.57,54.25,55.36,55.5,54.71,54.93,54.20,53.93,53.90,53.63]
    plt.xlabel('Length of DNA seq')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of 2-class (boundary/no boundary) classification vs Length Plot')
    plt.plot(lengths, accuracy_64, color='red', label='hidden dim: 64')
    #plt.plot(lengths, accuracy_128, color='blue', label='hidden dim: 128')
    plt.legend()
    plt.savefig(graphs_path + 'Accuracy vs Length Plot End.png')
    plt.show()

def sequence_logo(boundary):
    rem_path = '/all/boundaryCertainPoint_orNot_2classification/'
    final_data_path = sys_params['DATA_WRITE_FOLDER']+rem_path+str(boundary)+'_bound_half'
    f = open(final_data_path + '/dna_seq_start', "r").readlines()
    f = list(map(lambda x: x.strip('\n'), f))
    df = pd.DataFrame(columns=['A', 'C', 'T', 'G'])
    print('Running logo maker for {}'.format(final_data_path))
    for i in range(0,len(f[0])):
        letters_at_pos_i = [x[i].upper() for x in f]
        counts = Counter(letters_at_pos_i)
        prob_A = counts['A']/len(letters_at_pos_i)
        prob_C = counts['C'] / len(letters_at_pos_i)
        prob_T = counts['T'] / len(letters_at_pos_i)
        prob_G = counts['G'] / len(letters_at_pos_i)
        # convert probabilities to data frame
        dict_probs = {'A': prob_A, 'C': prob_C, 'T': prob_T, 'G': prob_G}
        df = df.append(dict_probs, ignore_index=True)

    # create Logo object
    ss_logo = logomaker.Logo(df, width=.8, vpad=.05, fade_probabilities=False, stack_order='small_on_top', color_scheme='classic')

    # style using Logo methods
    ss_logo.style_spines(spines=['left', 'right'], visible=False)

    # style using Axes methods
    if(len(df)<50):
        ss_logo.ax.set_xticks(range(len(df)))
    elif(len(df)>70):
        ss_logo.ax.xaxis.set_tick_params(labelsize=5)
        ss_logo.ax.set_xticks(range(len(df)))
    else:
        ss_logo.ax.xaxis.set_tick_params(labelsize=7)
        ss_logo.ax.set_xticks(range(len(df)))
    ss_logo.ax.set_xticklabels('%d' % x for x in range(len(df)))
    ss_logo.ax.set_yticks([0, .5, 1])
    ss_logo.ax.axvline(len(df)/2, color='k', linewidth=1, linestyle=':')  # marking boundary position
    ss_logo.ax.set_ylabel('probability')

    ss_logo.fig.savefig(sys_params['LOGS_BASE_FOLDER']+'/final_half/logo{}.png'.format(boundary))
    print('Done.')

#exon_postion()
#accuracy_vs_length()
for i in range(10,101,10):
    sequence_logo(i)