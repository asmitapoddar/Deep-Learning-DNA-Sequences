import numpy as np
import pathlib
import matplotlib.pyplot as plt

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/chrm21/regression/cds_start_n875_l100"
graphs_path = curr_dir_path + "/Graphs/"
print(graphs_path)

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

exon_postion()