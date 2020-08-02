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

def accuracy_vs_length():
    lengths = [10,20,30,40,50,60,70,80,90,100]
    accuracy_64 = [54.25,54.64,55.4,55.9,54.84,54.69,54.68,54.48,54.12,53.7]
    accuracy_128 = [53.71,52.82,55.13,54.77,54.67,54.88,52.53,54.25,52.70,54.61]
    plt.xlabel('Length of DNA seq')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of 2-class (boundary/no boundary) classification vs Length Plot')
    plt.plot(lengths, accuracy_64, color='red', label='hidden dim: 64')
    #plt.plot(lengths, accuracy_128, color='blue', label='hidden dim: 128')
    plt.legend()
    plt.savefig(graphs_path + 'Accuracy vs Length Plot Test 1')
    plt.show()


#exon_postion()
accuracy_vs_length()