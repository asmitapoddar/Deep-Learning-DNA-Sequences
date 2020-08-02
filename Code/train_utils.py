import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class SequenceDataset(Dataset):

    def __init__(self, data, labels):

        self.data = torch.from_numpy(data).float()
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data (seq_len, batch, input_dim), label for index
        return (self.data[idx], self.labels[idx])

def string_metadata(config):
    '''
    Model name which will be stored (as a string)
    :param config: json
            config file containing the experiment details
    :return: str
            name of the folder which will be stored in '/saved_models/' for this experiment
    '''
    s = config['EXP_NAME'] + '_' + config['VALIDATION']['type'] + '_' + config['MODEL_NAME'] + '['
    s += str(config['MODEL']['embedding_dim']) + ',' + str(config['MODEL']['hidden_dim']) + ',' + \
         str(config['MODEL']['hidden_layers']) + ',' + str(config['MODEL']['output_dim']) + ']'
    s += '_BS' + str(config['DATA']['BATCH_SIZE']) + '_' + config['OPTIMIZER']['type']
    return s

def create_train_val_split(split, n_samples):
    '''

    :param split:
    :param n_samples:
    :return:
    '''
    idx_full = np.arange(n_samples)
    if split == 0.0:
        return idx_full, None

    np.random.seed(0)
    np.random.shuffle(idx_full)

    if isinstance(split, int):
        assert split > 0
        assert split < n_samples, "validation set size is configured to be larger than entire dataset; " \
                                  "entire dataset will be used for validation"
        len_valid = split
    else:
        len_valid = int(n_samples * split)

    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))
    return train_idx, valid_idx

def create_train_val_split_mixed(split, y):
    '''
    Create train/val data splits such that the sequences from the same exon/intron/containing same boundary
    are mixed between the train and val set
    :param split: float: proportion of the dataset to be used for validation
    :param n_samples: int: total length of dataset
    :return: 2 list of int
        train_idx - indices to be used to splice out the train set
        valid_idx - indices to be used to splice out the val set
    '''
    n_samples = len(y)
    idx_full = np.arange(n_samples)
    if split == 0.0:
        return idx_full, None

    np.random.seed(0)  # For reproducibility
    np.random.shuffle(idx_full)

    if isinstance(split, int):
        assert split > 0
        assert split < n_samples, "validation set size is configured to be larger than entire dataset; " \
                                  "entire dataset will be used for validation"
        len_valid = split
    else:
        len_valid = int(n_samples * split)

    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))
    return train_idx, valid_idx

def create_train_val_split_separate(split, y):
    '''
    Create train/val data splits such that the sequences from the same exon/intron/containing same boundary
    are NOT mixed between the train and val set
    :param split: float: proportion of the dataset to be used for validation
    :param n_samples: int: total length of dataset
    :return: 2 list of int
        train_idx - indices to be used to splice out the train set
        valid_idx - indices to be used to splice out the val set
    '''
    n_samples = len(y)
    idx_full = np.arange(n_samples)
    if split == 0.0:
        return idx_full, None

    if isinstance(split, int):
        assert split > 0
        assert split < n_samples, "validation set size is configured to be larger than entire dataset; " \
                                  "entire dataset will be used for validation"
        len_valid = split
    else:
        len_valid = int(n_samples * split)

    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))
    return train_idx, valid_idx


def create_train_val_split_mixed_Kfold(n_samples, k, K):
    '''
    Create train/val data splits such that the sequences from the same exon/intron/containing same boundary
    are mixed between the train and val set
    :param k: int: current validation fold (not used in this function)
    :param K: int: Total number of validation folds
    :param n_samples: int: total length of dataset
    :return: 2 list of int
        train_idx - indices to be used to splice out the train set
        valid_idx - indices to be used to splice out the val set
    '''
    idx_full = np.arange(n_samples)
    np.random.seed(0)  # For reproducibility
    np.random.shuffle(idx_full)
    len_valid = int(n_samples * 1/K)

    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))
    return train_idx, valid_idx

def create_train_val_split_separate_Kfold(n_samples, k, K):
    '''
    Create train/val data splits such that the sequences from the same exon/intron/containing same boundary
    are NOT mixed between the train and val set
    :param n_samples: int: total length of dataset
    :param k: int: current validation fold
    :param K: int: Total number of validation folds
    :return: 2 list of int
        train_idx - indices to be used to splice out the train set
        valid_idx - indices to be used to splice out the val set
    '''
    idx_full = np.arange(n_samples)
    len_valid = int(n_samples * k / K)
    split_size = int(n_samples * 1 / K)

    valid_idx = idx_full[len_valid-split_size:len_valid]
    train_idx = [e for e in idx_full if e not in valid_idx]

    return train_idx, valid_idx

def create_train_val_split_balanced(test_split, y, type='downsample', shuffle=True):
    '''
    Create a balanced dataset
    :param y: list of int: y_labels
    :param test_split: float: proportion of the dataset to be used for validation
    :param type: 'upsample'/'downsample' (default = 'downsample')
    :param shuffle:
    :return: 2 lists of int
        trainIndexes - indices to be used to splice out the train set
        testIndexes - indices to be used to splice out the val set
    '''
    classes = np.unique(y)

    if type=='upsample':
        # can give test_size as fraction of input data size of number of samples
        if test_split < 1:
            n_test = np.round(len(y) * test_split)
        else:
            n_test = test_split
        n_train = max(0, len(y) - n_test)
        n_train_per_class = max(1, int(np.floor(n_train / len(classes))))
        n_test_per_class = max(1, int(np.floor(n_test / len(classes))))

        ixs = []
        for cl in classes:
            if (n_train_per_class + n_test_per_class) > np.sum(y == cl):
                # if data has too few samples for this class, do upsampling
                # split data to train and test before sampling so data points won't be shared among train & test data
                splitix = int(np.ceil(n_train_per_class / (n_train_per_class + n_test_per_class) * np.sum(y == cl)))
                ixs.append(np.r_[np.random.choice(np.nonzero(y == cl)[0][:splitix], n_train_per_class),
                                 np.random.choice(np.nonzero(y == cl)[0][splitix:], n_test_per_class)])
            else:
                ixs.append(np.random.choice(np.nonzero(y == cl)[0], n_train_per_class + n_test_per_class,
                                            replace=False))

        # take same num of samples from all classes
        trainIndexes = np.concatenate([x[:n_train_per_class] for x in ixs])
        testIndexes = np.concatenate([x[n_train_per_class:(n_train_per_class + n_test_per_class)] for x in ixs])

    if type=='downsample':
        least_common_class = Counter(y).most_common()[-1][0]
        least_common_class_count = Counter(y).most_common()[-1][1]
        ixs = []
        np.random.seed(0)
        for cl in classes:
            if cl==least_common_class:
                ixs.extend(np.where(y == cl)[0])
            else:
                # downsample to the size of the least common class
                ixs.extend(np.random.choice(np.nonzero(y == cl)[0], least_common_class_count))
        downsampled_y = [y[i] for i in ixs]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=0)
        indices = list(splitter.split(X=np.zeros(len(downsampled_y)), y=downsampled_y))[0]
        trainIndexes_of_downsampled = list(indices[0])
        testIndexes_of_downsampled = list(indices[1])
        # get mapping of downsampled indices to original indices
        trainIndexes = [ixs[i] for i in trainIndexes_of_downsampled]
        testIndexes = [ixs[i] for i in testIndexes_of_downsampled]
    return trainIndexes, testIndexes


def get_class_dist(classes, dataset_type):
    '''
    Get ditribution of classes in data set
    :param classes: list - list of labels
    :param dataset_type: str - 'train'/'val'
    :return: str - string containing the class dist. for the dataset
    '''
    class_count = Counter(classes)
    s = 'Dist. of classes for {:s} set: {}'.format(dataset_type, class_count)
    return s

def check_output_dim(config, y_label):
    '''
    Assert that the output size of model is correct for the dataset being used
    :param config: Training configuration file
    :param y_label: numpy array containing the labels
    :return: None
    '''
    class_count = Counter(y_label)
    assert config['MODEL']['output_dim'] == len(class_count), \
        "`{}`class classification; specify in config file".format(len(class_count))

def get_weight_tensor(y_label, device):
    '''
    Inverse class count sampling to obtain weight tensor for CrossEntropy Loss
    '''
    unique_classes = sorted(set(y_label))
    weights = []
    for l in unique_classes:
        weights.append(len(y_label)/list(y_label).count(l))
    class_weights = torch.FloatTensor(weights).cuda(device)
    return class_weights



