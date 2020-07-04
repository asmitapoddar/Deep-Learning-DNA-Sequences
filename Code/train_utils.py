import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter


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
    s = config['EXP_NAME'] + '_' + config['MODEL_NAME'] + '['
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

def create_train_val_split_mixed(split, n_samples):
    '''
    Create train/val data splits such that the sequences from the same exon/intron/containing same boundary
    are mixed between the train and val set
    :param split: float: proportion of the dataset to be used for validation
    :param n_samples: int: total length of dataset
    :return: 2 list of int
        train_idx - indices to be used to splice out the train set
        valid_idx - indices to be used to splice out the val set
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

def create_train_val_split_separate(split, n_samples):
    '''
    Create train/val data splits such that the sequences from the same exon/intron/containing same boundary
    are NOT mixed between the train and val set
    :param split: float: proportion of the dataset to be used for validation
    :param n_samples: int: total length of dataset
    :return: 2 list of int
        train_idx - indices to be used to splice out the train set
        valid_idx - indices to be used to splice out the val set
    '''
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


