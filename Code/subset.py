import pandas as pd
import numpy as np
import argparse
from encode import *

parser = argparse.ArgumentParser(description='Get subset of a dataset')
parser.add_argument('-i', '--IN_FILE_PATH', type=str, help='full dataset base path')
parser.add_argument('-n', '--NO_SAMPLES', type=int, help='how many samples to subset')
args = parser.parse_args()

input_dna = pd.read_csv(args.IN_FILE_PATH+'dna_seq_start', header=None)
input_label = pd.read_csv(args.IN_FILE_PATH+'y_label_start', header=None)
n_samples = len(input_dna)
idx_full = np.arange(n_samples)
np.random.seed(0)  # For reproducibility
np.random.shuffle(idx_full)
subset_samples = idx_full[0:args.NO_SAMPLES]
output_dna = input_dna.iloc[subset_samples,:]
output_label = input_label.iloc[subset_samples,:]
output_dna.to_csv(args.IN_FILE_PATH+'dna_seq_start_sub', header=False, index=False)
output_label.to_csv(args.IN_FILE_PATH+'y_label_start_sub', header=False, index=False)

