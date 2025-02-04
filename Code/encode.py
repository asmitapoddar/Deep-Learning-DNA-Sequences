import numpy as np
import pathlib
import tqdm
import argparse

def encode(seq):

    encoded_seq = np.zeros(len(seq)*4,int)
    for j in range(len(seq)):
        if seq[j] == 'A' or seq[j] == 'a':
            encoded_seq[j*4] = 1
            encoded_seq[j*4+1] = 0
            encoded_seq[j*4+2] = 0
            encoded_seq[j*4+3] = 0

        elif seq[j] == 'C' or seq[j] == 'c':
            encoded_seq[j*4] = 0
            encoded_seq[j*4+1] = 1
            encoded_seq[j*4+2] = 0
            encoded_seq[j*4+3] = 0

        elif seq[j] == 'G' or seq[j] == 'g':
            encoded_seq[j*4] = 0
            encoded_seq[j*4+1] = 0
            encoded_seq[j*4+2] = 1
            encoded_seq[j*4+3] = 0

        elif seq[j] == 'T' or seq[j] == 't':
            encoded_seq[j*4] = 0
            encoded_seq[j*4+1] = 0
            encoded_seq[j*4+2] = 0
            encoded_seq[j*4+3] = 1

        else:
            encoded_seq[j*4] = 0
            encoded_seq[j*4+1] = 0
            encoded_seq[j*4+2] = 0
            encoded_seq[j*4+3] = 0

    return encoded_seq

def encode_seq(write_path, in_path=''):
    if in_path=='sub':
        encoded_f = open(write_path+'/encoded_seq_sub', 'w')
        seq_file = open(write_path + '/dna_seq_sub').read().splitlines()
    else:
        encoded_f = open(write_path + '/encoded_seq', 'w')
        seq_file = open(write_path+'/dna_seq').read().splitlines()

    print('Encoding dna seq...')
    for line in tqdm.tqdm(seq_file):
        seq = line
        encoded_seq = encode(seq)
        for base in encoded_seq:
            encoded_f.write(str(base))
            encoded_f.write("   ")
        encoded_f.write("\n")

    encoded_f.close()

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Create one-hot encoding for DNA sequence')
    parser.add_argument('--in_path', type=str, help='input_path for encoded dna seq')
    parser.add_argument('--write_path', type=str, help='write_path for encoded dna seq')
    args = parser.parse_args()

    encode_seq(write_path=args.write_path, in_path = args.in_path)
