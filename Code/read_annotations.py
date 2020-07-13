import numpy as np
import pathlib
import pandas as pd
import os
import argparse

curr_dir_path = str(pathlib.Path().absolute())
raw_data_path = curr_dir_path + "/raw_data/"

def preprocess_chromosomeseq_file(chr_file="chr21.fa", base_write_path='', write_to_file=False):
    '''
    Function to remove special characters in chromosome sequence
    :param chr_file: File containing nucleotide sequence for chromosome
    :param base_write_path: str
            base path for writing to file
    :param write_to_file: bool, optional
            Variable to control writing extracted information to txt file
    :return: str
            String containing nucleotide sequence for chromosome
    '''
    file_object = open(raw_data_path + chr_file, "r")

    chrm_seq = file_object.read()
    # print('Length of unprocessed chromosome: '+len(chrm))  # 47644190

    chrm_seq = chrm_seq.replace('\n', '')
    #Fasta file contains name of chromosome (eg. '>chr21') at the start: removing it
    if (chrm_seq[5].isdigit()):
        chrm_seq = chrm_seq[6:]
    else:
        chrm_seq = chrm_seq[5:]
    # print('Length of pre-processed chromosome: ', len(chrm))  # 46709983  #write to file

    #Append a blank at the beginning and end of string (since annotations are 1-indexed
    chrm_seq = ' '+chrm_seq+' '   # len = 46709985

    #print('Length of processed chromosome: ', len(chrm_seq))
    if write_to_file or not os.path.exists(base_write_path+chr_file.replace("fa", "txt")):
        print('Writing chromosome seq to txt file..')
        text_file = open(base_write_path + chr_file.replace("fa", "txt"), "w")
        text_file.write(chrm_seq)
        text_file.close()
    return chrm_seq


def create_chromosome_annotations(chrm='chr21', base_write_path='', write_to_file=False):
    '''
    Function to read annotation file for human genome sequence GRCh38 and get annotations for specified chromosome
    :param chrm: str
            chromosome information to be extracted
    :param base_write_path: str
            base path for writing to file
    :param write_to_file: bool, optional
            Variable to control writing extracted information to file
    :return: None
    '''
    ann_file = "gencode.v34.annotation.gtf"
    col_names = ['chr_name', 'source', 'type', 'start', 'end', '.', 'strand', ',', 'other']
    assert os.path.exists(raw_data_path + ann_file), "place genome annotation file in correct directory"
    print('Reading Annotation File {}...'.format(ann_file))
    ann = pd.read_csv(raw_data_path + ann_file, sep='\t', header=None)
    print('Done.')
    ann.columns = col_names    # print(ann.shape)  # (2912496, 9)

    chr_no = ann[ann['chr_name'] == chrm]
    del chr_no['.']
    del chr_no[',']

    l = list(map(lambda x: x.split(';'), chr_no['other']))
    df_l = pd.DataFrame(l)
    df_l[0] = list(map(lambda x: x[8:], df_l[0]))
    del chr_no['other']

    chr_no.reset_index(drop=True, inplace=True)
    df_l.reset_index(drop=True, inplace=True)

    final_chr = pd.concat([chr_no, df_l], axis=1)
    if write_to_file or not os.path.exists(base_write_path + chrm + '_annotations.csv'):
        print('Writing to sub annotation file..')
        final_chr.to_csv(base_write_path + chrm + '_annotations.csv', index=False, sep='\t')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create individual chromosome annotation and txt files')
    parser.add_argument('--no', type = str, help='chromosome number')
    args = parser.parse_args()

    chr_i = 'chr'+args.no
    base_write_txt_path = raw_data_path + 'txt/'
    base_write_ann_path = raw_data_path + 'annotations/'

    if not os.path.exists(base_write_txt_path):
        os.makedirs(base_write_txt_path)
    if not os.path.exists(base_write_ann_path):
        os.makedirs(base_write_ann_path)

    print('Pre-processing for Chromosome {}'.format(args.no))
    # read fasta file for chromosome, which generate the txt files --
    preprocess_chromosomeseq_file(chr_i+".fa", base_write_path=base_write_txt_path, write_to_file=True)
    # get annotations for particular chromosome --
    create_chromosome_annotations(chrm=chr_i, base_write_path=base_write_ann_path, write_to_file=True)
    print('Finished pre-processing Chromosome {}'.format(args.no))