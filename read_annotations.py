import numpy as np
import pathlib
import pandas as pd
from os import path

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

def start_end_positions(file="chr21_annotations.csv"):
    '''
    Function to read annotation given file for chromosome and get [start, end] base pair position of genes
    :param file:
            chromosome annotation file
    :return: ndarray (num_genes, 2)
            Array containing [start, end] base pair position on chromosome for each gene.
    '''
    final_chr = pd.read_csv(data_path + file, sep='\t')
    chr_gene = final_chr[final_chr['type'] == 'gene']
    start_end_pos = pd.DataFrame(
        {'start_pos': chr_gene['start'], 'end_pos': chr_gene['end']})  # shape: (875,2) for chr21
    start_end_pos.reset_index(drop=True, inplace=True)
    return start_end_pos


def preprocess_chromosomeseq_file(chr_file="chr21.fa", write_to_file=False):
    '''
    Function to remove special characters in chromosome sequence
    :param chr_file: File containing nucleotide sequence for chromosome
    :param write_to_file: bool, optional
            Variable to control writing extracted information to txt file
    :return: str
            String containing nucleotide sequence for chromosome
    '''
    file_object = open(data_path + chr_file, "r")

    chrm = file_object.read()
    # print('Length of unprocessed chromosome: '+len(chrm))  # 47644190

    chrm = chrm.replace('\n', '')
    #Fasta file contains name of chromosome (eg. '>chr21') at the start: removing it
    if (chrm[5].isdigit()):
        chrm = chrm[6:]
    else:
        chrm = chrm[5:]
    # print('Length of processed chromosome: ', len(chrm))  # 46709983

    #Append a blank at the beginning and end of string (since annotations are 1-indexed
    chrm = ' '+chrm+' '   # len = 46709985
    print('Length of processed chromosome: ', len(chrm))
    if write_to_file or not path.exists(data_path+chr_file.replace("fa", "txt")):
        text_file = open(data_path + chr_file.replace("fa", "txt"), "w")
        text_file.write(chrm)
        text_file.close()
    return chrm


def create_chromosome_annotations(chrm='chr21', write_to_file=False):
    '''
    Function to read annotation file for human genome sequence GRCh38 and get annotations for specified chromosome
    :param chrm: str
            chromosome information to be extracted
    :param write_to_file: bool, optional
            Variable to control writing extracted information to file
    :return: None
    '''
    ann_file = "gencode.v34.annotation.gtf"
    col_names = ['chr_name', 'source', 'type', 'start', 'end', '.', 'strand', ',', 'other']
    ann = pd.read_csv(data_path + ann_file, sep='\t', header=None)
    ann.columns = col_names

    # print(ann.shape)  # (2912496, 9)
    # print(ann.head())

    chr_no = ann[ann['chr_name'] == chrm]
    del chr_no['.']
    del chr_no[',']
    print(chr_no.head())

    l = list(map(lambda x: x.split(';'), chr_no['other']))
    df_l = pd.DataFrame(l)
    df_l[0] = list(map(lambda x: x[8:], df_l[0]))
    del chr_no['other']

    chr_no.reset_index(drop=True, inplace=True)
    df_l.reset_index(drop=True, inplace=True)

    final_chr = pd.concat([chr_no, df_l], axis=1)
    if write_to_file or not path.exists(data_path + chrm + '_annotations.csv'):
        final_chr.to_csv(data_path + chrm + '_annotations.csv', index=False, sep='\t')


if __name__ == "__main__":
    chrm = preprocess_chromosomeseq_file("chr21.fa", write_to_file=True)  #read fasta file for chromosome, which generate the txt files
    create_chromosome_annotations(chrm='chr21', write_to_file=True) #get annotations for particular chromosome
    file_object = open(data_path + "chr21.txt", "r")
    chrm = file_object.read()
    start_end_pos = start_end_positions("chr21_annotations.csv")
    print(start_end_pos.shape)