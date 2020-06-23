import pathlib
import pandas as pd
from os import path
import json
import itertools
import tqdm
import random
from dataset_utils import *

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"
MAX_LENGTH = 200
NO_OFFSETS_PER_EXON = 5
OFFSET_RANGE = [60, 340]
WRITE_TO_FILE = True
EXON_BOUNDARY = 'start'
DATASET_TYPE = 'classification'  # 'regression'

def manipulate(dataset):
    '''
    Function to generate the training dataset
    :param dataset: json file
                JSON file containing chromosome information
    :return: None
    '''
    ## Takes the parsed JSON dataset
    data = dataset["main"]

    start_training_x = []
    end_training_x = []
    start_training_y = []
    end_training_y = []
    len_exon = 0
    len_intron = 0
    len_pos = 0

    # print(len(data)) #875
    nonoverlapping_gene_intervals = remove_overlapping_genes(data)
    nonoverlapping_gene_intervals, _ = create_intervals(nonoverlapping_gene_intervals, 'none')
    print('non-overlapping genes ', len(nonoverlapping_gene_intervals), nonoverlapping_gene_intervals)  # 821

    # Iterating through all genes of the chromosome
    for gene in data:  #Note: controlling how many genes to use
        negative_start = []

        print(gene['gene_id'])
        gene_sequence = get_gene_seq(gene['gene_sequence'], gene['gene_strand'])
        gene_bounds = gene['gene_bounds']

        exons_ranges_in_transcript = []

        # Iterating through all transcripts of the gene
        for transcript in gene["transcripts"]:

            exon_ranges = []

            # Iterating through all exons of the transcript for the gene
            for exon in transcript['exons']:
                ranges = [x for x in exon['exon_ranges']]
                exon_ranges.append(ranges)

            if (len(exon_ranges) != 0):  # if there exist exons in the transcript
                exons_ranges_in_transcript.append(exon_ranges)
        # print('All exon ranges', exons_ranges_in_transcript)

        # if there exists at least one exon in transcript----
        if (len(exons_ranges_in_transcript) >= 1):
            nonoverlapping_exon_ranges_for_gene = get_nonoverlapping_exon_bounds(exons_ranges_in_transcript)

            # get exon start & end intervals - with offsets: list of intervals
            exon_boundary_list, exon_boundaries_y = create_intervals(nonoverlapping_exon_ranges_for_gene, EXON_BOUNDARY)
            exon_intervals_list, _ = create_intervals(nonoverlapping_exon_ranges_for_gene, 'none')

            exon_boundary_list = sorted(exon_boundary_list)
            exon_intervals_list = sorted(exon_intervals_list)

            # POSITIVE SAMPLES
            exon_boundary_set_final, exon_boundary_y_final = get_final_exon_intervals(exon_boundary_list,
                                                                                       exon_boundaries_y,
                                                                                       exon_intervals_list,
                                                                                       nonoverlapping_gene_intervals,
                                                                                       EXON_BOUNDARY)

            # NEGATIVE SAMPLES
            within_exon_seq_intervals, within_intron_seq_intervals = get_negative_samples(exon_intervals_list, nonoverlapping_gene_intervals)

            len_exon += len(within_exon_seq_intervals)
            len_intron += len(within_intron_seq_intervals)
            len_pos+=len(exon_boundary_set_final)
            #print(len_pos, len_exon, len_intron)

            # TRAINING SET CREATION ----
            if DATASET_TYPE=='classification':
                '''
                Sequences containing Exon Boundary: Class 0
                Purely Exonic Sequences: Class 1
                Purely Intronic Sequences: Class 2
                '''
                sxboundary, syboundary = create_training_set(exon_boundary_set_final, [0]*len(exon_boundary_y_final), gene)
                sxexon, syexon = create_training_set(within_exon_seq_intervals, [1] * len(within_exon_seq_intervals), gene)
                sxintron, syintron = create_training_set(within_intron_seq_intervals, [2] * len(within_intron_seq_intervals), gene)
                #print('sxboundary', sxboundary, syboundary)
                #print(sxexon, syexon)
                #print(sxintron, syintron)
                start_training_x.extend(sxboundary+sxexon+sxintron)
                start_training_y.extend(syboundary+syexon+syintron)

            if DATASET_TYPE == 'regression':
                '''
                Sequences containing Exon Boundary: Class 0
                Purely Exonic Sequences: Class 1
                Purely Intronic Sequences: Class 2
                '''
                sx, sy = create_training_set(exon_boundary_set_final, exon_boundary_y_final, gene)
                sxexon, syexon = create_training_set(within_exon_seq_intervals, [1] * len(within_exon_seq_intervals), gene)
                sxintron, syintron = create_training_set(within_intron_seq_intervals, [0] * len(within_intron_seq_intervals), gene)
                start_training_x.extend(sxboundary + sxexon + sxintron)
                start_training_y.extend(syboundary + syexon + syintron)

    print('No. of samples in training set:', len(start_training_x),
          len(start_training_y))  # , ",", len(end_training_y))

    print("no. positive samples", len_pos)
    print("no. exon samples", len_exon)
    print("no. intron samples", len_intron)
    # Write to file ----
    if WRITE_TO_FILE:
        write_to_file(start_training_y, data_path+DATASET_TYPE+'/y_label_'+EXON_BOUNDARY)
        write_to_file(start_training_x, data_path+DATASET_TYPE+'/dna_seq_'+EXON_BOUNDARY)

    return


# todo: get stats of how many positive and negattive samples there are in the set and write to file
if __name__ == "__main__":
    file = data_path + "chr21_data.json"
    with open(file, "r") as f:
        dataset = json.load(f)

    manipulate(dataset)