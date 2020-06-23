import pathlib
import pandas as pd
import os
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
EXON_BOUNDARY = 'start'  # or 'end'
DATASET_TYPE = 'classification'  # or 'regression'
WRITE_DATA_TO_FILE = True
DATA_LOG = True
log_path = data_path + DATASET_TYPE

def manipulate(dataset, chrm):
    '''
    Function to generate the training dataset
    :param dataset: json file
                JSON file containing chromosome information
    :param chrm: str
                Chromosome being parsed
    :return: None
    '''
    ## Takes the parsed JSON dataset
    data = dataset["main"]

    training_x = []
    training_y = []
    len_exon = 0
    len_intron = 0
    len_boundary = 0

    # print(len(data)) #875
    nonoverlapping_gene_intervals = remove_overlapping_genes(data)
    nonoverlapping_gene_intervals = convert_list_to_interval(nonoverlapping_gene_intervals)
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
            exon_boundary_list, exon_boundaries_y = create_boundary_intervals(nonoverlapping_exon_ranges_for_gene, EXON_BOUNDARY,
                                                                              MAX_LENGTH, NO_OFFSETS_PER_EXON, OFFSET_RANGE)
            exon_intervals_list = convert_list_to_interval(nonoverlapping_exon_ranges_for_gene)

            exon_boundary_list = sorted(exon_boundary_list)
            exon_intervals_list = sorted(exon_intervals_list)

            # POSITIVE SAMPLES
            exon_boundary_set_final, exon_boundary_y_final = get_final_exon_intervals(exon_boundary_list,
                                                                                       exon_boundaries_y,
                                                                                       exon_intervals_list,
                                                                                       nonoverlapping_gene_intervals,
                                                                                       EXON_BOUNDARY)

            # NEGATIVE SAMPLES
            within_exon_seq_intervals, within_intron_seq_intervals = get_negative_samples(exon_intervals_list,
                                                                                          nonoverlapping_gene_intervals,
                                                                                          MAX_LENGTH)

            len_exon += len(within_exon_seq_intervals)
            len_intron += len(within_intron_seq_intervals)
            len_boundary += len(exon_boundary_set_final)
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
                training_x.extend(sxboundary+sxexon+sxintron)
                training_y.extend(syboundary+syexon+syintron)

            if DATASET_TYPE == 'regression':
                '''
                Sequences containing Exon Boundary: Boundary point
                Purely Exonic Sequences: 0
                Purely Intronic Sequences: 0
                '''
                sxboundary, syboundary = create_training_set(exon_boundary_set_final, exon_boundary_y_final, gene)
                sxexon, syexon = create_training_set(within_exon_seq_intervals, [0] * len(within_exon_seq_intervals), gene)
                sxintron, syintron = create_training_set(within_intron_seq_intervals, [0] * len(within_intron_seq_intervals), gene)
                training_x.extend(sxboundary + sxexon + sxintron)
                training_y.extend(syboundary + syexon + syintron)

    write_dir = log_path + '/' + chrm
    # Write to file ----
    if WRITE_DATA_TO_FILE:

        print(write_dir)
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        write_to_file(training_y, write_dir + '/y_label_'+EXON_BOUNDARY)
        write_to_file(training_x, write_dir + '/dna_seq_'+EXON_BOUNDARY)

    if DATA_LOG:
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        with open(write_dir + '/info.log', 'w+') as f:
            f.write('DATASET TYPE: ' + str(DATASET_TYPE))
            f.write('\nEXON BOUNDARY: ' + str(EXON_BOUNDARY))
            f.write('\nMAX SEQUENCE LENGTH = ' + str(MAX_LENGTH))
            f.write('\nNO_OFFSETS_PER_EXON = ' + str(NO_OFFSETS_PER_EXON))
            f.write('\nOFFSET_RANGE = ' + str(OFFSET_RANGE))
            f.write('\n\nTotal no. of samples in training set = ' + str(len(training_x)))
            f.write("\nNo. samples containing intron-exon boundary = " + str(len_boundary))
            f.write("\nNo. of pure exon samples = " + str(len_exon))
            f.write("\nNo. of pure intron samples = " + str(len_intron))

        print('No. of samples in training set:', len(training_x))
        print("no. positive samples", len_boundary)
        print("no. exon samples", len_exon)
        print("no. intron samples", len_intron)

    return

if __name__ == "__main__":
    file = data_path + "chr21_data.json"
    with open(file, "r") as f:
        dataset = json.load(f)

    manipulate(dataset, 'chr21')