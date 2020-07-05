import pathlib
import pandas as pd
import os
import json
import itertools
import tqdm
from dataset_utils import *
from encode import *

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

MAX_LENGTH = 100
NO_OFFSETS_PER_EXON = 5
OFFSET_RANGE = [60, MAX_LENGTH-10]
EXON_BOUNDARY = 'start'  # or 'end'
DATASET_TYPE = 'regression' # 'classification'  # or 'regression'
classes = 'three'  # if DATASET_TYPE = 'classification'
SEQ_TYPE = 'cds'  # 'cds'/'exons'
NO_OF_GENES = 875

WRITE_DATA_TO_FILE = True
WRITE_TO_FILE_PATH = data_path + 'chrm21' + '/' + str(DATASET_TYPE) + '/'+ SEQ_TYPE + '_' + \
                     str(EXON_BOUNDARY) + '_n' + str(NO_OF_GENES) + '_l' + str(MAX_LENGTH) +'0-max'
DATA_LOG = True
SANITY_CHECK = False
META_DATA = False  #For Upamanyu

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
    all_exon_intervals = pd.DataFrame(columns=['Gene', 'Exon_Intervals'])
    len_exon = 0
    len_intron = 0
    len_boundary = 0

    columns = ['Type', 'Sequence', 'Interval_Indices', 'GeneID', 'Gene_Strand', 'Boundary_Position']
    finaldf = pd.DataFrame(columns=columns)  # For sanity check
    chrm_path = data_path + chrm + '/'

    # print(len(data)) #875
    nonoverlapping_gene_intervals = remove_overlapping_genes(data)
    nonoverlapping_gene_intervals = convert_list_to_interval(nonoverlapping_gene_intervals)
    print('non-overlapping genes ', len(nonoverlapping_gene_intervals), nonoverlapping_gene_intervals)  # 821

    # Iterating through all genes of the chromosome
    for gene in data[0:NO_OF_GENES]:  #Note: controlling how many genes to use

        print(gene['gene_id'])
        gene_sequence = get_gene_seq(gene['gene_sequence'], gene['gene_strand'])
        gene_bounds = gene['gene_bounds']

        exons_ranges_in_transcript = []
        # Iterating through all transcripts of the gene
        for transcript in gene["transcripts"]:
            exon_ranges = []
            # Iterating through all exons of the transcript for the gene
            for exon in transcript[SEQ_TYPE]:
                ranges = [x for x in exon[SEQ_TYPE+'_ranges']]
                exon_ranges.append(ranges)
            if (len(exon_ranges) != 0):  # if there exist exons in the transcript
                exons_ranges_in_transcript.append(exon_ranges)

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

            # Get training set info
            len_exon += len(within_exon_seq_intervals)
            len_intron += len(within_intron_seq_intervals)
            len_boundary += len(exon_boundary_set_final)
            print('Dataset stats (#boundary, #exon, #intron): ', len_boundary, len_exon, len_intron)

            all_exon_intervals = all_exon_intervals.append(
                {'Gene': gene['gene_id'], 'Exon_Intervals': exon_intervals_list}, ignore_index=True)

            # TRAINING SET CREATION ----
            if DATASET_TYPE=='classification':
                '''
                Sequences containing Exon Boundary: Class 0
                Purely Exonic Sequences: Class 1
                Purely Intronic Sequences: Class 2
                '''
                if classes=='three':
                    sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                                 [0]*len(exon_boundary_y_final), gene, MAX_LENGTH)
                    sxexon, syexon = create_training_set(within_exon_seq_intervals,
                                                         [1] * len(within_exon_seq_intervals), gene, MAX_LENGTH)
                    sxintron, syintron = create_training_set(within_intron_seq_intervals,
                                                             [2] * len(within_intron_seq_intervals), gene, MAX_LENGTH)
                    training_x.extend(sxboundary+sxexon+sxintron)
                    training_y.extend(syboundary+syexon+syintron)

                if classes=='two':
                    sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                                 [0] * len(exon_boundary_y_final), gene, MAX_LENGTH)
                    sxexon, syexon = create_training_set(within_exon_seq_intervals,
                                                         [1] * len(within_exon_seq_intervals), gene, MAX_LENGTH)
                    sxintron, syintron = create_training_set(within_intron_seq_intervals,
                                                             [1] * len(within_intron_seq_intervals), gene, MAX_LENGTH)
                    training_x.extend(sxboundary + sxexon + sxintron)
                    training_y.extend(syboundary + syexon + syintron)

                if classes == 'one':
                    sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                                 [0] * len(exon_boundary_y_final), gene, MAX_LENGTH)
                    training_x.extend(sxboundary)
                    training_y.extend(syboundary)

            if DATASET_TYPE == 'regression':
                '''
                Sequences containing Exon Boundary: Boundary point
                Purely Exonic Sequences: 0
                Purely Intronic Sequences: 0
                '''
                sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                             [x-60 for x in exon_boundary_y_final], gene, MAX_LENGTH)
                #sxintron, syintron = create_training_set(within_intron_seq_intervals,
                #                                         [0] * len(within_intron_seq_intervals), gene, MAX_LENGTH)
                training_x.extend(sxboundary)
                training_y.extend(syboundary)

                if SANITY_CHECK:
                    dfboundary = sanity_check(sxboundary, exon_boundary_set_final, columns, 'Boundary', gene, exon_boundary_y_final)
                    dfexon = sanity_check(sxexon, within_exon_seq_intervals, columns, 'Exon', gene)
                    dfintron = sanity_check(sxintron, within_intron_seq_intervals, columns, 'Intron', gene)
                    finaldf = finaldf.append([dfboundary, dfexon, dfintron])

    dataset_path = chrm_path + DATASET_TYPE
    # Write to file ----
    if WRITE_DATA_TO_FILE:

        print('WRITE_TO_FILE_PATH', WRITE_TO_FILE_PATH)
        if not os.path.exists(WRITE_TO_FILE_PATH):
            os.makedirs(WRITE_TO_FILE_PATH)
        write_to_file(training_y, WRITE_TO_FILE_PATH + '/y_label_'+EXON_BOUNDARY)
        write_to_file(training_x, WRITE_TO_FILE_PATH + '/dna_seq_'+EXON_BOUNDARY)

    if DATA_LOG:
        if not os.path.exists(WRITE_TO_FILE_PATH):
            os.makedirs(WRITE_TO_FILE_PATH)
        with open(WRITE_TO_FILE_PATH + '/info.log', 'w+') as f:
            f.write('DATASET TYPE: ' + str(DATASET_TYPE))
            f.write('\nSEQUENCE TYPE: ' + str(SEQ_TYPE))
            f.write('\nEXON BOUNDARY: ' + str(EXON_BOUNDARY))
            f.write('\nMAX SEQUENCE LENGTH = ' + str(MAX_LENGTH))
            f.write('\nNO_OFFSETS_PER_EXON = ' + str(NO_OFFSETS_PER_EXON))
            f.write('\nOFFSET_RANGE (from exon boundary into intronic region) = ' + str(OFFSET_RANGE))
            f.write('\n\nTotal no. of samples in training set = ' + str(len(training_x)))
            f.write("\nNo. samples containing intron-exon boundary = " + str(len_boundary))
            f.write("\nNo. of pure exon samples = " + str(len_exon))
            f.write("\nNo. of pure intron samples = " + str(len_intron))

    if SANITY_CHECK:
        if not os.path.exists(WRITE_TO_FILE_PATH):
            os.makedirs(WRITE_TO_FILE_PATH)
        finaldf.to_csv(WRITE_TO_FILE_PATH +'/sanity_'+str(MAX_LENGTH)+'.csv', header=columns, index = False)

    if META_DATA:
        if not os.path.exists(chrm_path):
            os.makedirs(chrm_path)
        write_to_file(nonoverlapping_gene_intervals, chrm_path+chrm+'_nonoverlapping_gene_intervals.txt')
        all_exon_intervals.to_csv(chrm_path+chrm+'_exon_intervals.csv', header=['Gene', 'Exon_Intervals'], index=False, doublequote=False)

    print('No. of samples in training set:', len(training_x))
    print("no. positive samples", len_boundary)
    print("no. exon samples", len_exon)
    print("no. intron samples", len_intron)

    return

if __name__ == "__main__":
    file = data_path + "chr21_"+SEQ_TYPE+"_data.json"
    with open(file, "r") as f:
        dataset = json.load(f)

    manipulate(dataset, 'chrm21')
    encode_seq(WRITE_TO_FILE_PATH)