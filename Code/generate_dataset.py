import pathlib
import pandas as pd
from os import path
import json
import itertools
import tqdm
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from heapq import merge
import portion as P
import random

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"
MAX_LENGTH = 350
NO_OFFSETS_PER_EXON = 5
OFFSET_RANGE = [60, 340]
WRITE_TO_FILE = False

def create_intervals(data, side, start=0, end=0):
    '''
    Function to get a list of lists (list of ranges) and return a list of intervals
    Ex. Exon interval = [8,20]
        MAX_LENGTH = 10, intron_offset = 5, exon_offset = 10-5-1 = 4
        Exon start interval = [8-5, 8+4] = [3,12], Exon boundary = 6 (the 6th nt in the seq of 10 is the start of the exon)
    :param data: list of lists (list of ranges)
    :param side: which side (exon_start/ exon_end)
    :param start: offset for getting (site-start) region
    :param end: offset for getting (site+end) region
    :return: 2 lists
            res: list of intervals containing the sequence
            exon_boundary: list of int containing the exon_boundary points
    '''
    # random.seed(1) # seed random number generator, used for de-bugging
    res = []
    exon_boundary = []

    for exon in data:
        if side == 'none':
            res.append(P.closed(exon[0], exon[1]))

        else:
            # generate 5 random numbers for exon boundary offset.
            for _ in range(NO_OFFSETS_PER_EXON):
                #range of offset into the intron is [60,340] (min intron length = 60, min exons length= 10 for each seq created)
                intron_offset = random.randint(OFFSET_RANGE[0], OFFSET_RANGE[1])
                exon_offset = MAX_LENGTH-intron_offset-1

                if side == 'start':
                    res.append(P.closed(exon[0] - intron_offset, exon[0] + exon_offset))
                if side == 'end':
                    res.append(P.closed(exon[1] - exon_offset, exon[1] + intron_offset))

                exon_boundary.append(intron_offset+1)  # 1-indexed (the start of seq is at index 1)

    return res, exon_boundary

def get_gene_seq(gene_sequence, strand):
    '''
    Function to return gene sequence
    :param gene_sequence: str
                Nucleotide sequence of gene
    :param strand: char
                Denoting '+' or '-' strand
    :return: str
                Nucleotide sequence of gene
    '''
    if (strand=='+'):
        return gene_sequence
    if (strand=='-'):
        my_seq = Seq(gene_sequence, generic_dna)
        my_seq.reverse_complement()
        return str(my_seq)

def get_nonoverlapping_exon_bounds(exon_ranges_in_transcripts):
    '''
    Function to get non-overlapping exon bounds
    :param exon_ranges_in_transcripts: list of lists (list of ranges)
    :return: list of lists (list of ranges)
    '''

    flat_t = [item for sublist in exon_ranges_in_transcripts for item in sublist]
    intervals = [list(x) for x in set(tuple(x) for x in flat_t)]
    intervals.sort(key=lambda x: x[0])

    # Merge intervals
    res = []
    res.append(intervals[0])
    for i in range(1, len(intervals)):
        last_elem = res[-1]
        if last_elem[1] >= intervals[i][0]:
            res[-1][1] = max(intervals[i][1], res[-1][1])
        else:
            res.append(intervals[i])
    return res

def remove_overlapping_genes(data):
    '''
    Function to divide the gene intervals into non-overlapping intervals
    :param data: JSON file
    :return: list of lists [[a,b],[c,d]]
        list of non-overlapping intervals
    Ex: Given list of gene intervals: [1,10], [2,4], [6,8], [20,30], [24,28], [35,40], [45,50]
        Get non-overlapping gene intervals: [1,2], [4,6], [8,10], [20,24], [28,30], [35,40], [45,50]
    '''

    gene_bounds_list = []
    for gene in data:
        gene_bounds = gene['gene_bounds']
        gene_bounds_list.append(gene_bounds)

    # gene_bounds_list = [[1,10], [5,20], [6,21],[17,25],[22,23], [24,50],[30,45],[60,70]]
    out = []
    starts = sorted([(i[0], 1) for i in gene_bounds_list])  # start of interval adds a layer of overlap
    ends = sorted([(i[1], -1) for i in gene_bounds_list])  # end removes one
    layers = 0
    current = []
    for value, event in merge(starts, ends):  # sorted by value, then ends (-1) before starts (1)
        layers += event
        if layers == 1:  # start of a new non-overlapping interval
            current.append(value)
        elif current:  # we either got out of an interval, or started an overlap
            current.append(value)
            out.append(current)
            current = []

    if WRITE_TO_FILE:
        write_to_file(out, 'non-overlapping_genes.txt')
    return out

def get_final_exon_intervals(exon_boundary_intervals, exon_boundaries, all_exon_intervals_list,
                             nonoverlapping_gene_intervals, side):
    '''
    Function to get final exon start and end lists for a gene, after checking the intervals satisfy all validity conditions
    :param exon_boundary_intervals: list of intervals
                            around the boundary for the given :param side (exon_start/ exon_end)
    :param exon_boundaries: list of int
                    containing the exon boundaries
    :param all_exon_intervals_list: list of exon intervals
    :param nonoverlapping_gene_intervals: list of non-overlapping gene intervals
    :param side: str
            which side ('exon_start'/ 'exon_end')
    :return: 2 lists
            exon_boundary_intervals_final: list of intervals
            exon_boundary_final: list of int
    '''
    exon_boundary_intervals_final = []
    exon_boundary_final = []

    for (exon_boundary_interval, exon_boundary, var) in zip(exon_boundary_intervals, exon_boundaries,
                                                          range(0,len(exon_boundary_intervals))):

        start_overlap_alert = False
        end_overlap_alert = False
        i = int(var/NO_OFFSETS_PER_EXON)

        # Check exon in within gene bounds
        for gene in nonoverlapping_gene_intervals:
            if exon_boundary_interval in gene:

                # Check exon site interval area does not overlap with other (5) exons on the left
                for j in range(i - 5, i):
                    if (j < 0):
                        continue
                    if not exon_boundary_interval > all_exon_intervals_list[j]:
                        #print('start overlap checked', exon_intervals_list[j])
                        start_overlap_alert = True
                        break

                # Check exon site interval area does not overlap with other (5) exons on the right
                for j in range(i + 1, i + 6):
                    if (j >= len(all_exon_intervals_list)):
                        continue
                    if not exon_boundary_interval < all_exon_intervals_list[j]:
                        end_overlap_alert = True
                        #print('exon end overlap checked', end_overlap_alert)
                        break

                "Check exon site interval area does not overlap with the same exon's other side"
                #For start sites, exon start site interval should not coincide with the exon end
                if (side=='start' and exon_boundary_interval < all_exon_intervals_list[i].upper \
                        and not start_overlap_alert and not end_overlap_alert):
                    exon_boundary_intervals_final.append(exon_boundary_interval)
                    exon_boundary_final.append(exon_boundary)
                # For end sites, exon end site interval should not coincide with the exon start
                if (side=='end' and exon_boundary_interval > all_exon_intervals_list[i].lower \
                        and not start_overlap_alert and not end_overlap_alert):
                    exon_boundary_intervals_final.append(exon_boundary_interval)
                    exon_boundary_final.append(exon_boundary)
                break

    return exon_boundary_intervals_final, exon_boundary_final

def get_negative_samples(exon_intervals, gene_bounds):
    '''
    Craete negative samples for the dataset
    :param exon_intervals:
    :param gene_bounds:
    :return: list of intervals
            intervals containing the purely exonic sequences (negative samples)
    '''
    # get purely exonic sequences ---
    within_exon_seq_interval = []

    for exon_interval in exon_intervals:
        # if the interval is subset-able
        if (exon_interval.upper - exon_interval.lower + 1) > MAX_LENGTH:
            for gene in gene_bounds:
                if exon_interval in gene:
                    #print('in gene!')
                    for _ in range(5):
                        n = random.randint(exon_interval.lower, exon_interval.upper-MAX_LENGTH)
                        within_exon_seq_interval.append(P.closed(n, n+MAX_LENGTH-1))

    # get purely intronic sequences
    '''
    2  5    6    9 11    15   20   25
    |  (    )    ( )     (    )    |
    G  E    E    E E     E    E    G     [E:exon_boundary, G:gene_boundary]
    intron intervals: [[3,4], [7,8], [12,14], [21,24]]   
    '''

    min_max_exon_interval = P.closed(exon_intervals[0].lower, exon_intervals[-1].upper)
    within_intron_seq_interval = []
    for gene in gene_bounds:
        if(min_max_exon_interval in gene):
            # create intron invervals
            exon_intervals.append(gene)
            flat_data = []
            for interval in exon_intervals:
                flat_data.extend([interval.lower, interval.upper])
            flat_data = sorted(flat_data)
            intron_intervals = [flat_data[i:i + 2] for i in range(0, len(flat_data), 2)]
            intron_intervals = list(map(lambda x: [x[0]+1, x[1]-1], intron_intervals)) # exon boundaries should not be included in intron intervals
            for intron_interval in intron_intervals:
                # if the interval is subset-able
                if (intron_interval[1] - intron_interval[0] + 1) > MAX_LENGTH:
                    print(intron_interval)
                    for _ in range(5):
                        n = random.randint(intron_interval[0], intron_interval[1] - MAX_LENGTH)
                        within_intron_seq_interval.append(P.closed(n, n + MAX_LENGTH - 1))
            break
    return within_exon_seq_interval

def create_training_set(exon_boundary_intervals_final, exon_boundary_final, gene):
    '''
    Function to create training set [start intervals/ end intervals] (positive samples)
    :param exon_boundary_intervals_final: list of intervals
    :param exon_boundary_final: list of int containing the exon boundaries
    :param gene: gene information (JSON)
    :return: 2 lists
            list of str: training_set_x containing the DNA seqeunces
            list of int: training_set_y containing the exon boundary position
    '''
    training_set_x = []
    training_set_y = []
    gene_sequence = gene['gene_sequence']
    gene_bounds = gene['gene_bounds']

    for (exon_interval, exon_boundary) in zip(exon_boundary_intervals_final, exon_boundary_final):
        print('Exon intervals', exon_interval.lower - gene_bounds[0], exon_interval.upper - gene_bounds[0]+1)
        seq = get_gene_seq(gene_sequence[exon_interval.lower - gene_bounds[0]:exon_interval.upper - gene_bounds[0]+1],
                         gene['gene_strand'])  #end index not included during offset: hence +1 during offset
        print(seq)
        training_set_x.append(seq)
        training_set_y.append(exon_boundary)

    return training_set_x, training_set_y


def write_to_file(data, file_name):
    '''
    Function to write to file
    :param data: Data to be written
    :param file_name: str - File name
    :return: None
    '''
    with open(data_path+file_name, "w+") as file:
        file.write("\n".join(str(item) for item in data))
    file.close()


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
    len_neg = 0
    len_pos = 0

    #print(len(data)) #875
    nonoverlapping_gene_intervals = remove_overlapping_genes(data)
    nonoverlapping_gene_intervals, _ = create_intervals(nonoverlapping_gene_intervals, 'none')
    print('non-overlapping genes ', len(nonoverlapping_gene_intervals), nonoverlapping_gene_intervals)  #821

    # Iterating through all genes of the chromosome
    for gene in data[0:3]:
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

            if (len(exon_ranges)!=0):  #if there exist exons in the transcript
                exons_ranges_in_transcript.append(exon_ranges)
        #print('All exon ranges', exons_ranges_in_transcript)

        # if there exists at least one exon in transcript----
        if (len(exons_ranges_in_transcript) >= 1):
            nonoverlapping_exon_ranges_for_gene = get_nonoverlapping_exon_bounds(exons_ranges_in_transcript)

            #get exon start & end intervals - with offsets: list of intervals
            exon_start_list, exon_start_boundaries = create_intervals(nonoverlapping_exon_ranges_for_gene, 'start')
            exon_end_list, exon_end_boundaries = create_intervals(nonoverlapping_exon_ranges_for_gene, 'end')
            exon_intervals_list, _ = create_intervals(nonoverlapping_exon_ranges_for_gene, 'none')

            exon_start_list = sorted(exon_start_list)
            exon_end_list = sorted(exon_end_list)
            exon_intervals_list = sorted(exon_intervals_list)

            # POSITIVE SAMPLES
            exon_start_set_final, exon_start_boundary_final = get_final_exon_intervals(exon_start_list, exon_start_boundaries,
                                                            exon_intervals_list, nonoverlapping_gene_intervals, 'start')

            exon_end_set_final, exon_end_boundary_final = get_final_exon_intervals(exon_end_list, exon_end_boundaries,
                                                          exon_intervals_list, nonoverlapping_gene_intervals, 'end')

            # NEGATIVE SAMPLES
            within_exon_seq_intervals = get_negative_samples(exon_intervals_list, nonoverlapping_gene_intervals)

            negative_start.extend(within_exon_seq_intervals)
            len_neg+=len(negative_start)
            len_pos+=len(exon_start_set_final)
            #print('final exon start set', exon_start_set_final, exon_start_boundary_final)
            #print('final exon end set', exon_end_set_final, exon_end_boundary_final)

            # Training set creation ---
            sx, sy = create_training_set(exon_start_set_final, exon_start_boundary_final, gene)
            nsx, nsy = create_training_set(negative_start, [0]*len(negative_start), gene)
            ex, ey = create_training_set(exon_end_set_final, exon_end_boundary_final, gene)
            start_training_x.extend(sx)
            start_training_y.extend(sy)
            start_training_x.extend(nsx)
            start_training_y.extend(nsy)
            end_training_x.extend(ex)
            end_training_y.extend(ey)

    print(start_training_y)
    print('No. of samples in start or end training set:', len(start_training_x), len(start_training_y)) #, ",", len(end_training_y))
    print("no. negative samples", len_neg)
    print("no. positive samples", len_pos)
    # Write to file ----
    if WRITE_TO_FILE:
        write_to_file(start_training_y, 'y_label_start')
        write_to_file(start_training_x, 'dna_seq_start')
        #write_to_file(end_training_y, 'y_label_end')
        #write_to_file(end_training_x, 'dna_seq_end')

    return

#todo: get stats of how many positive and negattive samples there are in the set
if __name__ == "__main__":

    file = data_path+"chr21_data.json"
    with open(file, "r") as f:
        dataset = json.load(f)

    manipulate(dataset)