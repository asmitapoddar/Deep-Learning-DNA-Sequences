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

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

def create_intervals(data, side, start=0, end=0):
    '''
    Function to get a list of lists (list of ranges) and return a list of intervals
    :param data: list of lists (list of ranges)
    :param side:
    :param start: offset for getting (site-start) region
    :param end: offset for getting (site+end) region
    :return: list of intervals
    '''
    res = []
    for exon in data:
        if side == 'start':
            res.append(P.closed(exon[0] - start, exon[0] + end))
        if side == 'end':
            res.append(P.closed(exon[1] - start, exon[1] + end))
        if side == 'none':
            res.append(P.closed(exon[0], exon[1]))
    return res

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
    :param data:
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

    return out

def get_final_exon_intervals(exon_side_intervals, exon_intervals_list, nonoverlapping_gene_intervals, side):
    '''
    Function to get final exon start and end lists for a gene
    :param exon_side_intervals:
    :param exon_intervals_list:
    :param nonoverlapping_gene_intervals:
    :param side:
    :return:
    '''
    exon_side_intervals_final = []
    for (exon_side_interval, i) in zip(exon_side_intervals, range(0, len(exon_intervals_list))):

        start_overlap_alert = False
        end_overlap_alert = False

        print(i, exon_side_interval, exon_intervals_list[i])
        # Check exon in within gene bounds
        for gene in nonoverlapping_gene_intervals:
            if exon_side_interval in gene:

                print('in gene')
                # Check exon site interval area does not overlap with other (5) exons on the left
                for j in range(i - 5, i):
                    if (j < 0):
                        continue
                    if not exon_side_interval > exon_intervals_list[j]:
                        print('start overlap checked', exon_intervals_list[j])
                        start_overlap_alert = True
                        break
                # Check exon site interval area does not overlap with other (5) exons on the right
                for j in range(i + 1, i + 6):
                    if (j >= len(exon_intervals_list)):
                        continue
                    if not exon_side_interval < exon_side_intervals[j]:
                        end_overlap_alert = True
                        print('exon end overlap checked', end_overlap_alert)
                        break

                "Check exon site interval area does not overlap with the same exon's other side"
                #For start sites, exon start site interval should not coincide with the exon end
                if (side=='start' and exon_side_interval < exon_intervals_list[i].upper \
                        and not start_overlap_alert and not end_overlap_alert):
                    exon_side_intervals_final.append(exon_side_interval)
                # For end sites, exon end site interval should not coincide with the exon start
                if (side=='end' and exon_side_interval > exon_intervals_list[i].lower \
                        and not start_overlap_alert and not end_overlap_alert):
                    exon_side_intervals_final.append(exon_side_interval)
                break

    return exon_side_intervals_final

def create_training_set(exon_intervals_final, gene):
    '''
    Function to create training set [start intervals/ end intervals] (positive samples)
    :param exon_intervals_final: list of intervals
    :param gene: gene information (JSON)
    :return: list
    '''
    training_set_x = []
    training_set_y = []
    gene_sequence = gene['gene_sequence']
    gene_bounds = gene['gene_bounds']

    for exon_interval in exon_intervals_final:
        training_set_x.append(
            get_gene_seq(gene_sequence[exon_interval.lower - gene_bounds[0]:exon_interval.upper - gene_bounds[0]],
                         gene['gene_strand']))
        training_set_y.append([exon_interval.lower - gene_bounds[0], exon_interval.upper - gene_bounds[0]])

    return training_set_x, training_set_y

def manipulate(dataset, exon_start_minus_offset, exon_start_plus_offset, exon_end_minus_offset, exon_end_plus_offset):
    '''
    Function to
    :param dataset: json file
                JSON file containing chromosome information
    :param exon_start_minus_offset: int
    :param exon_start_plus_offset: int
    :param exon_end_minus_offset: int
    :param exon_end_plus_offset: int
    :return: None
    '''
    ## Takes the parsed JSON dataset
    data = dataset["main"]

    start_training_x = []
    end_training_x = []
    start_training_y = []
    end_training_y = []

    #print(len(data)) #875
    nonoverlapping_gene_intervals = remove_overlapping_genes(data)
    nonoverlapping_gene_intervals = create_intervals(nonoverlapping_gene_intervals, 'none')
    print(len(nonoverlapping_gene_intervals), nonoverlapping_gene_intervals)  #821

    # Iterating through all genes of the chromosome
    for gene in data:
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
        print('All exon ranges', exons_ranges_in_transcript)

        # if there exists at least one transcript----
        if (len(exons_ranges_in_transcript) >= 1):
            nonoverlapping_exon_ranges_for_gene = get_nonoverlapping_exon_bounds(exons_ranges_in_transcript)

            #get exon start & end intervals - with offsets: list of intervals
            exon_start_list = create_intervals(nonoverlapping_exon_ranges_for_gene, 'start', exon_start_minus_offset, exon_start_plus_offset)
            exon_end_list = create_intervals(nonoverlapping_exon_ranges_for_gene, 'end', exon_end_minus_offset, exon_end_plus_offset)
            exon_intervals_list = create_intervals(nonoverlapping_exon_ranges_for_gene, 'none')

            exon_start_list = sorted(exon_start_list)
            exon_end_list = sorted(exon_end_list)
            exon_intervals_list = sorted(exon_intervals_list)

            #print('Exons interval set', exon_intervals_list)
            #print('Exon start set', exon_start_list)
            #print('Exon end set', exon_end_list)

            exon_start_set_final = get_final_exon_intervals(exon_start_list, exon_intervals_list, nonoverlapping_gene_intervals, 'start')
            #print('final exon start set', exon_start_set_final)
            exon_end_set_final = get_final_exon_intervals(exon_end_list, exon_intervals_list, nonoverlapping_gene_intervals, 'end')
            #print('final exon end set', exon_end_set_final)

            sx, sy = create_training_set(exon_start_set_final, gene)
            ex, ey = create_training_set(exon_end_set_final, gene)
            start_training_x.extend(sx)
            start_training_y.extend(sy)
            end_training_x.extend(ex)
            end_training_y.extend(ey)


    print(start_training_x, start_training_y)
    print(end_training_x, end_training_y)
    print('No. of samples in start and end training set:', len(start_training_y), ",", len(end_training_y))
    '''
    df = pd.DataFrame()
    df['x'] = training_set_x
    df['y'] = training_set_y
    #df.to_csv(data_path+'chr21_training_data.csv', index = False)
    length_seq = [len(i) for i in training_set_x]
    '''
    return

if __name__ == "__main__":

    file = data_path+"chr21_data.json"
    with open(file, "r") as f:
        dataset = json.load(f)

    manipulate(dataset, 300, 30, 30, 300)