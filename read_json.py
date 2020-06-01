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

def get_set_of_intervals(data, type, start = 0, end = 0):
    set_of_intervals = set()
    for interval in data:
        if(type=='open'):
            set_of_intervals.add(P.open(interval[0]-start, interval[1]+end))
        if (type == 'closed'):
            set_of_intervals.add(P.closed(interval[0]-start, interval[1]+end))
    return set_of_intervals


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

    :param exon_ranges_in_transcripts:
    :return:
    '''

    #print('exon_ranges_in_transcripts', exon_ranges_in_transcripts)
    flat_t = [item for sublist in exon_ranges_in_transcripts for item in sublist]
    intervals = [list(x) for x in set(tuple(x) for x in flat_t)]
    intervals.sort(key=lambda x: x[0])

    set_start = set()
    set_end = set()

    def add_to_set(set_start, set_end, interval):
        set_start.add(interval[0])
        set_end.add(interval[1])

    # Merge intervals
    res = []
    res.append(intervals[0])
    add_to_set(set_start, set_end, intervals[0])
    for i in range(1, len(intervals)):
        add_to_set(set_start, set_end, intervals[i])
        last_elem = res[-1]
        if last_elem[1] >= intervals[i][0]:
            res[-1][1] = max(intervals[i][1], res[-1][1])
        else:
            res.append(intervals[i])

    return res, set_start, set_end

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
    print(out)
    return out


def manipulate(dataset, exon_start_minus_offset, exon_start_plus_offset, exon_end_minus_offset, exon_end_plus_offset):
    '''
    Function to
    :param dataset: json file
                JSON file containing chromosome information
    :param exon_start_offset: int
    :param exon_end_offset: int
    :return: None
    '''
    ## Takes the parsed JSON dataset
    data = dataset["main"]

    training_set_x = []
    training_set_y = []
    set_start_pos = set()
    set_end_pos = set()
    gene_start_end_set = set()

    print(len(data)) #875
    nonoverlapping_intervals = remove_overlapping_genes(data[0:3])
    print(len(nonoverlapping_intervals))  #821


    for interval in nonoverlapping_intervals:
        set_start_pos.add(interval[0])
        set_end_pos.add(interval[1])

    gene_start_end_set = get_set_of_intervals(nonoverlapping_intervals, 'open')
    print('Nonoverlapping gene set', gene_start_end_set)

    # Iterating through all genes of the chromosome
    for gene in data[0:3]:
        print(gene['gene_id'])
        gene_sequence = get_gene_seq(gene['gene_sequence'], gene['gene_strand'])
        gene_bounds = gene['gene_bounds']

        exons_ranges_in_transcript = []

        # Iterating through all transcripts of the gene
        for transcript in gene["transcripts"]:
            #print(transcript['transcript_id'])
            exon_ranges = []

            # Iterating through all exons of the transcript for the gene
            for exon in transcript['exons']:
                ranges = [x for x in exon['exon_ranges']]
                exon_ranges.append(ranges)

            #if(gene['gene_strand']=='-'): #order the ranges from first to last exon
            #   exon_ranges = list(reversed(exon_ranges))

            if (len(exon_ranges)!=0):  #if there exist exons in the transcript
                exons_ranges_in_transcript.append(exon_ranges)

        print('All exon ranges', exons_ranges_in_transcript)
        if (len(exons_ranges_in_transcript) >= 1):
            nonoverlapping_exon_ranges_for_gene, ss, se = get_nonoverlapping_exon_bounds(exons_ranges_in_transcript)
            #exon_intervals_set = get_set_of_intervals(nonoverlapping_exon_ranges_for_gene, 'closed')

            def create_intervals(data, side, start, end):
                res = []
                for exon in data:
                    if side=='start':
                        res.append(P.closed(exon[0]-start, exon[0]+end))
                    if side=='end':
                        res.append(P.closed(exon[1] - start, exon[1] + end))
                return res

            #get exon start intervals - with offsets #list of intervals
            exon_start_set = create_intervals(nonoverlapping_exon_ranges_for_gene, 'start', exon_start_minus_offset, exon_start_plus_offset)
            exon_end_set = create_intervals(nonoverlapping_exon_ranges_for_gene, 'end', exon_end_minus_offset, exon_end_plus_offset)

            exon_start_set = sorted(exon_start_set)
            exon_end_set = sorted(exon_end_set)

            print('Exon start set', exon_start_set)
            print('Exon end set', exon_end_set)
            exon_start_set_final = []
            for (exon_start_interval, i) in zip(exon_start_set, range(0,len(exon_start_set))):

                exon_within_gene_alert = False
                start_overlap_alert = False
                end_overlap_alert = False

                print(exon_start_interval)
                #Check exon in within gene bounds
                for gene in gene_start_end_set:
                    if exon_start_interval in gene:
                        #exon_within_gene_alert = True
                        #f exon_within_gene_alert:
                        print('in gene')
                        #Check exon start area does not overlap with other exon start areas
                        for e in exon_start_set:
                            if e==exon_start:
                                continue
                            elif(exon_start.overlaps(e)):
                                print(e)
                                start_overlap_alert=True
                                break
                        print('exon start overlap checked', start_overlap_alert)
                        if not start_overlap_alert:
                            # Check exon start area does not overlap with other exon end areas
                            for e in exon_end_set:
                                if e == exon_start:
                                    continue
                                elif (exon_start.overlaps(e)):
                                    print(e)
                                    end_overlap_alert = True
                                    break
                        print('exon end overlap checked', end_overlap_alert)
                        if not end_overlap_alert:
                            exon_start_set_final.append(exon_start)


                print(exon_start_set_final)
        #sub_start = exons_ranges_in_transcript[0][0][0]-exon_start_offset
        #sub_end = exons_ranges_in_transcript[0][-1][1]+exon_end_offset+1

    '''
        training_set_x.append(get_gene_seq(gene_sequence[gene_bounds[0]-gene_bounds[0]:gene_bounds[1]-gene_bounds[0]], gene['gene_strand']))
        training_set_y.append(nonoverlapping_exon_ranges_for_gene)

    print('No. of samples in training set:', len(training_set_y))
    print(sorted(set_start_pos))
    print(sorted(set_end_pos))

    df = pd.DataFrame()
    df['x'] = training_set_x
    df['y'] = training_set_y
    #df.to_csv(data_path+'chr21_training_data.csv', index = False)
    length_seq = [len(i) for i in training_set_x]
    #print(training_set_x)
    
    print(training_set_y)
    print(length_seq)
    print(max(length_seq))
    print(min(length_seq))
    '''

    return

if __name__ == "__main__":

    file = data_path+"chr21_data.json"
    with open(file, "r") as f:
        dataset = json.load(f)

    manipulate(dataset, 300, 30, 300, 30)