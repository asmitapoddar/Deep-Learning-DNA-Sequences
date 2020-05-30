import pathlib
import pandas as pd
from os import path
import json
import itertools
import tqdm
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from heapq import merge

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

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
    #print("in function")
    #print(exon_ranges_in_transcripts)
    flat_t = [item for sublist in exon_ranges_in_transcripts for item in sublist]
    intervals = [list(x) for x in set(tuple(x) for x in flat_t)]
    intervals.sort(key=lambda x: x[0])
    #print(intervals)
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

    def gene_bounds_sorted(gene_bounds_list):
        ns = []
        for bound in gene_bounds_list:
            ns.append(bound[0])
        s = sorted(ns)
        print("Is sorted??", ns == s)
        return ns==s

    print('HI')
    gene_bounds_list = []
    for gene in data:
        gene_bounds = gene['gene_bounds']
        gene_bounds_list.append(gene_bounds)
        if(gene_bounds[0]>gene_bounds[1]):
            print("OHH NOO", gene_bounds)


    '''Printing
    for i in range(0, len(gene_bounds_list)):
        print(i, gene_bounds_list[i])
    '''
    gene_bounds_list = [[1,10],[5,20], [6,21],[17,25],[22,23], [24,50],[30,55],[60,70]]
    prev_gene_bounds = gene_bounds_list[0]
    overlap_list = []
    nonoverlap_list = []
    nonoverlap_list.append(gene_bounds_list[0])
    print(nonoverlap_list)

    if gene_bounds_sorted(gene_bounds_list):
        #range(130, len(gene_bounds_list[100:140]))
        for i in range(1, len(gene_bounds_list)):
            curr_gene_bounds = gene_bounds_list[i]
            prev_gene_bounds = nonoverlap_list[-1]
            print(i, nonoverlap_list[-1], curr_gene_bounds, overlap_list)

            if curr_gene_bounds[0]<prev_gene_bounds[0]:
                ''' Eg.     
                    16    |          68      |
                    |    21           |     89
                case1:       31   41
                             --del--
                case2:       31       |--del--|      92     
                curr_gene-> 31-start
                prev_gene-> 68-start    
                '''
                if curr_gene_bounds[1]<prev_gene_bounds[0]: #case1
                    continue
                if curr_gene_bounds[1] < prev_gene_bounds[1]:  #case2
                    nonoverlap_list[-1][0] = curr_gene_bounds[1]
                if curr_gene_bounds[1]>prev_gene_bounds[1]:
                    # previous gene was completely overlapping within current gene,
                    # so replace previous gene by current (bigger) gene and put previous gene into overlap list
                    overlap_list.append(nonoverlap_list[-1])
                    new_bound = [gene_bounds_list[i][0], gene_bounds_list[i][1]]
                    nonoverlap_list.pop()
                    nonoverlap_list.append([new_bound[0], new_bound[1]])
                    print(nonoverlap_list, overlap_list)

            elif curr_gene_bounds[0] > prev_gene_bounds[0] and curr_gene_bounds[1] < prev_gene_bounds[1]:
                # completely within another gene
                overlap_list.append([curr_gene_bounds[0], curr_gene_bounds[1]])

            elif curr_gene_bounds[0] < prev_gene_bounds[1]:
                # partially overlapping with another gene
                print('overlap', overlap_list, 'nonoverlap', nonoverlap_list)
                #print(overlap_list[-1][0],nonoverlap_list[-1][0], overlap_list[-1][1],nonoverlap_list[-1][1])
                '''
                if len(overlap_list)>0 and overlap_list[-1][0]>nonoverlap_list[-1][0] and overlap_list[-1][1]<nonoverlap_list[-1][1]:
                    print(overlap_list[-1], 'nonoverlap', nonoverlap_list[-1], '\n cur gene:', curr_gene_bounds)
                    nonoverlap_list[-1][0] = overlap_list[-1][1]
                    overlap_list.pop()
                    print(nonoverlap_list)
                    
                else:
                '''
                new_bound = [nonoverlap_list[-1][1], curr_gene_bounds[1]]
                nonoverlap_list[-1][1] = curr_gene_bounds[0]
                nonoverlap_list.append([new_bound[0], new_bound[1]])

            else:
                # not overlapping with another gene
                nonoverlap_list.append([gene_bounds_list[i][0], gene_bounds_list[i][1]])
    """
    
    if gene_bounds_sorted(gene_bounds_list):
        #range(130, len(gene_bounds_list[100:140]))
        for i in range(1, len(gene_bounds_list)):
            curr_gene_bounds = gene_bounds_list[i]

            print(i, prev_gene_bounds, curr_gene_bounds, overlap_list)

            if curr_gene_bounds[0]<prev_gene_bounds[0]:
                ''' Eg.     
                    16    |          68      |
                    |    21           |     89
                case1:       31   41
                             --del--
                case2:       31       |--del--|      92     
                curr_gene-> 31-start
                prev_gene-> 68-start    
                '''
                if curr_gene_bounds[1]<prev_gene_bounds[1]:
                    continue
                if curr_gene_bounds[1]>prev_gene_bounds[1]:
                    # previous gene was completely overlapping within cuurent gene,
                    # so replace previous gene by current (bigger) gene and put previous gene into overlap list
                    overlap_list.append([gene_bounds_list[i-1][0], gene_bounds_list[i-1][1]])
                    gene_bounds_list[i-1][0] = gene_bounds_list[i][0]
                    gene_bounds_list[i - 1][1] = gene_bounds_list[i][1]
                    prev_gene_bounds = gene_bounds_list[i]

            elif curr_gene_bounds[0] > prev_gene_bounds[0] and curr_gene_bounds[1] < prev_gene_bounds[1]:
                # completely within another gene
                overlap_list.append([curr_gene_bounds[0], curr_gene_bounds[1]])
                gene_bounds_list[i][0] = gene_bounds_list[i-1][0]
                gene_bounds_list[i][1] = gene_bounds_list[i-1][1]

            elif curr_gene_bounds[0] < prev_gene_bounds[1]:
                # partially overlapping with another gene
                temp = gene_bounds_list[i - 1][1]
                gene_bounds_list[i - 1][1] = curr_gene_bounds[0]
                gene_bounds_list[i][0] = temp
                prev_gene_bounds = gene_bounds_list[i]

            else:
                # not overlapping with another gene
    """
    print('Overlap', len(overlap_list), overlap_list)
    print('Non Overlap', len(nonoverlap_list), nonoverlap_list)

    # check if any of the overlapping_list intervals falls between the master intervals - remove them

    unique_data = [list(x) for x in set(tuple(x) for x in gene_bounds_list)] # get unique intervals
    print('Gene_bounds_list', gene_bounds_list)
    print('Unique data list', unique_data)

    # check if any of the overlapping_list intervals does not fall between the master intervals - remove them
    within_overlapping_intervals = []

    for small in overlap_list:
        for master in unique_data:
            if (small[0]==master[0] and small[1]==master[1]):
                continue
            if (small[0]>master[0] and small[1]<master[1]):
                if(small not in within_overlapping_intervals):
                    within_overlapping_intervals.append([small[0], small[1]])

    for t in overlap_list:
        if t not in within_overlapping_intervals:
            print('OUTSIDER', t)
    print('AHAA: ', len(within_overlapping_intervals), len(overlap_list))

    for o in within_overlapping_intervals:
        nonoverlap_list.append(o)  # append the overlapping intervals

    nonoverlap_list.sort(key=lambda tup: tup[0])

    #gene_bounds_list = [v for i, v in enumerate(gene_bounds_list) if i not in overlap]

    flat_data = sorted([x for sublist in nonoverlap_list for x in sublist])
    new_gene_intervals = [flat_data[i:i + 2] for i in range(0, len(flat_data), 2)]  # get non-overlapping intervals
    print("new intervals", len(new_gene_intervals), new_gene_intervals)

    return data

def manipulate(dataset, exon_start_offset, exon_end_offset):
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

    print(len(data))
    nonoverlapping_intervals = remove_overlapping_genes(data)
    print(len(data))
    '''
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

                if(exon['exon_ranges'][0] == gene_bounds[0] or
                    exon['exon_ranges'][1] == gene_bounds[1] or
                    exon['exon_ranges'][0]-exon_start_offset <= gene_bounds[0] or
                    exon['exon_ranges'][1]+exon_end_offset >= gene_bounds[1]):
                    continue # if exon bound coincides with gene bound, discard the exon

                ranges = [x - gene_bounds[0] + 1 for x in exon['exon_ranges']]
                exon_ranges.append(ranges)

            #if(gene['gene_strand']=='-'): #order the ranges from first to last exon
            #   exon_ranges = list(reversed(exon_ranges))

            if (len(exon_ranges)!=0):  #if there exist exons in the transcript
                exons_ranges_in_transcript.append(exon_ranges)

        if (len(exons_ranges_in_transcript) >= 1):
            nonoverlapping_exon_ranges_for_gene, ss, se = get_nonoverlapping_exon_bounds(exons_ranges_in_transcript)
            #print('non overlapping exons')
            #print(nonoverlapping_exon_ranges_for_gene)
            set_start_pos.update(ss)
            set_end_pos.update(se)

        #sub_start = exons_ranges_in_transcript[0][0][0]-exon_start_offset
        #sub_end = exons_ranges_in_transcript[0][-1][1]+exon_end_offset+1

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

    manipulate(dataset, 500, 30)