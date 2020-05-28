import pathlib
import pandas as pd
from os import path
import json
import itertools
import tqdm
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna

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

def remove_within_bounds_gene(data):
    gene_bounds_list = []
    for gene in data:
        gene_bounds = gene['gene_bounds']
        gene_bounds_list.append(gene_bounds)

    return

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
    set_start = set()
    set_end = set()

    #remove_within_bounds_genes(data)

    # Iterating through all genes of the chromosome
    for gene in data[0:3]:
        print(gene['gene_id'])
        gene_sequence = get_gene_seq(gene['gene_sequence'], gene['gene_strand'])
        gene_bounds = gene['gene_bounds']

        exons_ranges_in_transcript = []

        # Iterating through all transcripts of the gene
        for transcript in gene["transcripts"]:
            print(transcript['transcript_id'])
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

            if(gene['gene_strand']=='-'): #order the ranges from first to last exon
                exon_ranges = list(reversed(exon_ranges))

            if (len(exon_ranges)!=0):  #if there exist exons in the transcript
                exons_ranges_in_transcript.append(exon_ranges)
                continue
        print('exon ranges in transcript')
        print(exons_ranges_in_transcript)
        nonoverlapping_exon_ranges_for_gene, ss, se = get_nonoverlapping_exon_bounds(exons_ranges_in_transcript)

        set_start.update(ss)
        set_start.update(se)

        if(len(exons_ranges_in_transcript) >= 1):
            print('cool'+str(len(exons_ranges_in_transcript)))
            #print(exons_ranges_in_transcript[0], gene_bounds)

            #working with exons in the first transcript
            sub_start = exons_ranges_in_transcript[0][0][0]-exon_start_offset
            sub_end = exons_ranges_in_transcript[0][-1][1]+exon_end_offset+1
            #print(len(gene_sequence),sub_end, sub_start, sub_start- sub_end)

            training_set_x.append(get_gene_seq(gene_sequence[sub_start:sub_end], gene['gene_strand']))
            training_set_y.append(exon_ranges)

    print('No. of samples in training set:', len(training_set_y))

    df = pd.DataFrame()
    df['x'] = training_set_x
    df['y'] = training_set_y
    #df.to_csv(data_path+'chr21_training_data.csv', index = False)
    length_seq = [len(i) for i in training_set_x]
    print(length_seq)
    print(max(length_seq))
    print(min(length_seq))

    return

if __name__ == "__main__":

    file = data_path+"chr21_data.json"
    with open(file, "r") as f:
        dataset = json.load(f)

    manipulate(dataset, 500, 30)