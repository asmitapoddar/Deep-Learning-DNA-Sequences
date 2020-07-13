import pathlib
import pandas as pd
from os import path
import json
import itertools
import Bio
import argparse
import os

curr_dir_path = str(pathlib.Path().absolute())
raw_data_path = curr_dir_path + "/raw_data/"

def gene_start_end_positions(chrm_ann):
    '''
    Function to read annotation for given chromosome and get [start, end] base pair position of genes
    :param chrm_ann: data frame
            Annotations for given chromosome
    :return: ndarray (num_genes, 2)
            Array containing [start, end] base pair position on chromosome for each gene.
    '''
    chr_gene = chrm_ann[chrm_ann['type'] == 'gene']
    gene_start_end_pos = pd.DataFrame(
        {'start_pos': chr_gene['start'], 'end_pos': chr_gene['end']})  # shape: (875,2) for chr21
    gene_start_end_pos.reset_index(drop=True, inplace=True)
    print('Start,end shape:', gene_start_end_pos.shape)
    return gene_start_end_pos

def get_indices_of_table(df, last_index):
    '''
    Get indices of the rows of the (sliced) data frame [which, in this case df corresponds to 'type' gene]
    :param df: data frame
    :return: list of indexes of the data frame
    '''
    indices = pd.Index.to_list(df.index)
    indices.append(last_index)

    return indices

def no_transcripts_per_gene(cur, nex, chrm_ann):
    '''
    Function to get number of transcripts between the cur and nex index of the chrm_ann data frame)
    :param cur: int: current index
    :param nex: int: next index
    :param chrm_ann: data_frame: annotation file for a particular chromosome
    :return: int: no. of transcripts found
    '''
    gene_table = chrm_ann.iloc[cur:nex]
    transcript_counts_per_gene=gene_table['type'].value_counts()['transcript']

    return transcript_counts_per_gene

def get_chunk(cur, nex, chrm_ann, type):
    '''
    Function to get subset of the annotation data frame according to condition
    :param cur: int: Start index
    :param nex: int: End index
    :param chrm_ann: data frame: Annotation data frame
    :return: data frame
            Subset of data frame
    '''
    table = chrm_ann.iloc[cur:nex]
    df = table[table['type'] == type]
    return df

def write_meta_data(chrm_no, my_dictionary, chrm_seq):
    '''
    Write chromosome meta-info to file (appending)
    :param chrm_no: int: chromosome no.
    :param my_dictionary: dict: dictionary containing the relevant chromosome info
    :param chrm_seq: str: Nucleotide seq for the chromosome
    :return: None
    '''
    print('Writing meta-info to log file...')
    log_file = open(raw_data_path + 'json_files/length_info.log', "a+")
    log_file.write('\n\nCHROMOSOME {} ---------'.format(chrm_no))
    log_file.write('\nNo. of genes in chromosome: {}'.format(len(my_dictionary['main'])))
    log_file.write('\nLength of chromosome sequence : {} nts.'.format(len(chrm_seq)))
    log_file.close()

def create_dict(chrm_seq, chrm_ann):
    '''
    Function to create the json file
    :param chrm_seq: string
            Nucleotide sequence of the chromosome
    :param chrm_ann: data frame
            Annotation file for the chromosome
    :return: dictionary
        Dictionary to be stored as the json file
    '''
    chrm_gene = chrm_ann[chrm_ann['type'] == 'gene']  #todo write length to file

    gene_ids = list(chrm_gene['0'])  #[l.strip('"') for l in

    gene_start_end_pos = gene_start_end_positions(chrm_ann)
    gene_bounds = list(zip(gene_start_end_pos.start_pos, gene_start_end_pos.end_pos))

    gene_strand = list(chrm_gene['strand'])
    indices = list(range(0, len(gene_start_end_pos)))
    gene_sequence = list(map(lambda x: chrm_seq[gene_start_end_pos['start_pos'][x]:gene_start_end_pos['end_pos'][x]]
            , indices))
    gene_indices = get_indices_of_table(chrm_gene, len(chrm_ann))

    my_dictionary = {'main': []}

    for i in range(0,len(chrm_gene)):
        cur = gene_indices[i]
        nex = gene_indices[i+1]
        lis = {}

        lis.update({'gene_id': gene_ids[i]})
        lis.update({'gene_strand': gene_strand[i]})
        lis.update({'gene_bounds': gene_bounds[i]})
        lis.update({'gene_sequence':gene_sequence[i]})  #take into account reverse complementarity

        no_transcripts = no_transcripts_per_gene(cur, nex, chrm_ann)
        lis.update({'no_of_transcripts': int(no_transcripts)})

        transcripts = []
        transcript = get_chunk(cur, nex, chrm_ann, 'transcript')
        transcript_ids = [l.strip('"') for l in list(transcript['1'])]
        transcript_ranges = list(zip(transcript['start'], transcript['end']))
        transcript_indices = get_indices_of_table(transcript, nex)

        for j in range(0,len(transcript_ranges)):

            list_transcript = {}

            list_transcript.update({'transcript_id': transcript_ids[j]})
            list_transcript.update({'transcript_range': transcript_ranges[j]})

            cds = get_chunk(transcript_indices[j], transcript_indices[j+1], chrm_ann, 'CDS')
            cds_ranges = list(zip(cds['start'], cds['end']))

            list_transcript.update({'no_of_cds': int(len(cds_ranges))})
            cds = []

            for er in cds_ranges:
                cds.append({'cds_ranges': er})

            list_transcript.update({'cds': cds})
            transcripts.append(list_transcript)

        lis.update({'transcripts': transcripts})
        my_dictionary['main'].append(lis)

    return my_dictionary

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create individual cds json files')
    parser.add_argument('--no', type = str, help='chromosome number')
    args = parser.parse_args()

    print('Starting JSON processing Chromosome {}'.format(args.no))
    chrm_txt_file = "chr{}.txt".format(args.no)
    chrm_ann_file = "chr{}_annotations.csv".format(args.no)

    file_object = open(raw_data_path + 'txt/' + chrm_txt_file, "r")
    chrm_seq = file_object.read()
    chrm_ann = pd.read_csv(raw_data_path + 'annotations/' + chrm_ann_file, sep='\t')

    my_dictionary = create_dict(chrm_seq, chrm_ann)

    if not os.path.exists(raw_data_path+'json_files'):
        os.makedirs(raw_data_path+'json_files')
    print('Writing to json file...')
    with open(raw_data_path+'json_files/'+'chr'+str(args.no)+'_cds_data.json', 'w') as file:
        json.dump(my_dictionary, file)
    write_meta_data(args.no, my_dictionary, chrm_seq)
    print('Finished JSON processing Chromosome {}'.format(args.no))