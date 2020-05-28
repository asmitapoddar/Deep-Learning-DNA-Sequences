import pathlib
import pandas as pd
from os import path
import json
import itertools
import Bio

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

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
    Function to
    :param df: data frame
    :return: list of indexes of the data frame
    '''
    indices = pd.Index.to_list(df.index)
    indices.append(last_index)

    return indices

def no_transcripts_per_gene(cur, nex, chrm_ann):
    '''
    Function to
    :param cur:
    :param nex:
    :param chrm_ann:
    :return:
    '''
    gene_table = chrm_ann.iloc[cur:nex]
    transcript_counts_per_gene=gene_table['type'].value_counts()['transcript']

    return transcript_counts_per_gene

def get_chunk(cur, nex, chrm_ann, type):
    '''
    Function to get subset of the annotation data frame according to condition
    :param cur: int
            Start index
    :param nex: int
            End index
    :param chrm_ann: data frame
            Annotation data frame
    :return: data frame
            Subset of data frame
    '''
    table = chrm_ann.iloc[cur:nex]
    df = table[table['type'] == type]
    return df

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
    chrm_gene = chrm_ann[chrm_ann['type'] == 'gene']

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

            exon = get_chunk(transcript_indices[j], transcript_indices[j+1], chrm_ann, 'exon')
            exon_ranges = list(zip(exon['start'], exon['end']))

            list_transcript.update({'no_of_exons': int(len(exon_ranges))})
            exons = []
            list_exon_range = {}

            for er in exon_ranges:
                list_exon_range.update({'exon_ranges': er})
                exons.append({'exon_ranges': er})

            list_transcript.update({'exons': exons})
            transcripts.append(list_transcript)

        lis.update({'transcripts': transcripts})
        my_dictionary['main'].append(lis)

    return my_dictionary

if __name__ == "__main__":

    chrm_txt_file = "chr21.txt"
    chrm_ann_file = "chr21_annotations.csv"

    file_object = open(data_path + chrm_txt_file, "r")
    chrm_seq = file_object.read()
    chrm_ann = pd.read_csv(data_path + chrm_ann_file, sep='\t')

    #with open(data_path+"data.json") as f:
    #        data = json.load(f)
    my_dictionary = create_dict(chrm_seq, chrm_ann)
    with open(data_path+chrm_txt_file.replace('.txt','')+'_data.json', 'w') as file:
        json.dump(my_dictionary, file)