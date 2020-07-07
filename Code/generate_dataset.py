import pathlib
import pandas as pd
import os
import itertools
import tqdm
import yaml
from dataset_utils import *
from encode import *
from generate_dataset_types import *

curr_dir_path = str(pathlib.Path().absolute())
data_path = curr_dir_path + "/Data/"

class GenerateDataset():
    def __init__(self, DATASET_TYPE, EXON_BOUNDARY, MAX_LENGTH, NO_OFFSETS_PER_EXON, OFFSET_RANGE,
                 SEQ_TYPE, NO_OF_GENES, WRITE_DATA_TO_FILE, WRITE_TO_FILE_PATH, DATA_LOG,
                 SANITY_CHECK, META_DATA = False):
        self.DATASET_TYPE = DATASET_TYPE
        self.EXON_BOUNDARY = EXON_BOUNDARY
        self.MAX_LENGTH = MAX_LENGTH
        self.NO_OFFSETS_PER_EXON = NO_OFFSETS_PER_EXON
        self.OFFSET_RANGE = OFFSET_RANGE
        self.SEQ_TYPE = SEQ_TYPE
        self.NO_OF_GENES = NO_OF_GENES
        self.WRITE_DATA_TO_FILE = WRITE_DATA_TO_FILE
        self.WRITE_TO_FILE_PATH = WRITE_TO_FILE_PATH
        self.DATA_LOG = DATA_LOG
        self.SANITY_CHECK = SANITY_CHECK
        self.META_DATA = META_DATA

    def manipulate(self, dataset, chrm):
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
        for gene in data[0:self.NO_OF_GENES]:  #Note: controlling how many genes to use

            print(gene['gene_id'])
            gene_sequence = get_gene_seq(gene['gene_sequence'], gene['gene_strand'])
            gene_bounds = gene['gene_bounds']

            exons_ranges_in_transcript = []
            # Iterating through all transcripts of the gene
            for transcript in gene["transcripts"]:
                exon_ranges = []
                # Iterating through all exons of the transcript for the gene
                for exon in transcript[self.SEQ_TYPE]:
                    ranges = [x for x in exon[self.SEQ_TYPE+'_ranges']]
                    exon_ranges.append(ranges)
                if (len(exon_ranges) != 0):  # if there exist exons in the transcript
                    exons_ranges_in_transcript.append(exon_ranges)

            # if there exists at least one exon in transcript----
            if (len(exons_ranges_in_transcript) >= 1):
                nonoverlapping_exon_ranges_for_gene = get_nonoverlapping_exon_bounds(exons_ranges_in_transcript)

                # get exon start & end intervals - with offsets: list of intervals todo: might change for single offset exons
                exon_boundary_list, exon_boundaries_y = create_boundary_intervals(nonoverlapping_exon_ranges_for_gene, self.EXON_BOUNDARY,
                                                                                  self.MAX_LENGTH, self.NO_OFFSETS_PER_EXON, self.OFFSET_RANGE)
                exon_intervals_list = convert_list_to_interval(nonoverlapping_exon_ranges_for_gene)

                exon_boundary_list = sorted(exon_boundary_list)
                exon_intervals_list = sorted(exon_intervals_list)

                # POSITIVE SAMPLES
                exon_boundary_set_final, exon_boundary_y_final = get_final_exon_intervals(exon_boundary_list,
                                                                                           exon_boundaries_y,
                                                                                           exon_intervals_list,
                                                                                           nonoverlapping_gene_intervals,
                                                                                           self.NO_OFFSETS_PER_EXON,
                                                                                           self.EXON_BOUNDARY)

                # NEGATIVE SAMPLES
                within_exon_seq_intervals, within_intron_seq_intervals = get_negative_samples(exon_intervals_list,
                                                                                              nonoverlapping_gene_intervals,
                                                                                              self.MAX_LENGTH, self.NO_OFFSETS_PER_EXON)

                # Get training set info
                len_exon += len(within_exon_seq_intervals)
                len_intron += len(within_intron_seq_intervals)
                len_boundary += len(exon_boundary_set_final)
                print('Dataset stats (#boundary, #exon, #intron): ', len_boundary, len_exon, len_intron)

                all_exon_intervals = all_exon_intervals.append(
                    {'Gene': gene['gene_id'], 'Exon_Intervals': exon_intervals_list}, ignore_index=True)

                # TTRAINING SET CREATION ----
                (train_x, train_y), sanity_data = eval(self.DATASET_TYPE)(exon_boundary_set_final, exon_boundary_y_final,
                                                             within_exon_seq_intervals, within_intron_seq_intervals,
                                                             gene, self.MAX_LENGTH, self.OFFSET_RANGE)
                training_x.extend(train_x)
                training_y.extend(train_y)

                if self.SANITY_CHECK:
                    finaldf = finaldf.append(sanity_data)

        dataset_path = chrm_path + self.DATASET_TYPE
        # Write to file ----
        if self.WRITE_DATA_TO_FILE:
            print('WRITE_TO_FILE_PATH', self.WRITE_TO_FILE_PATH)
            if not os.path.exists(self.WRITE_TO_FILE_PATH):
                os.makedirs(self.WRITE_TO_FILE_PATH)
            write_to_file(training_y, self.WRITE_TO_FILE_PATH + '/y_label_'+self.EXON_BOUNDARY)
            write_to_file(training_x, self.WRITE_TO_FILE_PATH + '/dna_seq_'+self.EXON_BOUNDARY)

        if self.DATA_LOG:
            self.write_data_log(training_x, len_boundary, len_exon, len_intron)

        if self.SANITY_CHECK:
            if not os.path.exists(self.WRITE_TO_FILE_PATH):
                os.makedirs(self.WRITE_TO_FILE_PATH)
            finaldf.to_csv(self.WRITE_TO_FILE_PATH +'/sanity_'+str(self.MAX_LENGTH)+'.csv', header=columns, index = False)

        if self.META_DATA:
            self.write_meta_data()

        print('No. of samples in training set:', len(training_x))
        print("no. positive samples", len_boundary)
        print("no. exon samples", len_exon)
        print("no. intron samples", len_intron)

        return

    def write_data_log(self, training_x, len_boundary, len_exon, len_intron):
        if not os.path.exists(self.WRITE_TO_FILE_PATH):
            os.makedirs(self.WRITE_TO_FILE_PATH)
        with open(self.WRITE_TO_FILE_PATH + '/info.log', 'w+') as f:
            f.write('DATASET TYPE: ' + str(self.DATASET_TYPE))
            f.write('\nSEQUENCE TYPE: ' + str(self.SEQ_TYPE))
            f.write('\nEXON BOUNDARY: ' + str(self.EXON_BOUNDARY))
            f.write('\nMAX SEQUENCE LENGTH = ' + str(self.MAX_LENGTH))
            f.write('\nNO_OFFSETS_PER_EXON = ' + str(self.NO_OFFSETS_PER_EXON))
            f.write('\nOFFSET_RANGE (from exon boundary into intronic region) = ' + str(self.OFFSET_RANGE))
            f.write('\n\nTotal no. of samples in training set = ' + str(len(training_x)))
            f.write("\nNo. samples containing intron-exon boundary = " + str(len_boundary))
            #todo: len_exon, len_intron for those not containing these
            f.write("\nNo. of pure exon samples = " + str(len_exon))
            f.write("\nNo. of pure intron samples = " + str(len_intron))

    def write_meta_data(self, chrm_path, chrm, nonoverlapping_gene_intervals, all_exon_intervals):
        if not os.path.exists(chrm_path):
            os.makedirs(chrm_path)
        write_to_file(nonoverlapping_gene_intervals, chrm_path + chrm + '_nonoverlapping_gene_intervals.txt')
        all_exon_intervals.to_csv(chrm_path + chrm + '_exon_intervals.csv', header=['Gene', 'Exon_Intervals'],
                                  index=False, doublequote=False)


if __name__ == "__main__":

    MAX_LENGTH = 100
    NO_OFFSETS_PER_EXON = 5
    MIN_INTRON_OFFSET = 60
    MIN_EXON_OFFSET = 10
    OFFSET_RANGE = [MIN_INTRON_OFFSET, MAX_LENGTH - MIN_EXON_OFFSET]
    EXON_BOUNDARY = 'start'  # or 'end'
    DATASET_TYPE = 'boundaryCertainPoint_orNot_2classification'  # 'classification'  # or 'regression'
    SEQ_TYPE = 'cds'  # 'cds'/'exons'
    #NO_OF_GENES = 3

    WRITE_DATA_TO_FILE = True

    DATA_LOG = True
    SANITY_CHECK = True # For Wilfred
    META_DATA = False  # For Upamanyu

    with open('system_specific_params.yaml', 'r') as params_file:
        sys_params = yaml.load(params_file)

    file = sys_params['DATA_READ_FOLDER'] + "/chr21_" + SEQ_TYPE + "_data.json"
    with open(file, "r") as f:
        dataset = json.load(f)

    NO_OF_GENES = len(dataset['main'])

    if DATASET_TYPE == 'boundaryCertainPoint_orNot_2classification':
        NO_OFFSETS_PER_EXON = 1
        OFFSET_RANGE = [99,99]   #Note: Change boundary point here
        assert NO_OFFSETS_PER_EXON == 1
        assert OFFSET_RANGE[0] == OFFSET_RANGE[1]

    WRITE_TO_FILE_PATH = sys_params['DATA_WRITE_FOLDER'] + '/chrm21' + '/' + str(DATASET_TYPE) + '/' + SEQ_TYPE + '_' + \
                         str(EXON_BOUNDARY) + '_n' + str(NO_OF_GENES) + '_l' + str(MAX_LENGTH) + \
                         '_o' + str(OFFSET_RANGE[0])

    obj = GenerateDataset(DATASET_TYPE, EXON_BOUNDARY, MAX_LENGTH, NO_OFFSETS_PER_EXON, OFFSET_RANGE,
                 SEQ_TYPE, NO_OF_GENES, WRITE_DATA_TO_FILE, WRITE_TO_FILE_PATH, DATA_LOG,
                 SANITY_CHECK, META_DATA)
    obj.manipulate(dataset, 'chrm21')
    encode_seq(WRITE_TO_FILE_PATH)
