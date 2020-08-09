import pathlib
import pandas as pd
import os
import itertools
import tqdm
import yaml
from dataset_utils import *
from encode import *
import json
from generate_dataset import *
from generate_dataset_types import *

curr_dir_path = str(pathlib.Path().absolute())

MAX_LENGTH = 100
NO_OFFSETS_PER_EXON = 5
MIN_INTRON_OFFSET = 8
MIN_EXON_OFFSET = 10
OFFSET_RANGE = [MIN_INTRON_OFFSET, MAX_LENGTH - MIN_EXON_OFFSET]
EXON_BOUNDARY = 'end'  # or 'end'
DATASET_TYPE = 'boundaryCertainPoint_orNot_2classification'
SEQ_TYPE = 'cds'  # 'cds'/'exons'
chrm = 'all'

WRITE_DATA_TO_FILE = True

DATA_LOG = True
SANITY_CHECK = True # For Wilfred
META_DATA = False  # For Upamanyu

with open('system_specific_params.yaml', 'r') as params_file:
    sys_params = yaml.load(params_file)

chrom_ignore = [3, 12]
f = open(sys_params['DATA_WRITE_FOLDER']+'/all/raw_data/genes_ignore.txt', "r")
genes_ignore = [line.strip() for line in f]

for i in range(1,23):  # full chromosome range = [1,23]
    if i in chrom_ignore:
        continue

    json_file = sys_params['DATA_WRITE_FOLDER']+'/all/raw_data/json_files/chr' + str(i) + "_cds_data.json"
    with open(json_file, "r") as f:
        dataset = json.load(f)

    all_genes_in_chrm = list(map(lambda x: dataset['main'][x]['gene_id'].strip('"'),
                             list(range(0, len(dataset['main'])))))

    # Do not consider the chromosomes that Wilfried wants ignored
    if len(set(all_genes_in_chrm).intersection(set(genes_ignore))) != 0:
        print('Skipping gene:', set(all_genes_in_chrm).intersection(set(genes_ignore)))
        continue

    NO_OF_GENES = len(dataset['main'])
    print('Processing Chromosome {}:'.format(i))

    if DATASET_TYPE == 'boundaryCertainPoint_orNot_2classification':
        NO_OFFSETS_PER_EXON = 1
        OFFSET_RANGE = [50,50]   #Note: Change boundary point here
        assert NO_OFFSETS_PER_EXON == 1
        assert OFFSET_RANGE[0] == OFFSET_RANGE[1]

    WRITE_TO_FILE_PATH = sys_params['DATA_WRITE_FOLDER'] + '/' + chrm +'/' + str(DATASET_TYPE) + '/'+ \
                         str(EXON_BOUNDARY) + '/' + str(MAX_LENGTH) + \
                         '/chrm{}_'.format(i) + SEQ_TYPE + '_' + \
                         str(EXON_BOUNDARY) + '_n' + str(NO_OF_GENES) + '_l' + str(MAX_LENGTH) + \
                         '_i' + str(OFFSET_RANGE[0]) + '_e' + str(MIN_EXON_OFFSET)


    # Generate dataset of required type for chromosome 'i'
    obj = GenerateDataset(DATASET_TYPE, EXON_BOUNDARY, MAX_LENGTH, NO_OFFSETS_PER_EXON, OFFSET_RANGE,
                 SEQ_TYPE, NO_OF_GENES, WRITE_DATA_TO_FILE, WRITE_TO_FILE_PATH, DATA_LOG,
                 SANITY_CHECK, META_DATA)
    obj.manipulate(dataset, 'chrm{}'.format(i))
    encode_seq(WRITE_TO_FILE_PATH)
