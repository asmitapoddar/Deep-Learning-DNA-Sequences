# DeepDeCode: Deep Learning for Identifying Important Motifs in DNA Sequences

This is the code base for the dissertation: 'DeepDeCode: Deep Learning for Identifying Important Motifs in DNA Sequences'. The code has been written in Python and the deep learning framework used is PyTorch.

## Requirements
- Python >= 3.6   
- PyTorch >= 1.5.0  
- tensorboard >= 1.14 (see Tensorboard Visualization)  
- argparse >= 1.1  
- biopython >=1.76  
- json >= 2.0.9  
- logomaker >= 0.8  
- matplotlib >= 3.1.0  
- numpy >= 1.18.5  
- pandas >= 0.24.2  
- seaboarn >= 0.10.1  
- sklearn >= 0.21.3  
- yaml >= 5.1.2  

## Features
- `.json` file for convenient parameter tuning
- `.yml` file for path specification to data directory, model directory and visualization(Tensorboard and Attention) directory
- Checkpoint saving and resuming.

## Data

The human DNA sequence data from the Dec. 2013 assembly of the human genome ([hg38, GRCh38 Genome Reference Consortium Human Reference 38](http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/)) was obtained from the [UCSC Genome Browser](https://genome.ucsc.edu/) [80] in the FASTA format (a text-based format for representing nucleotide sequences, in which nucleotides are represented
using single-letter codes) for each chromosome.
We obtained the location of exons within the DNA sequences from the latest release ([Release 34, GRCh38.p13](https://www.gencodegenes.org/human/)) of GENCODE annotations [81] in the Gene Transfer Format (GTF), which contains comprehensive gene annotations on the reference chromosomes in the human genome.

## Pre-processing Steps
The aim of pre-processing is to extract the relevant parts of the genome to get high quality data for our models. The following pre-processing steps were performed to create the datasets:

### Usage   
- [read_annotations.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/read_annotations.py): Read annotation file for human genome sequence GRCh38 and get annotations for specified chromosome
- [create_json_cds.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/create_json_cds.py): To create a JSON file (containing the keys: `gene_id`, `gene_sequence`, `gene_strand`, `no_of_transcript`, `transcript_id`, `transcript_range`, `no_of_exons`, `exon_ranges`) containing infomation about the _Coding DNA Sequences (CDS)_ sequences from the GENCODE annotation file.    
- [create_json.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/create_json.py): To create a JSON file containing the same attributes as above for _exon sequences_ from the GENCODE annotation file.  
- [dataset_utils.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/dataset_utils.py): Contains the utility functions for creating the positive and negative sequences for our dataset.   

## Dataset Creation
The final dataset for the experiments were created after pre-processing the data. Standard data format used among all my datasets:  
**Input**: The input to the models consist of a DNA sequence of a particluar length, _l_. A single encoded training sample has the dimensions [_l, 4_]:

**Label**: The label (output) of the model depends on the Experiment Type. The label could be:  
- 0 (Contains a splice junction) / 1 (Does not contain a splice junction) : for binary classification
- 0 (containing splice junction) / 1 (exon) / 2 (intron) : for 3-class classification
- Any number \[1,l\] : for multi-class classification

### Usage  
- [generate_dataset_types.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/generate_dataset_types.py): Create the datasets tailered for the three experiments (Experiment I: boundaryCertainPoint_orNot_2classification, Experiment II: boundary_exon_intron_3classification and Experiment III: find_boundary_Nclassification).   
- [generate_dataset.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/generate_dataset.py): Containing the entire pipeline to generate the required type of dataset from the created JSON file for a particular chromosome.  
- [generate_entire_dataset.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/generate_entire_dataset.py): To generate the required type of dataset for the entire genome by calling `generate_dataset.py` for every chromosome.    
[encode.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/encode.py): Creates the one-hot encoding of the DNA sequences.  
- [process.sh](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/process.sh): Shell script written to automate the process of creating the dataset and encoding it for all the chromosomes over the full genome. The pipeline consists of creating the text and a JSON files for the indivisual chromosomes, the DNA sequences along with the corresponding labels, and then creating a one-hot encoding for the data.  
- [meta_info_stitch.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/meta_info_stitch.py): To obtain the meta-information about our dataset and writing it to file.  

## Model Training 

### Usage  

#### Config file format  

Config files are in `.json` format:
```
{
    "EXP_NAME": "Len100",    // training session name
    "TASK_TYPE": "classification",    // classification/regression
    "DATASET_TYPE": "boundaryCertainPoint_orNot_2classification",    // classification/regression
    "MODEL_NAME": "AttLSTM",
    "LOSS": "CrossEntropyLoss",
    "DATA": {
        "DATALOADER": "SequenceDataLoader",
        "DATA_DIR": "/end/100",
        "BATCH_SIZE": 32,
        "SHUFFLE": true,
        "NUM_WORKERS": 2
    },
    "VALIDATION": {
        "apply": true,
        "type": "balanced",
        "val_split": 0.1,
        "cross_val": 10
    },
    "MODEL": {
        "embedding_dim": 4,
        "hidden_dim": 32,
        "hidden_layers": 2,
        "output_dim": 2,
        "bidirectional": true
    },
    "OPTIMIZER": {
        "type": "Adam",
        "lr": 0.1,
        "weight_decay": 0.0,
        "amsgrad": true
    },
    "LR_SCHEDULER": {
        "apply": true,
        "type": "StepLR",
        "step_size": 20,
        "gamma": 0.05
    },
    "TRAINER": {
        "epochs": 110,
        "dropout": 0.1,
        "save_all_model_to_dir": false,
        "save_model_to_dir": true,
        "save_dir": "/all/att_start/",
        "save_period": 250,
        "monitor": "acc",
        "early_stop": 50,
        "tensorboard": true,
        "tb_path": "/mnt/sdc/asmita/Code/runs/chr21/end_half/Len30_balanced_SimpleLSTM[4,64,2,2]_BS8_Adam_25-08_15:24"
    }
}
```
- [test_model.py]():
- [test_regression.py]():
- [train_attention.py]():
- [train_Kfold.py]():
- [train_utils.py]():
- [train.py]():
- [hyperparameter_search.py](): 
- [models.py]():
- [metrics.py]():


Tensorboard Visualization

## Visualizations  

### Usage  
   
- [perturbation_test.py]():
subset.py
- [visualize_attention.py]():
- [graphs.py](): Code to generate the graphs (distribution of exon position for Experiment III, variation of model accuracy for over length of the DNA sequence for Experiment I.
