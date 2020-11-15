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
- `.yml` file for base path specification to data directory, model directory and visualization(Tensorboard and Attention) directory
- Writing and visulaization of model training logs using Tensorboard

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
**Input**: The input to the models consist of a DNA sequence of a particluar length, _L_. A single encoded training sample has the dimensions [_L, 4_].  
**Label**: The label (output) of the model depends on the Experiment Type. The label could be:  
- 0 (Contains a splice junction) / 1 (Does not contain a splice junction) : for 2-class classification
- 0 (containing splice junction) / 1 (exon) / 2 (intron) : for 3-class classification
- Any number \[1,_L_\] : for multi-class classification

### Usage  
- [generate_dataset_types.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/generate_dataset_types.py): Create the datasets tailered for the three experiments (Experiment I: boundaryCertainPoint_orNot_2classification, Experiment II: boundary_exon_intron_3classification and Experiment III: find_boundary_Nclassification).   
- [generate_dataset.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/generate_dataset.py): Containing the entire pipeline to generate the required type of dataset from the created JSON file for a particular chromosome.  
- [generate_entire_dataset.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/generate_entire_dataset.py): To generate the required type of dataset for the entire genome by calling `generate_dataset.py` for every chromosome.    
[encode.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/encode.py): Creates the one-hot encoding of the DNA sequences.  
- [process.sh](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/process.sh): Shell script written to automate the process of creating the dataset and encoding it for all the chromosomes over the full genome. The pipeline consists of creating the text and a JSON files for the indivisual chromosomes, the DNA sequences along with the corresponding labels, and then creating a one-hot encoding for the data.  
- [meta_info_stitch.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/meta_info_stitch.py): To obtain the meta-information about our dataset and writing it to file.  
- [subset.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/subset.py): Create a random subset of a specified number of samples from a larger dataset. 

## Model Training  
The model architectures that we implemented are:
- Convolutional Neural Network (CNN). 
- Long Short-Term Memory Network (LSTM). 
- DeepDeCode. 

### Usage  

- [train.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/train.py): Generalised training pipeline for classification or regression. The training pipeline consists of passing the hyper-parameters for the model, training for each batch, saving the trained models, writing to Tensorboard for visualization of the training process, writing training metrics (loss, performance metrics) to file.  
- [train_utils.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/train_utils.py): Contains utility function for training the deep learning models.  
- [train_attention.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/train_attentiom.py): Training pipeline for the DeepDeCode model, which uses attention. 
- [train_Kfold.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/train_Kfold.py): Training with K-fold cross validation to prevent model over-fitting. 
- [models.py](): Contains the various model architectures used for our experiment. The architectures implemented are CNN, LSTM and DeepDecode. 
- [test_model.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/test_model.py):To evaluate the trained models using the test set for classification tasks. 
- [test_regression.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/test_regression.py): To evaluate the trained models using the test set for classification tasks
- [hyperparameter_search.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/hyperparameter_search.py): To perform hyper-parameter search for the specified hyper-paramters for the various models over a search space. the results for each model are stored in a CSV file.  
- [metrics.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/metrics.py): Calculates the evaluation metrics for our models.

#### Config file format  

Config files are in `.json` format:
```
{
    "EXP_NAME": "Len100",                                            // training session name
    "TASK_TYPE": "classification",                                   // classification/regression
    "DATASET_TYPE": "boundaryCertainPoint_orNot_2classification",    // type of dataset being used for experiment
    "MODEL_NAME": "AttLSTM",                                         // model architecture
    "LOSS": "CrossEntropyLoss",                                      // loss
    "DATA": {
        "DATALOADER": "SequenceDataLoader",           // selecting data loader
        "DATA_DIR": "/end/100",                       // dataset path
        "BATCH_SIZE": 32,                             // batch size
        "SHUFFLE": true,                              // shuffle training data before splitting
        "NUM_WORKERS": 2                              // number of cpu processes to be used for data loading
    },
    "VALIDATION": {
        "apply": true,                    // whether to have a validation split (true/ false)
        "type": "balanced",               // type of validation split (balanced/mixed/separate)
        "val_split": 0.1,                 // size of validation dataset - float (portion of samples)
        "cross_val": 10                   // no. of folds for K-fold cross validation
    },
    "MODEL": {                            // Hyper-parameters for LSTM-based model
        "embedding_dim": 4,                
        "hidden_dim": 32,
        "hidden_layers": 2,
        "output_dim": 2,
        "bidirectional": true
    },
    "OPTIMIZER": {
        "type": "Adam",                   // optimizer type
        "lr": 0.1,                        // learning rate
        "weight_decay": 0.0,              // weight decay
        "amsgrad": true            
    },
    "LR_SCHEDULER": {
        "apply": true,            // whether to apply learning rate scheduling (true/ false)
        "type": "StepLR",         //type of LR scheduling.  More options available at https://pytorch.org/docs/stable/optim.html
        "step_size": 20,          // period of learning rate deca
        "gamma": 0.05             // multiplicative factor of learning rate decay
    },
    "TRAINER": {
        "epochs": 110,                               // number of training epochs
        "dropout": 0.1,                              // dropout
        "save_all_model_to_dir": false,              // whether to save models of every epoch (true/false)
        "save_model_to_dir": true,                   // whether to save any model to directory (true/false)
        "save_dir": "/all/att_start/",               // path for saving model checkpoints: './saved_models/all/att_start'
        "save_period": 250,                          // save checkpoints every save_period epochs
        "monitor": "acc",                            // metric to monitor for early stopping
        "early_stop": 50,                            // number of epochs to wait before early stop. Set 0 to disable
        "tensorboard": true,                         // enable tensorboard visualizations
        "tb_path": "chr21/end_half/Len30_model"      // path for tensoroard visualizations: './runs/chr21/end_half/Len30_model'
    }
}
```
This file is used during model training. The values in the required fields can be changed to set paths or parameters.  

#### Tensorboard Visualization
1. **Run Training**: Make sure that tensorboard option in the `config` file is turned on: `"tensorboard" : true`.  
2. **Open Tensorboard server**: In the command line, type `tensorboard --logdir runs/log/ --port 6006` at the project root, then server will open at `http://localhost:6006`. If you want to run the tensorboard visualizations from a remote server on your local machine using SSH, run the following on your local computer:
```
ssh -N -f -L localhost:16006:localhost:6006 <remote host username>@<remote host IP>
tensorboard --logdir runs --port 6006
```
The server will open at: `http://localhost:6006`  

## Visualizations 
Inference of biologically relevant information learnt by models in the genomic domain is a challenge. We identify sequence motifs in the genome that code for exon location. We explore various intrinsic and extrinsic visualization techniques to find the important sequence motifs informing the the existence of acceptor sites or donor sites.  
     
### Usage  
   
- [perturbation_test.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/perturbation_test.py): To perform the sequence pertubation test using the trained models. Various lengths of perturbations can be performed over the DNA sequences.  
- [visualize_attention.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/visualize_attention.py): Visualize the attention maps.  
- [graphs.py](https://github.com/asmitapoddar/NN-Genomics/blob/master/Code/graphs.py): Code to generate the graphs (distribution of exon position for Experiment III, variation of model accuracy for over length of the DNA sequence for Experiment I.

## License
This project is licensed under the MIT License. See LICENSE for more details. 
