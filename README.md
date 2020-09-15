# DeepDeCode: Deep Learning for Identifying Important Motifs in DNA Sequences

This folder contains the code for the dissertation titled 'DeepDeCode: Deep Learning for Identifying Important Motifs in DNA Sequences'.

The code folder contains the following Python files:  
-  create_json_cds.py: To create a JSON file (containing the keys: gene_id, gene_sequence, gene_strand, no_of_transcript, transcript_id, transcript_range, no_of_exons, exon_ranges) containing infomation about the CDS sequences from the GENCODE annotation file.    
- create_json.py: To create a JSON file (containing the keys: gene_id, gene_sequence, gene_strand, no_of_transcript, transcript_id, transcript_range, no_of_cds, cds_ranges) containing infomation about the exon sequences from the GENCODE annotation file.  
- dataset_utils.py: Contains the utility functions ('convert_list_to_interval', 'create_boundary_intervals', 'get_gene_seq', 'get_nonoverlapping_exon_bounds', 'remove_overlapping_genes', 'get_final_exon_intervals', 'get_negative_samples', 'create_training_set', 'write_to_file', 'sanity_check') for creating the positive and negative sequences for our dataset.   
- encode.py: Creates the one-hot encoding of the DNA sequences
- generate_dataset_types.py: Create the datasets tailered for the three experiments (Experiment I: boundaryCertainPoint_orNot_2classification, Experiment II: boundary_exon_intron_3classification and Experiment III: find_boundary_Nclassification). 
- generate_dataset.py: Containing the entire pipeline to generate the required type of dataset from the created JSON file for a particular chromosome. 
- generate_entire_dataset.py: To generate the required type of dataset for the entire genome by calling 'generate_dataset.py' for every chromosome. 
- graphs.py: Code to generate the graphs (distribution of exon position for Experiment III, variation of model accuracy for over length of the DNA sequence for Experiment I.  
- hyperparameter_search.py. 
meta_info_stitch.py
metrics.py
models.py
perturbation_test.py
process.sh
read_annotations.py
subset.py
test_model.py
test_regression.py
train_attention.py
train_Kfold.py
train_utils.py
train.py
training_pipeline_classification.py
training_pipeline_regression.py
visualize_attention.py
