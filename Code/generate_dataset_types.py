import pandas as pd
from dataset_utils import *

columns = ['Type', 'Sequence', 'Interval_Indices', 'GeneID', 'Gene_Strand', 'Boundary_Position']
global finaldf
finaldf = pd.DataFrame(columns=columns)  # For sanity check

def boundary_exon_intron_3classification(exon_boundary_set_final,exon_boundary_final,
                                         within_exon_seq_intervals, within_intron_seq_intervals,
                                         gene, MAX_LENGTH, OFFSET_RANGE):
    '''
        Function to create training set [start intervals/ end intervals] (positive samples)
        :param exon_boundary_intervals_final: list of intervals
        :param exon_boundary_final: list of int containing the exon boundaries
        :param within_exon_seq_intervals: list of intervals
        :param within_intron_seq_intervals: list of intervals
        :param gene: gene information (JSON)
        :param MAX_LENGTH:
        :return: 2 lists
                list of str: training_set_x containing the DNA seqeunces
                list of int: training_set_y containing the exon boundary position

        Sequences containing Exon Boundary: Class 0
                            Purely Exonic Sequences: Class 1
                            Purely Intronic Sequences: Class 2
        '''
    sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                 [0] * len(exon_boundary_final), gene, MAX_LENGTH)
    sxexon, syexon = create_training_set(within_exon_seq_intervals,
                                         [1] * len(within_exon_seq_intervals), gene, MAX_LENGTH)
    sxintron, syintron = create_training_set(within_intron_seq_intervals,
                                             [2] * len(within_intron_seq_intervals), gene, MAX_LENGTH)

    # For Sanity Checking -
    dfboundary = sanity_check(sxboundary, exon_boundary_set_final, columns, 'Boundary', gene, exon_boundary_final)
    dfexon = sanity_check(sxexon, within_exon_seq_intervals, columns, 'Exon', gene)
    dfintron = sanity_check(sxintron, within_intron_seq_intervals, columns, 'Intron', gene)
    sanity_df = [dfboundary, dfexon, dfintron]

    return (sxboundary + sxexon + sxintron, syboundary + syexon + syintron), sanity_df

def boundary_orNot_2classification(exon_boundary_set_final,exon_boundary_final,
                                         within_exon_seq_intervals, within_intron_seq_intervals,
                                         gene, MAX_LENGTH, OFFSET_RANGE):
    sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                 [0] * len(exon_boundary_final), gene, MAX_LENGTH)
    sxexon, syexon = create_training_set(within_exon_seq_intervals,
                                         [1] * len(within_exon_seq_intervals), gene, MAX_LENGTH)
    sxintron, syintron = create_training_set(within_intron_seq_intervals,
                                             [1] * len(within_intron_seq_intervals), gene, MAX_LENGTH)

    # For Sanity Checking -
    dfboundary = sanity_check(sxboundary, exon_boundary_set_final, columns, 'Boundary', gene, exon_boundary_final)
    dfexon = sanity_check(sxexon, within_exon_seq_intervals, columns, 'Exon', gene)
    dfintron = sanity_check(sxintron, within_intron_seq_intervals, columns, 'Intron', gene)
    sanity_df = [dfboundary, dfexon, dfintron]

    return (sxboundary + sxexon + sxintron, syboundary + syexon + syintron), sanity_df

def find_boundary_Nclassification(exon_boundary_set_final, exon_boundary_final,
                                         within_exon_seq_intervals, within_intron_seq_intervals,
                                         gene, MAX_LENGTH, OFFSET_RANGE):
    '''
    Sequences containing Exon Boundary: Boundary point - OFFSET_RANGE[0] (0 indexed classes)
    '''
    sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                 [x - OFFSET_RANGE[0] for x in exon_boundary_final], gene, MAX_LENGTH)
    # For Sanity Checking -
    dfboundary = sanity_check(sxboundary, exon_boundary_set_final, columns, 'Boundary', gene, exon_boundary_final)
    return (sxboundary, syboundary), [dfboundary]

def boundaryCertainPoint_orNot_2classification(exon_boundary_set_final, exon_boundary_final,
                                         within_exon_seq_intervals, within_intron_seq_intervals,
                                         gene, MAX_LENGTH, OFFSET_RANGE):
    assert OFFSET_RANGE[0]==OFFSET_RANGE[1], "Have only one seq for particular boundary"
    sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                 [0] * len(exon_boundary_final), gene, MAX_LENGTH)
    sxexon, syexon = create_training_set(within_exon_seq_intervals,
                                         [1] * len(within_exon_seq_intervals), gene, MAX_LENGTH)
    sxintron, syintron = create_training_set(within_intron_seq_intervals,
                                             [1] * len(within_intron_seq_intervals), gene, MAX_LENGTH)

    # For Sanity Checking -
    dfboundary = sanity_check(sxboundary, exon_boundary_set_final, columns, 'Boundary', gene, exon_boundary_final)
    dfexon = sanity_check(sxexon, within_exon_seq_intervals, columns, 'Exon', gene)
    dfintron = sanity_check(sxintron, within_intron_seq_intervals, columns, 'Intron', gene)
    sanity_df = [dfboundary, dfexon, dfintron]

    return (sxboundary + sxexon + sxintron, syboundary + syexon + syintron), sanity_df

def seq_1classification(exon_boundary_set_final, exon_boundary_final,
                                         within_exon_seq_intervals, within_intron_seq_intervals,
                                         gene, MAX_LENGTH, OFFSET_RANGE):
    sxboundary, syboundary = create_training_set(exon_boundary_set_final,
                                                 [0] * len(exon_boundary_final), gene, MAX_LENGTH)

    # For Sanity Checking -
    dfboundary = sanity_check(sxboundary, exon_boundary_set_final, columns, 'Boundary', gene, exon_boundary_final)
    return (sxboundary, syboundary), [dfboundary]

def find_boundary_regression(exon_boundary_set_final, exon_boundary_final,
                                         within_exon_seq_intervals, within_intron_seq_intervals,
                                         gene, MAX_LENGTH, OFFSET_RANGE):
    sxboundary, syboundary = create_training_set(
        exon_boundary_set_final, exon_boundary_final, gene, MAX_LENGTH)

    # For Sanity Checking -
    dfboundary = sanity_check(sxboundary, exon_boundary_set_final, columns, 'Boundary', gene, exon_boundary_final)
    return (sxboundary, syboundary), [dfboundary]

