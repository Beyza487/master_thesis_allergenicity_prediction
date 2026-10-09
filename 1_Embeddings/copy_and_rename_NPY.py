import os
import pandas as pd
import numpy as np
import shutil
from Bio import SeqIO


def get_test_set_NPY (correspondance, test_set_fasta):
    """ Copy and renaming the embeddings of the proteins in
        the test set """
    
    if not os.path.exists('NPY_test/') :
        os.makedirs('NPY_test/')
        
    for record in SeqIO.parse(test_set_fasta, 'fasta'):
        row = correspondance[correspondance['new_id'] == record.id].iloc[0]
        old_name = row['old_id']
        old_file = os.path.splitext(row['old_file'])[0]
        
        shutil.copy(f'../1A_Embeddings_Celeste/NPY_{old_file}/{old_name}.npy', f'NPY_test/{record.id}.npy')
    

def get_training_set_NPY (correspondance, training_set_fasta):
    """ Copy and renaming the embeddings of the proteins in
        the training set """
    
    if not os.path.exists('NPY_training/') :
        os.makedirs('NPY_training/')
        
    for record in SeqIO.parse(training_set_fasta, 'fasta'):
        row = correspondance[correspondance['new_id'] == record.id].iloc[0]
        old_name = row['old_id']
        old_file = os.path.splitext(row['old_file'])[0]
        
        shutil.copy(f'../1A_Embeddings_Celeste/NPY_{old_file}/{old_name}.npy', f'NPY_training/{record.id}.npy')
    

def get_undersampling_set_NPY (correspondance, undersamp_set_fasta):
    """ Copy and renaming the embeddings of the proteins in
        the undersampled training set """
    
    if not os.path.exists('NPY_undersamp_training/') :
        os.makedirs('NPY_undersamp_training/')
        
    for record in SeqIO.parse(undersamp_set_fasta, 'fasta'):
        row = correspondance[correspondance['new_id'] == record.id].iloc[0]
        old_name = row['old_id']
        old_file = os.path.splitext(row['old_file'])[0]
        
        shutil.copy(f'../1A_Embeddings_Celeste/NPY_{old_file}/{old_name}.npy', f'NPY_undersamp_training/{record.id}.npy')
    



if __name__ == '__main__' :
    
    correspondance = pd.read_csv('id_correspondance.csv') #columns = ['old_file', 'old_id', 'new_id']
    test_set_fasta = 'Test_set.txt'
    training_set_fasta = 'Training_set.txt'
    undersamp_set_fasta = 'Undersamp_training_set.txt'


    get_test_set_NPY(correspondance, test_set_fasta)
    get_training_set_NPY (correspondance, training_set_fasta)
    get_undersampling_set_NPY (correspondance, undersamp_set_fasta)
    
    
