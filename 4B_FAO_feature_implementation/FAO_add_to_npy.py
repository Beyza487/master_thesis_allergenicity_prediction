import pandas as pd
import os
import shutil
import numpy as np


def copy_embeddings (pooling_methods, df678) :
    """
    Copie les NPY dans le répertoire de destination à partir du répertoire source
    
    The file path has to be changed for :
    -   dest_embedd = file path of the desired folder for the embeddings
    -   source_embedd = file path of the folder containing the NPY that we have to copy

    """
    for pooling in pooling_methods :
        
        
        dest_embedd = f'NPY_undersamp_training/combi2/NPY_training_{pooling}'
        #dest_embedd = f'NPY_test/combi2/NPY_test_{pooling}'
        
        if not os.path.exists(dest_embedd):
            os.makedirs(dest_embedd)
        
        for (index, row) in df678.iterrows() :
            name = row['query_id']
            
            source_embedd = f'../2_Pooling/undersamp_training_set/NPY_training_{pooling}/{name}_{pooling}.npy'
            #source_embedd = f'../2_Pooling/Test_set/NPY_test_{pooling}/{name}_{pooling}.npy'

            
            shutil.copy(source_embedd,dest_embedd)
    
    
# ============= RAJOUTER LES FEATURES POUR LES K-MERS ==============
def add_FAO_678_features (pooling_methods, df678):
    """ Rajoute les FAO features basés sur le second critère aux NPY
        
        The file path has to be changed for :
        -   embedd_file = file path to the embedding that was copied into the destination folder

    """
    
    for pooling in pooling_methods :
        for (_, row) in df678.iterrows():
            name = row['query_id']
            
            embedd_file = f'NPY_undersamp_training/combi2/NPY_training_{pooling}/{name}_{pooling}.npy'
            #embedd_file = f'NPY_test/combi2/NPY_test_{pooling}/{name}_{pooling}.npy'

            emb = np.load(embedd_file)
            
            features_to_add = np.array([row['n6_norm'],row['n7_norm'],row['n8_norm'], row['FAO_score1']])
            #features_to_add = np.array([row['n6_norm'],row['n7_norm'],row['n8_norm'], row['FAO_score2']])
            
            emb = np.concat((features_to_add,emb))
            np.save(embedd_file, emb)
                

# ========= RAJOUTER LES FEATURES DE FENETRES DE 80 AA ===========
def add_FAO_35_80_features (pooling_methods, df3580):
    """ Rajoute les FAO features basés sur le premier critère aux NPY

        The file path has to be changed for :
        -   embedd_file = file path to the embedding that was copied into the destination folder
        
        """
    
    for pooling in pooling_methods :
        for (_, row) in df3580.iterrows():
            name = row["protein_id"]
            
            embedd_file = f'NPY_undersamp_training/combi2/NPY_training_{pooling}/{name}_{pooling}.npy'
            #embedd_file = f'NPY_test/combi2/NPY_test_{pooling}/{name}_{pooling}.npy'

            emb = np.load(embedd_file)
            
            #features_to_add = np.array([row['n_hits_norm'],row['pident_max_35_filt']])
            features_to_add = np.array([row['n_hits_norm'],row['pident_max_raw']])
            
            emb = np.concat((features_to_add,emb))
            np.save(embedd_file, emb)


   
# ============ PARAMETERS =================
FAO_678_file = 'FAO_678mers_training_set.csv' 
FAO_80_35_file = 'FAO_80_35_training_set.csv'

# FAO_678_file = 'FAO_678mers_test_set.csv'
# FAO_80_35_file = 'FAO_80_35_test_set.csv'


df678 = pd.read_csv(FAO_678_file)
df3580 = pd.read_csv(FAO_80_35_file, sep=",")    

pooling_methods = ['max', 'mean', 'median', 'min', 'norm', 'std']


# ============ TO RUN ==============
copy_embeddings(pooling_methods, df678)
data = np.load('NPY_undersamp_training/combi2/NPY_training_max/Positive_BE_6_max.npy') #verification of dimension
# data = np.load('NPY_test/combi2/NPY_test_max/Positive_BE_15_max.npy') #verification of dimension
print(data.shape)

add_FAO_678_features (pooling_methods, df678)
data = np.load('NPY_undersamp_training/combi2/NPY_training_max/Positive_BE_6_max.npy') #verification of dimension
# data = np.load('NPY_test/combi2/NPY_test_max/Positive_BE_15_max.npy') #verification of dimension
print(data.shape)

add_FAO_35_80_features (pooling_methods, df3580)
data = np.load('NPY_undersamp_training/combi2/NPY_training_max/Positive_BE_6_max.npy') #verification of dimension
# data = np.load('NPY_test/combi2/NPY_test_max/Positive_BE_15_max.npy') #verification of dimension
print(data.shape)

