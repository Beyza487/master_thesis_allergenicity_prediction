from Bio import SeqIO
import os


"""
Filter the proteins from the Ind2 dataset for which we don't
have the embeddings 

"""


# ========================================
# ============ PARAMETERS ================
# ========================================

file = 'initial_databases/Ind2.txt' # initial Ind2 dataset
Ind2_folder = '../1A_Embeddings_Celeste/NPY_Ind2/' # file path to the folder containing the NPY of Ind2
NPY_files = sorted(os.listdir(Ind2_folder))



missing = []
records = []

for record in SeqIO.parse(file, 'fasta') :
    name = f'{record.id}.npy'
    if name not in NPY_files :
        missing.append(name)
    else :
        records.append(record)

print(len(missing))


# Creation of the nex fasta file 
with open('initial_databases/Ind2_new.txt', 'w') as f :
    SeqIO.write(records, f, 'fasta')
    







