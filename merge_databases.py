from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd


""" Generation of the merged database from the training set
    and the independent sets of SEP-AlgPro. 
    
    + Generation of a file with the old names of the proteins
    and their new ones (id_correspondance.csv)

"""


# ===============================================
# ================== FILES ======================
# ===============================================


files = ['Training.txt','Ind1.txt', 'Ind2.txt', 'Ind3.txt', 'Ind4.txt'] # files of the sets from SEP-AlgPro
complete_file = 'complete_db.txt' # Database created by the merging of the ones of SEP-AlgPro
correspondance_file = 'id_correspondance.csv' # file with the old names and the new ones



# ===============================================
# ===============================================


index_pos_new_db = 1
index_neg_new_db = 1
new_records = []

correspondance = {"old_file":[],
                  "old_id":[],
                  "new_id":[]}


for file in files :
    print(f'Working on file {file}')
    file_path = f'initial_databases/{file}'
    for record in SeqIO.parse(file_path, 'fasta') :
        
        if record.id.startswith('n') or record.id.startswith('N') :
            new_neg_id = f'Negative_BE_{index_neg_new_db}'
            new_neg_record = SeqRecord(Seq(record.seq), id=new_neg_id, description="")
            new_records.append(new_neg_record)
            
            index_neg_new_db += 1
            
            correspondance["old_file"].append(file)
            correspondance["old_id"].append(record.id)
            correspondance["new_id"].append(new_neg_id)
            
        elif record.id.startswith('a') or record.id.startswith('P') :
            new_pos_id = f'Positive_BE_{index_pos_new_db}'
            new_pos_record = SeqRecord(Seq(record.seq), id=new_pos_id, description="")
            new_records.append(new_pos_record)
            
            index_pos_new_db += 1
            
            correspondance["old_file"].append(file)
            correspondance["old_id"].append(record.id)
            correspondance["new_id"].append(new_pos_id)

        else :
            print(f'ERROR — protein name : {record.id}')




with open(complete_file, 'w') as new_fasta :
    SeqIO.write(new_records, new_fasta, 'fasta')


df = pd.DataFrame(correspondance)
df.to_csv(correspondance_file)



    
