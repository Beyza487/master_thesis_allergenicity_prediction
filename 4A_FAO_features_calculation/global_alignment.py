import pandas as pd
from Bio import Align
from Bio import SeqIO


""" Script to perform global alignment with the allergen database
    in order to filter proteins with >95% of sequence identity with a
    known protein
    """


subject_seqs = []
subject_id = []
query_seqs = []
query_id = []
global_percentages = []


# Getting the sequences and the ids of the allergens that are
# in the allergen database ('allergen_db.txt')
for subject in SeqIO.parse('allergen_db.txt', 'fasta') :
    subject_seqs.append(subject.seq)
    subject_id.append(subject.id)
print(f'nombre de prot dans la database : {len(subject_id)}')    
    


# Getting the sequences and the ids of the allergens that are in the
# Training set and the Test set ('allergen_from_sets.txt')
for query in SeqIO.parse('allergen_from_sets.txt', 'fasta'):
    query_seqs.append(query.seq)
    query_id.append(query.id)
print(f'nombre de prot dans Training + Test : {len(query_id)}')    



# =====================================================
# ================== GLOBAL ALIGNMENT =================
# =====================================================

aligner = Align.PairwiseAligner()
aligner.mode = 'global'

quer_align_id = []
subj_align_id = []


for (subj_seq, subj_id) in zip(subject_seqs, subject_id) :
    print(f"Processing {subj_id}")
    for (quer_seq, quer_id) in zip(query_seqs, query_id) :
        seq1 = quer_seq
        seq2 = subj_seq
    
        
        quer_align_id.append(quer_id)
        subj_align_id.append(subj_id)
        
        alignments = aligner.align(seq1, seq2) # aligning the sequences of the query and the subject
        best_alignment = alignments[0] # taking the best alignemnt done
        
        seqA_aligned = best_alignment[0] 
        seqB_aligned = best_alignment[1] 

        
        # Calculation of the sequence identity percentage
        identical = sum(1 for a, b in zip(seqA_aligned, seqB_aligned) if a == b)
        length = len(seqA_aligned)
        
        percent_identity = (identical / length) * 100
        global_percentages.append(percent_identity)
        

# Save the results in a table 
results_table = {
    'query' : quer_align_id,
    'subject' : subj_align_id,
    '% ident' : global_percentages
    }

df = pd.DataFrame(results_table)
df.to_csv('global_alignment.csv')



# ========================================================================
# ================== FILTERING PROTEINS TO 95% IDENTITY ==================
# ========================================================================


file = pd.read_csv('global_alignment.csv')
query_filt = []
subj_filt = []


# Getting the pairs of subject-query that have >=95% sequence identity 
for (index, row) in file.iterrows() :
    if row['query'] == row['subject'] :
        continue
    if row['% ident'] >= 95 :
        query_filt.append(row['query'])
        subj_filt.append(row['subject'])


# Saving the results in a table
results = {
    'query' : query_filt,
    'subject' : subj_filt
    }

df = pd.DataFrame(results)
df.to_csv('protein_to_filter.csv')
        


