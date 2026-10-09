from Bio import SeqIO
import pandas as pd
import csv


""" Script to compute the FAO-derived features based on the second criteria:
    exact match of 6 consecutive residues in protein sequence with a known allergen
    """


def load_sequences_ref(ids, seqs, query_id, lookup):
    """ If the query protein is an allergen, then we substract this protein
    from the list containing all the allergen sequences in FAO_ref_fasta 
    If the query is a non-allergen, then we just use the list of FAO_ref_fasta as it is
    """
    
    filtered_seqs = []
    if query_id.startswith('N'):
        return seqs.copy()
    
    similar = lookup.get(query_id, set())
    
        
    for id_ref, seq_ref in zip(ids,seqs) : 
        if query_id == id_ref :
            continue
        if id_ref in similar :
            print(f'similar prot found : {query_id} -> {id_ref}')
            continue
        
        filtered_seqs.append(seq_ref)
        
    return filtered_seqs
    



def build_allergen_kmer_sets(allergen_seqs):
    """
    Construit les ensembles de k-mers allergènes (6,7,8)
    de la database de référence : FAO_ref_fasta.
    """
    kmers_allergen = {
        6: set(),
        7: set(),
        8: set()
    }

    for seq in allergen_seqs:
        L = len(seq)
        for k in [6, 7, 8]:
            for i in range(L - k + 1):
                kmers_allergen[k].add(seq[i:i+k])
                #print(f"Verif longueur : k={k} -> interval={len(seq[i:i+k])}")

    return kmers_allergen



def count_fao_kmers_vs_allergens(query_seq, allergen_seqs, kmers_allergen):
    """
    Compte les 6-,7-,8-mers FAO communs avec la base d'allergène
    de référence selon les règles d'inclusion totale FAO
    Les 6-mers entièrement présents dans les 7- et 8-mers sont exclus.
    Les 7-mers entièrement présents dans les 8-mers sont exclus. 
    """
    
    L = len(query_seq)

    # lists that will contain the positions of already found 7- & 8-mers
    used_8mers = []
    used_7mers = []
    
    # Number of 6-, 7- and 8-mers
    n8 = 0
    n7 = 0
    n6 = 0
    occ8 = 0
    occ7 = 0
    occ6 = 0


    # 1) 8-mers
    for i in range(L - 8 + 1):
        kmer = query_seq[i:i+8]
        if kmer in kmers_allergen[8]:
            positions = set(range(i, i+8))
            used_8mers.append(positions)
            n8 += 1
            occ8 += 1
    n8_norm = n8/L


    # 2) 7-mers exclusifs
    for i in range(L - 7 + 1):
        kmer = query_seq[i:i+7]
        if kmer not in kmers_allergen[7]:
            continue

        occ7 += 1
        positions = set(range(i, i+7))

        # exclure si totalement inclus dans un 8-mer commun
        if any(positions.issubset(p8) for p8 in used_8mers):
            continue

        used_7mers.append(positions)
        n7 += 1
    n7_norm = n7/L

    # 3) 6-mers exclusifs
    for i in range(L - 6 + 1):
        kmer = query_seq[i:i+6]
        if kmer not in kmers_allergen[6]:
            continue

        occ6 += 1
        positions = set(range(i, i+6))

        # exclure si inclus dans 8-mer commun
        if any(positions.issubset(p8) for p8 in used_8mers):
            continue

        # exclure si inclus dans 7-mer commun
        if any(positions.issubset(p7) for p7 in used_7mers):
            continue

        n6 += 1
    n6_norm = n6/L

    return n8, n7, n6, n8_norm, n7_norm, n6_norm, occ8, occ7, occ6




# ================= Parameters ==================== :
FAO_ref_fasta = "allergen_db.txt" #file path to the fasta containing the allergens of reference
#query_fasta = "Undersamp_training_set.txt"
query_fasta = "Test_set.txt"
data = [["query_id","n8","n7","n6","n8_norm","n7_norm","n6_norm", "FAO_score1", "FAO_score2"]] # name of the features to compute
protein_to_filter = 'protein_to_filter.csv' #proteins with >95% sequence identity to exclude


# reading and processing the file with the proteins to filter
prot_to_filt = pd.read_csv(protein_to_filter)
lookup = {}

for row in prot_to_filt.itertuples():
    lookup.setdefault(row.query, set()).add(row.subject)

print(f'{protein_to_filter} read')



# Getting the sequences from the allergen database
seqs = []
ids = []
for record in SeqIO.parse(FAO_ref_fasta, "fasta") :
    ids.append(record.id)
    seqs.append(record.seq)

print(f'{FAO_ref_fasta} read')


# Computing the FAO-feature of the proteins from the query fasta and 
for query in SeqIO.parse(query_fasta, 'fasta') :
    query_seq = query.seq
    query_id = query.id
    print(f'Processing {query_id}')
    
    # Filtering the allergen fasta to have only allergens with <95% identity with query
    allergen_seqs = load_sequences_ref(ids, seqs, query_id, lookup)


    # Generation of the sets containing all the k-mers of the FAO_ref_fasta 
    kmers_allergen = build_allergen_kmer_sets(allergen_seqs)
    
    n8, n7, n6, n8_norm, n7_norm, n6_norm, occ8, occ7, occ6 = count_fao_kmers_vs_allergens(query_seq, allergen_seqs, kmers_allergen)
    FAO_score1 = 6*n6_norm + 7*n7_norm + 8*n8_norm
    FAO_score2 = 36*n6_norm + 49*n7_norm + 64*n8_norm
    
    data.append([query_id,n8,n7,n6,n8_norm,n7_norm,n6_norm,FAO_score1, FAO_score2])
    

# Saving the computed features
with open('FAO_678mers_test_set.csv','w', newline='') as csv_file :
    writer = csv.writer(csv_file)
    writer.writerows(data)



