import os
import subprocess
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


""" Script to calculate the FAO-derived features based on the first criteria:
    sequence identity >35% over a window of 80 amino acids with a known protein
    """

def make_blast_db(input_makeblast, FAO_blast_db):
    makeblast_cmd = [
        "/opt/homebrew/bin/makeblastdb",
        "-in", input_makeblast,
        "-dbtype", "prot",
        "-out", FAO_blast_db
    ]
    
    subprocess.run(makeblast_cmd, check=True)


# -----------------------------
# 1️⃣ Sliding window
# -----------------------------

def generate_windows(record, window_size=80):
    """ Generation of windows of 80 amino acids """
    
    windows = []
    seq = str(record.seq)
    protein_id = record.id
    
    for i in range(len(seq) - window_size + 1):
        window_seq = seq[i:i+window_size]
        window_id = f"{protein_id}_win_{i}"
        windows.append(SeqRecord(Seq(window_seq), id=window_id, description=""))
    
    return windows


# -----------------------------
# 2️⃣ BLAST d’une protéine
# -----------------------------

def blast_protein(record, FAO_blast_db, protein_to_filter, tmp_query="temp_query.fasta", tmp_out="temp_out.txt"):
    """ blast of the windows of 80 amino acids of the query protein
        with the allergen database """
    
    prot_to_filt = pd.read_csv(protein_to_filter)
    lookup = {}

    for row in prot_to_filt.itertuples():
        lookup.setdefault(row.query, set()).add(row.subject)
    
    # Générer fenêtres
    windows = generate_windows(record)
    n_windows = len(windows)
    
    if n_windows == 0:
        return {
            "protein_id": record.id,
            "n_hits_35_80": 0,
            "n_hits_norm": 0,
            "pident_max_raw": 0,
            "pident_max_35_filt": 0
        }
    
    # Écriture fichier query
    SeqIO.write(windows, tmp_query, "fasta")
    
    # Lancer BLAST
    subprocess.run([
        "/opt/homebrew/bin/blastp",
        "-query", tmp_query,
        "-db", FAO_blast_db,
        "-outfmt", "6 qseqid sseqid pident length",
        "-out", tmp_out
    ])
    
    # Parsing résultats
    n_hits_35_80 = 0
    pident_max = 0
    
    with open(tmp_out) as f:
        for line in f :
            qseqid, sseqid, pident, length = line.strip().split()
            pident = float(pident)
            length = int(length)
            
            # Exclure self-hit
            if record.id.startswith('P'):
                if sseqid == record.id :
                    continue
                
            # exclure les protéines très proches
            # celles qui sont dans le doc prot_to_filter
            if record.id in lookup and sseqid in lookup[record.id] :  
                # print(f'similar prot found : {record.id} -> {sseqid}')
                continue
                
            
            if length >= 80 : 
                
                # Max identité globale
                if pident > pident_max:
                    pident_max = pident
                
                # Règle FAO
                if pident >= 35 :
                    n_hits_35_80 += 1
    
    # Normalisation par nombre de fenêtres
    n_hits_norm = n_hits_35_80 / n_windows
    
    # Feature seuil 35%
    pident_max_35_filt = pident_max if pident_max >= 35 else 0
    
    return {
        "protein_id": record.id,
        "n_hits_35_80": n_hits_35_80,
        "n_hits_norm": n_hits_norm,
        "pident_max_raw": pident_max,
        "pident_max_35_filt": pident_max_35_filt
    }


# -----------------------------
# 3️⃣ Pipeline complet
# -----------------------------

def compute_all_features(query_fasta, FAO_blast_db,protein_to_filter):
    """ Calculation of the features fore each protein in the query fasta """
    
    all_records = list(SeqIO.parse(query_fasta, "fasta"))
    results = []
    
    for record in all_records:
        print(f"Processing {record.id}")
        features = blast_protein(record, FAO_blast_db, protein_to_filter)
        results.append(features)
    
    df = pd.DataFrame(results)
    return df






if __name__ = "__main__" :
    # Fasta with the query proteins
    #query_fasta = "Undersamp_training_set.txt"
    query_fasta = "Test_set.txt"
    
     
    FAO_blast_db = "allergen_db/allergen_db" # output of the makeblast command
    input_makeblast = 'allergen_db.txt' # allergen database
    
    
    # Proteins with >95% of sequence identity with the allergen database (allergen_db.txt)
    # that we have to exclude for the calculation
    protein_to_filter = 'protein_to_filter.csv'  


    # Pre-processing the database to make the blast
    # make_blast_db(input_makeblast, FAO_blast_db)
    
    # Computation of features
    df_features = compute_all_features(query_fasta, FAO_blast_db, protein_to_filter)
    df_features.to_csv("FAO_80_35_test_set.csv") # output file that contains the features
    print(df_features.head())

