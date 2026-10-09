from Bio import SeqIO

""" Script used to extract all the allergens from the
    Training set and the Test set (which corresponds to the
    Filtered database "complete_db_40.txt") 
    """


records = []

# Getting the ids of the allergens from the complete_db_40.txt
for record in SeqIO.parse('/Users/beyzaer/00memoire/0_Database/complete_db_40.txt', 'fasta') :
    if record.id.startswith('P'):
        records.append(record)


# Creation of a new fasta containing only the allergens
with open('/Users/beyzaer/00memoire/4A_FAO_features_calculation/allergen_from_sets.txt', 'w') as f :
    SeqIO.write(records, f, 'fasta')