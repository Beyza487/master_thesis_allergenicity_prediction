from Bio import SeqIO


"""
Script to create the allergen database (allergen_db.txt) that is
composed of all the allergens in the complete database (complete_db.txt)

"""



records = []

for record in SeqIO.parse('complete_db.txt', 'fasta'):
    if record.id.startswith('P') :
        records.append(record)
    elif record.id.startswith('N') :
        continue
    else :
        print(f'ERROR — protein name : {record.id}')
        

with open('allergen_db.txt', 'w') as file :
    SeqIO.write(records, file, 'fasta')
