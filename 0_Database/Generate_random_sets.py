from Bio import SeqIO
import random
import os


"""
Pour générer des sets aléatoires de Test set qui auront
le même nombre de positifs que de négatifs

"""


def generate_balanced_sets (input_file, output_folder,nb_set):
    # crée le dossier s'il n'existe pas déjà
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    negatives = []
    positives = []
    
    # Former les listes contenant les positifs et négatifs
    for record in SeqIO.parse(input_file,"fasta") :
        iden = record.id
        
        if iden.startswith("Positive") :
            positives.append(record)
            
        elif iden.startswith("Negative") :
            negatives.append(record)
    
    # vérifier le nombre de chaque classe 
    nb_pos = len(positives)
    nb_neg = len(negatives)
    
    print(f"Positives : {nb_pos}\nNegatives : {nb_neg}")
    
    # Génération des sets aléatoires
    for i in range(1, nb_set+1) :
        neg_sample = random.sample(negatives, nb_pos)
        
        balanced_set = positives + neg_sample
        
        output_path = os.path.join(output_folder,f"set_{i}.txt")
        SeqIO.write(balanced_set, output_path, "fasta")
    
    print(f"{nb_set} sets générés dans le dossier {output_folder}")


if __name__ == "__main__":
    input_file = "Test_set.txt"
    output_folder = "random_test_sets"
    nb_set = 100
    
    generate_balanced_sets(input_file, output_folder, nb_set)




