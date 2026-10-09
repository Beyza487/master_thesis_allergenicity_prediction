import numpy as np
import os
import logging
from Bio import SeqIO

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Fonction pour savoir labeller les données
def extract_label(file_name: str) -> int:
    """
    Extrait le label d'un fichier en fonction de son nom :
    - `Positive` dans le nom -> 1 (Allergène)
    - `Negative` dans le nom -> 0 (Non-Allergène)

    Args:
        file_name (str): Nom du fichier.

    Returns:
        int: 1 si allergène, 0 si non-allergène, -1 si label inconnu.
    """    
    
    if file_name.startswith("Positive") :
        return 1
    elif file_name.startswith("Negative"):
        return 0
    else:
        print(f"Unknown label in filename: {file_name}. Assigning -1.")
        return -1  # Label inconnu

# Fonction pour concaténer le fichier NPY
def concat_npy(file_path : str, label_list, all_results=None):

    """
    Récupère les pooling qui ont déjà été faits et qui se trouvent
    dans file_path.
    Il faut juste les charger ici avec np.load 
    """
    
    result = np.load(file_path)
    
    # Ajouter le résultat à la liste pour la concaténation
    all_results.append(result)
    label_list.append(extract_label(os.path.basename(file_path)))

    return all_results, label_list


def concat_dir_mode(mode : str, output_folder :str, embedd_files : list, label: bool,
                    random_nb):
    """
    Parcoure les NPY contenus dans le répertoire embedd_files.
    Et réalise le stacking pour le random set qu'il est en train de parcourir


    Args:
        mode (str): Mode de transformation ('mean', 'max', etc.).
        output_path (str): Chemin du dossier de sortie.
        embedd_files (list): List contenant les noms des fichiers NPY du random set.
    """
    
    # Création du répertoire de sortie, qui aura un nom ≠ pour les ≠ rand sets
    output_sub_folder = f"set_{random_nb}/"
    output_path = os.path.join(output_folder,output_sub_folder)
    if not os.path.exists(output_path) :
        os.makedirs(output_path,exist_ok=True)
    
    all_results = [] # Liste des résultats après concaténation
    label_list = [] # Liste des Labels

    # Parcourt les fichiers embeddings
    for file in embedd_files:
        if not file.endswith('.npy'):
            continue
        
        file_name = os.path.splitext(file)[0]  # Nom du fichier sans extension .npy
        logging.info(f'Stacking pour le set {random_nb} pour le fichier {file_name}')
        file_path = os.path.join(embeddings_folder, file)  # Chemin complet du fichier .npy

        # Créer le chemin du fichier de sortie individuel
        # output_file_name = file_name + '_' + mode + '.npy'
        # output_path = os.path.join(output_dir, output_file_name)
        

        # Ajouter le résultat du fichier à la liste
        all_results, label_list = concat_npy(file_path, label_list, all_results)

    if all_results:
        # Une fois tous les résultats obtenus, on les empile
        stacked_result = np.stack(all_results, axis=0)

        # Sauvegarde du fichier empilé final
        np.save(os.path.join(output_path, f'stacked_{mode}_{random_nb}.npy'), stacked_result)

        # Sauvegarde des labels associés
        if label:
            np.save(os.path.join(output_path, f'labels_{mode}.npy'), np.array(label_list))
            logging.info(f"Labels saved in {output_path}/labels_{mode}.npy")



def get_ids_random_sets (random_sets_folder,embeddings_folder,random_set_number,
                         output_folder, mode):
    """
    Récupère les identifiant du random set qu'il est en train de parcourir
    Puis, via une autre fonction, trouve les fichiers npy correspondants
    
    embeddings_folder = input_dir
    """
    random_sets_files = os.listdir(random_sets_folder)
    
    # Parcoure les différents random set
    for random_set_file,random_nb in zip(random_sets_files,random_set_number) :
        logging.info(f"Working on set_{random_nb}")
        random_set_path = os.path.join(random_sets_folder,random_set_file)
        
        ids = []
        
        # Récupère les identifiants des protéines dans le random set
        for record in SeqIO.parse(random_set_path,"fasta") :
            ids.append(record.id+f"_{mode}")
        logging.info(f"Ids of set_{random_nb} collected")
        
        # Récupère une liste avec les noms des embeddings des protéines du random set
        embedd_files, embedding_files = get_embedd_of_random(ids,embeddings_folder)
        logging.info(f"Embeddings of set_{random_nb} collected")
        
        concat_dir_mode(mode, output_folder, embedd_files, label,random_nb)
    
    return embedd_files, ids, embedding_files
        



def get_embedd_of_random (ids : list, embeddings_folder):
    """
    Récupère les fichiers .npy individuels qui ont le
    même nom que les identifiants de mon random set
    embeddings folder = input_dir
    """
    
    embed_files_of_rand_set = []
    embedding_files = sorted(os.listdir(embeddings_folder))
    
    for embedd_file in embedding_files :
        if os.path.splitext(embedd_file)[0] in ids :
            embed_files_of_rand_set.append(embedd_file)
            
    return embed_files_of_rand_set, embedding_files




def main(mode : list, random_sets_folder, embeddings_folder : str ,random_set_number : str,
         output_folder  : str, label: bool):
    """
    Applique plusieurs modes de transformation sur un ensemble de fichiers NPY.

    Args:
        mode (str): mode de concaténation. Tous les faire comme ca on les a.
        embeddings_folder (str): Répertoire contenant les fichiers NPY individuels,
            avec la méthode de pooling choisie.
        random_sets_folder : Répertoire contenant les fichiers fasta des random sets
        output_folder (str): Nom du répertoire de sortie.
    """
    
    logging.info(f'Concatenation selon le mode: {mode}')
    embedd_files, ids, embedding_files = get_ids_random_sets(random_sets_folder,embeddings_folder,random_set_number,
                                            output_folder, mode)



if __name__ == "__main__":
    # Modes de Concaténation ('max', 'mean', 'median', 'min', 'std', 'norm')
    mode = 'mean'
    # Répertoire avec les fichiers NPY.
    embeddings_folder = f"NPY_test/combi4/NPY_test_{mode}/"

    
    # Répertoire avec les sets randomisés NPY.
    random_sets_folder = "../0_Database/random_test_sets/"
    # Répertoire de sortie
    output_folder = "NPY_test/combi4/random_sets"
    
    # Labelisation
    label = True
    # Nombre des random sets
    random_set_number = list(range(1,101))

    main(mode, random_sets_folder,embeddings_folder, random_set_number, output_folder, label)
