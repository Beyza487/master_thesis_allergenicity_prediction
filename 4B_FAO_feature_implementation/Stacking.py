import numpy as np
import os
import logging
import numpy as np

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(liste_mode : list, directory  : str, label: bool):
    """
    Applique plusieurs modes de transformation sur un ensemble de fichiers NPY.

    Args:
        liste_mode (list): Liste des modes de transformation à appliquer.

    """
    for mode in liste_mode:
        logging.info(f'Concatenation selon le mode: {mode}')
        concat_dir_mode(mode, directory, label)

def concat_dir_mode(mode : str, directory : str, label: bool):
    """
    Parcourt un dossier contenant des fichiers NPY
    et stocke les résultats dans un dossier de sortie.
    """
    
    directory_path = directory + 'NPY_training_' + mode

    all_results = [] # Liste des résultats après concaténation
    label_list = [] # Liste des Labels

    # Parcours des fichiers dans le répertoire
    for file in sorted(os.listdir(directory_path)):
        if not file.endswith('.npy'):
            continue
        
        file_name = os.path.splitext(file)[0]  # Nom du fichier sans extension .npy
        logging.info(f'Concatenation selon le mode: {mode} pour le fichier {file_name}')
        file_path = os.path.join(directory_path, file)  # Chemin complet du fichier .npy

        # Ajouter le résultat du fichier à la liste
        all_results, label_list = concat_npy(file_path, mode, directory_path, label_list, all_results)

    if all_results:
        # Une fois tous les résultats obtenus, on les empile
        stacked_result = np.stack(all_results, axis=0)

        # Sauvegarde du fichier empilé final
        np.save(os.path.join(directory, f'stacked_{mode}.npy'), stacked_result)

        # Sauvegarde des labels associés
        if label:
            np.save(os.path.join(directory, f'labels_{mode}.npy'), np.array(label_list))
            logging.info(f"Labels saved in {directory}/labels_{mode}.npy")


# Fonction pour savoir labeller les données
def extract_label(file_name: str) -> int:

    if file_name.startswith("Positive") :
        return 1
    elif file_name.startswith("Negative") :
        return 0
    else:
        print(f"Unknown label in filename: {file_name}. Assigning -1.")
        return -1  # Label inconnu


# Fonction pour concaténer le fichier NPY
def concat_npy(file_path : str, mode : str, directory_path : str, label_list, all_results=None):
    """Returns:
        tuple: (all_results, label_list)
            - all_results (list): Liste mise à jour des résultats après transformation.
            - label_list (list): Liste mise à jour des labels.
    """
    
    if all_results is None:
        all_results = []  # Initialiser la liste si elle est None

    # Charger le tenseur
    tensor = np.load(file_path).astype(np.float64)  # Shape [l, 256]
    
    # Ajouter le résultat à la liste pour la concaténation
    all_results.append(tensor)
    label_list.append(extract_label(os.path.basename(file_path)))

    return all_results, label_list



if __name__ == "__main__":
    # Modes de Concaténation ( 'mean', 'max', 'std', 'min', 'median', 'norm')
    liste_mode = ['max', 'mean', 'median', 'min', 'norm', 'std']
        
    # directory
    directory = "NPY_undersamp_training/combi4/"

    # Labelisation
    label = True

    main(liste_mode, directory, label)


