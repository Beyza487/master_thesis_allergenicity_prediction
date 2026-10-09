import numpy as np
import os
import logging
import numpy as np

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
def concat_npy(file_path : str, mode : str, output_path : str, label_list, all_results=None):
    """Returns:
        tuple: (all_results, label_list)
            - all_results (list): Liste mise à jour des résultats après transformation.
            - label_list (list): Liste mise à jour des labels.
    """
    
    if all_results is None:
        all_results = []  # Initialiser la liste si elle est None

    # Charger le tenseur
    tensor = np.load(file_path).astype(np.float64)  # Shape [l, 256]
    
    # Dictionnaire des modes de concaténation
    mode_funcs = { # Capturer l'information:
        'mean': np.mean, # globale
        'max': np.max, # locale
        'min': np.min, # locale minimale
        'std': np.std, # Variabilité
        'median': np.median # Robuste aux outliers
    }

    # Application du mode choisi :
    if mode in mode_funcs:
        result = mode_funcs[mode](tensor, axis=0)
    elif mode == 'norm': # Normalisation
        result = np.linalg.norm(tensor, axis=0, ord=2)
    else:
        raise ValueError("Mode must be one of: 'mean', 'max', 'std', 'min', 'median', 'norm")

    # Sauvegarder le résultat individuel
    np.save(output_path, result)
    
    # Ajouter le résultat à la liste pour la concaténation
    all_results.append(result)
    # all_results = np.load(output_path)
    label_list.append(extract_label(os.path.basename(file_path)))

    return all_results, label_list


def concat_dir_mode(mode : str, output_dir_name :str, input_dir : str, label: bool):
    """
    Parcourt un dossier contenant des fichiers NPY, applique une transformation à chaque fichier,
    et stocke les résultats dans un dossier de sortie.
    """
    
    # Création du répertoire de sortie avec le mode appliqué
    output_dir = output_dir_name + 'NPY_training_' + mode
    os.makedirs(output_dir, exist_ok=True) 

    all_results = [] # Liste des résultats après concaténation
    label_list = [] # Liste des Labels

    # Parcours des fichiers dans le répertoire d'entrée
    for file in sorted(os.listdir(input_dir)):
        if not file.endswith('.npy'):
            continue
        
        file_name = os.path.splitext(file)[0]  # Nom du fichier sans extension .npy
        logging.info(f'Concatenation selon le mode: {mode} pour le fichier {file_name}')
        file_path = os.path.join(input_dir, file)  # Chemin complet du fichier .npy

        # Créer le chemin du fichier de sortie individuel
        output_file_name = file_name + '_' + mode + '.npy'
        output_path = os.path.join(output_dir, output_file_name)

        # Ajouter le résultat du fichier à la liste
        all_results, label_list = concat_npy(file_path, mode, output_path, label_list, all_results)

    if all_results:
        # Une fois tous les résultats obtenus, on les empile
        stacked_result = np.stack(all_results, axis=0)

        # Sauvegarde du fichier empilé final
        np.save(os.path.join(output_dir, f'stacked_{mode}.npy'), stacked_result)

        # Sauvegarde des labels associés
        if label:
            np.save(os.path.join(output_dir, f'labels_{mode}.npy'), np.array(label_list))
            logging.info(f"Labels saved in {output_dir}/labels_{mode}.npy")


def main(liste_mode : list, input_dir : str, output_dir_name  : str, label: bool):
    """
    Applique plusieurs modes de transformation sur un ensemble de fichiers NPY.

    Args:
        liste_mode (list): Liste des modes de transformation à appliquer.
        input_dir (str): Répertoire contenant les fichiers NPY.
        output_dir_name (str): Nom du répertoire de sortie.
    """
    for mode in liste_mode:
        logging.info(f'Concatenation selon le mode: {mode}')
        concat_dir_mode(mode, output_dir_name, input_dir, label)


if __name__ == "__main__":
    # Modes de Concaténation ( 'mean', 'max', 'std', 'min', 'median', 'norm')
    liste_mode = ['max', 'mean', 'median', 'min', 'norm', 'std']
    # Répertoire avec tous les fichiers NPY.
    input_dir = "../1_Embeddings/NPY_undersamp_training/"  
    # Répertoire de sortie
    output_dir_name = "Undersamp_training_set/"
    # Labelisation
    label = True

    main(liste_mode, input_dir, output_dir_name, label)
    



    
