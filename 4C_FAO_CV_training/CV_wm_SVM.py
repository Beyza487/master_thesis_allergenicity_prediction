import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import os



def load_data(concat_method: str, combi:'str'):
    """
    Charge les données selon la méthode de concaténation.

    Args:
        concat_method (str): Méthode de concaténation à tester dans cette liste : ('mean', 'min', 'max', 'norm', 'median', 'std')
    
    Returns:
        X (np.array): données
        y (np.array): labels
    """
    X = np.load(f'NPY_training_undersamp/{combi}/stacked_{concat_method}.npy')
    y = np.load(f'NPY_training_undersamp/{combi}/labels_{concat_method}.npy')
    print(f"Data loaded with method {concat_method}:")
    print(f"Data shape: {X.shape}")
    
    return X, y


def evaluate_model(X, y, k_values, selector, classifier, concat):
    """
    Évalue les performances du modèle pour différentes valeurs de k avec validation croisée.
    """
    # Initialisation des résultats
    results = []

    # Si le dossier de sauvegarde n'existe pas, on le crée
    if not os.path.exists('results_training'):
        os.makedirs('results_training')

    # Initialisation de la validation croisée
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        print(f"Calcul pour k = {k}")

        fold_results = []
        
        for fold_index, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf_k = make_pipeline(SelectKBest(selector, k=k), MinMaxScaler(), classifier)
            clf_k.fit(X_train, y_train)

            y_pred = clf_k.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Calcul des métriques
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            bacc = (sensitivity + specificity) / 2
            f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
            mcc = (tp * tn - fp * fn) / ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))**0.5 if ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) !=0 else 0
            distance = np.sqrt((1-specificity)**2+(1-sensitivity)**2)

            fold_results.append((sensitivity, specificity, bacc, f1_score, mcc, distance))
            # print(fold_results)
            # Affichage des résultats pour chaque pli
            # print(f"  Pli {fold_index + 1}: Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f}, BACC = {bacc:.3f}, F1-Score = {f1_score:.3f}, MCC = {mcc:.3f}")

        # Moyenne et écart-type des résultats pour les 5 plis
        metrics_names = ['sensitivity', 'specificity', 'bacc', 'f1', 'mcc', 'distance']

        mean_results = pd.Series(np.mean(fold_results, axis=0),index=metrics_names)
        std_results = pd.Series(np.std(fold_results, axis=0),index=metrics_names)
        

        # Affichage des résultats moyens
        # print(f"Résultats moyens pour k = {k} : Sensitivity = {mean_results[0]:.3f}, Specificity = {mean_results[1]:.3f}, BACC = {mean_results[2]:.3f}, F1-Score = {mean_results[3]:.3f}, MCC = {mean_results[4]:.3f}")

        # Ajout des résultats à la liste
        results.append({
            'k': k,
            'sensitivity': f'{mean_results["sensitivity"]:.3f} ± {std_results["sensitivity"]:.3f}',
            'specificity': f'{mean_results["specificity"]:.3f} ± {std_results["specificity"]:.3f}',
            'bacc': f'{mean_results["bacc"]:.3f} ± {std_results["bacc"]:.3f}',
            'f1_score': f'{mean_results["f1"]:.3f} ± {std_results["f1"]:.3f}',
            'mcc': f'{mean_results["mcc"]:.3f} ± {std_results["mcc"]:.3f}',
            'distance': f'{mean_results["distance"]:.3f} ± {std_results["distance"]:.3f}'
        })
    
    return results


def save_results_to_csv(results, filename):
    """
    Sauvegarde les résultats dans un fichier CSV.
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main(concat_method, classifier, selector, str_classifier, str_selector, step, k_start,combi):
    """
    Fonction principale pour exécuter le code avec des paramètres personnalisés.
    """
    for concat in concat_method:
        filename = f'results_training/{combi}/{str_selector}_{str_classifier}/results_{str_selector}_{str_classifier}_{concat}.csv'
        if not os.path.exists(f'results_training/{combi}/{str_selector}_{str_classifier}/'):
            os.makedirs(f'results_training/{combi}/{str_selector}_{str_classifier}/')
        
        # Charger les données
        X, y = load_data(concat, combi)
    
        # Définir les valeurs de k
        k_end = X.shape[1] + 1
        k_values = list(range(k_start, k_end, step))
    
        # Évaluer les modèles
        results = evaluate_model(X, y, k_values, selector, classifier, concat)

        # Sauvegarder les résultats dans un fichier CSV
        save_results_to_csv(results, filename)


if __name__ == '__main__':
    # Parameters
    # Stacking method: 'max', 'mean', 'median', 'min', 'norm', 'std'
    concat_method = ['max', 'mean', 'median', 'min', 'norm', 'std']
    # Feature combination
    combi = 'combi4'
    # feature selection : f_classif, mutual_info_classif
    selectors = (f_classif, mutual_info_classif)
    str_selectors = ('f_classif', 'mutual_info_classif')
    # Classificator : LinearSVC(max_iter=5000)
    classifier = LinearSVC(max_iter=5000)
    str_classifier = 'SVM'
    step = 15
    k_start = 7

    
    # Exécution du script avec des paramètres personnalisés
    for (selector, str_selector) in zip(selectors, str_selectors):
        main(concat_method, classifier, selector, str_classifier, str_selector, step, k_start, combi)
