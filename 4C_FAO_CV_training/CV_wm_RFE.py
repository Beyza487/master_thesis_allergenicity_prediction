import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb

def load_data(concat_method: str, combi):
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

def select_features(X, selected_features):
    return X[:, selected_features]

def evaluate_model(X, y, k_values, selector, classifier, concat, thresholds):
    """
    Évalue les performances du modèle pour différentes valeurs de k avec validation croisée.
    """
    # Initialisation des résultats
    results = []


    # Initialisation de la validation croisée
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        print(f"Calcul pour k = {k}")

        fold_results = []
        results_proba = []
        
        for fold_index, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Créer un objet RFE à chaque itération pour ré-entraîner la sélection de caractéristiques
            rfe = RFE(classifier, n_features_to_select=k, step=15)
            print(f"entrainement du RFE pour le fold {fold_index}")
            rfe.fit(X_train, y_train)
            
            # Sélectionner les meilleures caractéristiques selon le classement de RFE
            selected_features = np.where(rfe.support_)[0]
            
            
            feature_selector = FunctionTransformer(select_features, kw_args={'selected_features': selected_features}, validate=False)
            clf_rfe = make_pipeline(feature_selector, MinMaxScaler(), classifier)
            clf_rfe.fit(X_train, y_train)
            
            y_proba = clf_rfe.predict_proba(X_test)[:, 1]


            
            for i in range(len(y_proba)) :
                results_proba.append({
                    'k' : k,
                    'fold' : fold_index+1,
                    'y_proba' : y_proba[i],
                    'y_test' : y_test[i]
                    })
            #print(results_proba)
        res = optimize_threshold(results_proba, thresholds, k)
        results.extend(res)
    
    return results


def optimize_threshold (results_proba, thresholds,k):
    results = []
    df = pd.DataFrame(results_proba)
    for threshold in thresholds :
        threshold_results = []
        for index in range(1,6):
            fold_results = df[df['fold']==index]
            #print(fold_results)
            y_pred = (fold_results['y_proba'] >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(fold_results['y_test'], y_pred, labels=[0,1]).ravel()
            
            # Calcul des métriques
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            bacc = (sensitivity + specificity) / 2
            f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
            mcc = (tp * tn - fp * fn) / ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))**0.5 if ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) !=0 else 0
            distance = np.sqrt((1-specificity)**2+(1-sensitivity)**2)
            
            threshold_results.append((sensitivity, specificity, bacc, f1_score, mcc,distance))

            # Affichage des résultats pour chaque pli
            # print(f"  Pli {fold_index + 1}: Sensitivity = {sensitivity:.3f}, Specificity = {specificity:.3f}, BACC = {bacc:.3f}, F1-Score = {f1_score:.3f}, MCC = {mcc:.3f}")

        # Moyenne et écart-type des résultats pour les 5 plis
        metrics_names = ['sensitivity','specificity','bacc','f1','mcc','distance']

        mean_results = pd.Series(np.mean(threshold_results, axis=0),index=metrics_names)
        std_results = pd.Series(np.std(threshold_results, axis=0),index=metrics_names) 

        # Affichage des résultats moyens
        #print(f"Résultats moyens pour k = {k} et seuil = {threshold} : Sensitivity = {mean_results[0]:.3f}, Specificity = {mean_results[1]:.3f}, BACC = {mean_results[2]:.3f}, F1-Score = {mean_results[3]:.3f}, MCC = {mean_results[4]:.3f}")

        # Ajout des résultats à la liste
        results.append({
            'k': k,
            'threshold' : threshold,
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

def main(concat_method, classifier, selector, str_classifier, str_selector, step, k_start, thresholds, combi):
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
        results = evaluate_model(X, y, k_values, selector, classifier, concat, thresholds)
        
        
        # Sauvegarder les résultats dans un fichier CSV
        save_results_to_csv(results, filename)

if __name__ == '__main__':
    # Parameters
    # Stacking method: 'max', 'mean', 'median', 'min', 'norm', 'std'
    concat_method = ['max', 'mean', 'median', 'min', 'norm', 'std']
    # Feature combination
    combi = 'combi4'
    # feature selection : RFE
    selector = RFE
    str_selector = 'RFE'
    # Classificator : GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    #xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42), 
    classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    str_classifier = 'GB'
    step = 25 
    k_start = 12
    thresholds = list(np.arange(0.2,0.85,0.05))
    

    
    # Exécution du script avec des paramètres personnalisés
    main(concat_method, classifier, selector, str_classifier, str_selector, step, k_start, thresholds, combi)
