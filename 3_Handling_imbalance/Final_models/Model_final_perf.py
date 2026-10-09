import os
import pandas as pd
import pickle
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score


# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data(input_path,random_nb,mode):
    """ for info : input_path = os.path.join(random_sets_folder,f"set_{random_nb}/")
    """
    stacked_path = os.path.join(input_path, f"stacked_{mode}_{random_nb}.npy")
    X_test = np.load(stacked_path)
    
    labels_path = os.path.join(input_path, f"labels_{mode}.npy")
    y_test = np.load(labels_path)
    
    logging.info(f" - Données chargees : X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
    return X_test, y_test



def prediction_threshold(model, X_test, y_test, threshold, random_nb):
    # Predict with threshold
    print(f"Prediction avec un seuil de {threshold}...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    print(f" - Predictions realisees pour le set_{random_nb}: {np.bincount(y_pred)} (negatifs/positifs)")
    return y_proba, y_pred



def computation_metric(y_test, y_pred):
    label = [1,0]
    cm = confusion_matrix(y_test, y_pred, labels=label)
    TP, FN, FP, TN = cm.ravel() # C'est normal que ce soit inverse parce que sinon ça comprends pas bien qui est positif et qui est negatif...
    # Calcul des métriques : 
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0  # Rappel (Sensibilite)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  # Specificite
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Accuracy
    bacc = (sensitivity + specificity) / 2  # Balanced Accuracy (BACC)
    mcc = (TP * TN - FP * FN) / ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))**0.5 # MCC
    f1_score = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0 # f1-score
    ppv = TP / (TP + FP) if (TP + FP) != 0 else 0  # Precision positive predictive (PPV)
    fpr = FP / (FP + TN) if (FP + TN) != 0 else 0  # Faux Positif Rate (FPR)
    fnr = FN / (FN + TP) if (FN + TP) != 0 else 0  # False Negative Rate (FNR)
    dist = np.sqrt((1-specificity)**2 + (1-sensitivity)**2)
    auc = roc_auc_score(y_test, y_pred)
    print(f" - Sensitivity : {sensitivity:.3f}, Specificity : {specificity:.3f}, Sn/Sp dist : {dist:.3f}")
    
    return cm, TN, FP, FN, TP, sensitivity, specificity, accuracy, bacc, mcc, f1_score, ppv, fpr, fnr, auc, dist



def calculate_mean_and_std(cm_list, TN_list, FP_list, FN_list, TP_list,
                sensitivity_list,specificity_list, accuracy_list, bacc_list,
                mcc_list, f1_score_list, ppv_list, fpr_list, fnr_list,auc_list,dist_list) :

    
    mean_cm = np.mean(cm_list,axis=0)
    mean_TN = np.mean(TN_list)
    mean_FP = np.mean(FP_list)
    mean_FN = np.mean(FN_list)
    mean_TP = np.mean(TP_list)
    mean_sensitivity = np.mean(sensitivity_list)
    mean_specificity = np.mean(specificity_list)
    mean_accuracy = np.mean(accuracy_list)
    mean_bacc = np.mean(bacc_list)
    mean_mcc = np.mean(mcc_list)
    mean_f1_score = np.mean(f1_score_list)
    mean_ppv = np.mean(ppv_list)
    mean_fpr = np.mean(fpr_list)
    mean_fnr = np.mean(fnr_list)
    mean_auc = np.mean(auc_list)
    mean_dist = np.mean(dist_list)
    
    std_cm = np.std(cm_list,axis=0)
    std_TN = np.std(TN_list)
    std_FP = np.std(FP_list)
    std_FN = np.std(FN_list)
    std_TP = np.std(TP_list)
    std_sensitivity = np.std(sensitivity_list)
    std_specificity = np.std(specificity_list)
    std_accuracy = np.std(accuracy_list)
    std_bacc = np.std(bacc_list)
    std_mcc = np.std(mcc_list)
    std_f1_score = np.std(f1_score_list)
    std_ppv = np.std(ppv_list)
    std_fpr = np.std(fpr_list)
    std_fnr = np.std(fnr_list)
    std_auc = np.std(auc_list)
    std_dist = np.std(dist_list)
    
    return (mean_cm,mean_TN,mean_FP,mean_FN,mean_TP,mean_sensitivity,
            mean_specificity,mean_accuracy,mean_bacc,mean_mcc,mean_f1_score,
            mean_ppv,mean_fpr,mean_fnr,mean_auc,mean_dist,
            std_cm,std_TN,std_FP,std_FN,std_TP,std_sensitivity,
            std_specificity,std_accuracy,std_bacc,std_mcc,std_f1_score,std_ppv,
            std_fpr,std_fnr,std_auc,std_dist)


def plot_confusion_matrix(mean_cm, labels_cm, results_folder):
    """
    Affiche la matrice de confusion sous forme de carte thermique.

    Args:
        cm (np.ndarray): Matrice de confusion.
        labels_cm (list): Liste des labels de classification ("Allergen", "Non-Allergen").
    """
    
    plt.figure(figsize=(6,4))
    sns.heatmap(mean_cm, annot=True, fmt=".3f",cmap="YlOrBr", xticklabels=labels_cm, yticklabels=labels_cm)
    plt.xlabel("Predictions")
    plt.ylabel("Vrais Labels")
    plt.title("Matrice de Confusion du nouveau modèle")
    plt.savefig(os.path.join(results_folder,"confusion_matrix.png"))
    #plt.show()


def plot_radar_metrics(mean_sensitivity, mean_specificity, mean_accuracy, mean_bacc,
                       mean_mcc, mean_f1_score, mean_auc, results_folder):
    labels = [
        "Recall (Sensibilité)",
        "Spécificité",
        "Accuracy",
        "MCC",
        "F1-Score",
        "BACC",
        "AUC"
    ]
    values = [mean_sensitivity, mean_specificity, mean_accuracy,
              mean_mcc, mean_f1_score, mean_bacc, mean_auc]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='darkorange', linewidth=2)
    ax.fill(angles, values, color='darkorange', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("Radar des métriques du nouveau modèle", size=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder,"radar_plot.png"))
    #plt.show()


def save_results(result_file, mean_cm,mean_TN,mean_FP,mean_FN,mean_TP,
                 mean_sensitivity,mean_specificity,mean_accuracy,mean_bacc,mean_mcc,
                 mean_f1_score,mean_ppv,mean_fpr,mean_fnr,mean_auc,mean_dist,
                 std_cm,std_TN,std_FP,std_FN,std_TP,std_sensitivity,
                 std_specificity,std_accuracy,std_bacc,std_mcc,std_f1_score,std_ppv,
                 std_fpr,std_fnr,std_auc,std_dist):
   
    
    with open(result_file, "w") as f:
        # Matrice de Confusion
        f.write("Matrice de Confusion:\n")
        f.write("\tPredictions ->\tAllergen\tNon-Allergen\n")
        f.write(f"\tAllergen\t{mean_cm[0,0]}\t\t{mean_cm[0,1]}\n")
        f.write(f"\tNon-Allergen\t{mean_cm[1,0]}\t\t{mean_cm[1,1]}\n")
        f.write("\tVrais Labels\n\n")
        # Rapport de Classification
        f.write("Rapport de Classification:\n")
        f.write(f"Vrai Positif (TP): {mean_TP:.3f} ± {std_TP:.3f}\n")
        f.write(f"Faux Positif (FP): {mean_FP:.3f} ± {std_FP:.3f}\n")
        f.write(f"Faux Negatif (FN): {mean_FN:.3f} ± {std_FN:.3f}\n")
        f.write(f"Vrai Negatif (TN): {mean_TN:.3f} ± {std_TN:.3f}\n\n")
        f.write(f"Sensibilite (Recall): {mean_sensitivity:.3f} ± {std_sensitivity:.3f}\n")
        f.write(f"Specificite: {mean_specificity:.3f} ± {std_specificity:.3f}\n")
        f.write(f"Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {mean_mcc:.3f} ± {std_mcc:.3f}\n")
        f.write(f"Precision positive predictive (PPV): {mean_ppv:.3f} ± {std_ppv:.3f}\n")
        f.write(f"F1-Score: {mean_f1_score:.3f} ± {std_f1_score:.3f}\n")
        f.write(f"Balanced Accuracy (BACC): {mean_bacc:.3f} ± {std_bacc:.3f}\n")
        f.write(f"Taux de faux positifs (FPR): {mean_fpr:.3f} ± {std_fpr:.3f}\n")
        f.write(f"Taux de faux négatifs (FNR): {mean_fnr:.3f} ± {std_fnr:.3f}\n")
        f.write(f"AUC (Air sous la courbe ROC): {mean_auc:.3f} ± {std_auc:.3f}\n")
        f.write(f"Distance : {mean_dist:.3f} ± {std_dist:.3f}\n")
    print(f"Les resultats ont été enregistrés dans {result_file}")




def main(model, random_sets_folder, threshold, random_set_number,
         results_folder, labels_cm, mode):
    
    results = []
    logging.info("Debut de l'évaluation du modèle...")
    
    cm_list = []
    TN_list = []
    FP_list = []
    FN_list = []
    TP_list = []
    sensitivity_list = []
    specificity_list = []
    accuracy_list = []
    bacc_list = []
    mcc_list = []
    f1_score_list = []
    ppv_list = []
    fpr_list = []
    fnr_list = []
    auc_list = []
    dist_list = []
    
    for random_nb in random_set_number : 
        input_path = os.path.join(random_sets_folder,f"set_{random_nb}/")
        
        logging.info(f"Chargement des données du set_{random_nb}...")
        X_test, y_test = load_data(input_path,random_nb,mode)
        y_proba, y_pred = prediction_threshold(model, X_test, y_test, threshold, random_nb)
        
        logging.info(f"Calcul des metriques de performance du set_{random_nb}...")
        cm, TN, FP, FN, TP, sensitivity, specificity, accuracy, bacc, mcc, f1_score, ppv, fpr, fnr, auc, dist = computation_metric(y_test, y_pred)

        cm_list.append(cm)
        TN_list.append(TN)
        FP_list.append(FP)
        FN_list.append(FN)
        TP_list.append(TP)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        accuracy_list.append(accuracy)
        bacc_list.append(bacc)
        mcc_list.append(mcc)
        f1_score_list.append(f1_score)
        ppv_list.append(ppv)
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        auc_list.append(auc)
        dist_list.append(dist)
        
        logging.info(f"Données du set_{random_nb} chargées dans les listes...")

    logging.info(f"--- Fin de l'evaluation pour tous les sets ----")
    
    logging.info(f"Calcul de la moyenne des performances...")
    (mean_cm,mean_TN,mean_FP,mean_FN,mean_TP,mean_sensitivity,mean_specificity,
     mean_accuracy,mean_bacc,mean_mcc,mean_f1_score,mean_ppv,mean_fpr,mean_fnr,mean_auc, mean_dist,
     std_cm,std_TN,std_FP,std_FN,std_TP,std_sensitivity,std_specificity,std_accuracy,
     std_bacc,std_mcc,std_f1_score,std_ppv,std_fpr,std_fnr,std_auc, std_dist) =calculate_mean_and_std(
        cm_list, TN_list, FP_list, FN_list, TP_list, sensitivity_list,
        specificity_list, accuracy_list, bacc_list, mcc_list,
        f1_score_list, ppv_list, fpr_list, fnr_list,auc_list,dist_list)
    
    logging.info(f"---- Moyenne des performances calculées ----")
    
#     logging.info(f"Génération des résultats imagés (CM et radar plot)...")
#     plot_radar_metrics(mean_sensitivity, mean_specificity, mean_accuracy, mean_bacc,
#                        mean_mcc, mean_f1_score, mean_auc, results_folder)
#     plot_confusion_matrix(mean_cm, labels_cm, results_folder)
    
    logging.info(f"Ecriture et sauvegarde des résultats de performance...")
    result_file = os.path.join(results_folder,"results.txt")
    save_results(result_file,mean_cm,mean_TN,mean_FP,mean_FN,mean_TP,mean_sensitivity,
                 mean_specificity,mean_accuracy,mean_bacc,mean_mcc,mean_f1_score,
                 mean_ppv,mean_fpr,mean_fnr,mean_auc,mean_dist,std_cm,std_TN,std_FP,std_FN,std_TP,std_sensitivity,
                 std_specificity,std_accuracy,std_bacc,std_mcc,std_f1_score,std_ppv,
                 std_fpr,std_fnr,std_auc,std_dist)
    
    print("Fin de la sauvegarde des résultats")
    
    all_list = pd.DataFrame([
        {'TN' : mean_TN,
         'FP' : mean_FP,
         'FN' : mean_FN,
         'TP' : mean_TP,
         'Sn': mean_sensitivity,
         'Sp' : mean_specificity,
         'ACC' : mean_accuracy,
         'BACC' : mean_bacc,
         'MCC' : mean_mcc,
         'F1-score' : mean_f1_score,
         'PPV' : mean_ppv,
         'FPR' : mean_fpr,
         'FNR' : mean_fnr,
         'AUC' : mean_auc,
         'Dist' : mean_dist 
            }])
    
    all_list.to_csv(f'{results_folder}results.csv')



if __name__ == "__main__":
    #Chemin vers le modèle à tester
    model_path = 'final_model_mutual_info_classif_XGBoost_mean_k241.pkl'
    
    mode = 'mean' #pooling method
    #Load du modèle avec pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    #Fichiers stacked à charger
    random_sets_folder = "NPY_test/random_sets/"

    #Threshold
    threshold = 0.6
    
    # Labels pour la matrice de confusion
    labels_cm=["Allergen", "Non-Allergen"]
    
    # Liste de nombre de 1 à 100 pour pouvoir nommer les fichiers
    random_set_number = list(range(1,101))

    #Chemin vers le dossier dans lequel les résultats sont sauvegardés.
    results_folder = "Result/"
    if not os.path.exists(results_folder) :
        os.makedirs(results_folder,exist_ok=True)

    main(model, random_sets_folder, threshold, random_set_number, results_folder, labels_cm, mode)
    
