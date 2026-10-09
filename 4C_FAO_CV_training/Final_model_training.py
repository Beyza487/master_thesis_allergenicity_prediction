import numpy as np
from sklearn.pipeline import make_pipeline
#from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import pickle
import xgboost as xgb
import os
from sklearn.metrics import confusion_matrix



""" ============== Parameters ================= """
# Stacking method: 'max', 'mean', 'median', 'min', 'norm', 'std'
concat_method = 'median'
# Feature combination
combi = 'combi2'
# feature selection : f_classif, mutual_info_classif
selector = mutual_info_classif
str_selector = 'mutual_info_classif'
# Classificators : LinearSVC(max_iter=5000), GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
#xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42), 
classifier = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
str_classifier = 'GB'
k=247
    
    

X = np.load(f'NPY_training_undersamp/{combi}/stacked_{concat_method}.npy')
y = np.load(f'NPY_training_undersamp/{combi}/labels_{concat_method}.npy')

print(f"Data loaded with method {concat_method}:")
print(f"Data shape: {X.shape}")


clf_k = make_pipeline(SelectKBest(selector, k=k), MinMaxScaler(), classifier)
clf_k.fit(X, y)

            
model_filename = f"Final_models/final_model_{combi}_{str_selector}_{str_classifier}_{concat_method}_k{k}.pkl"
if not os.path.exists('Final_models/'):
    os.makedirs('Final_models/')

with open(model_filename, 'wb') as f:
    pickle.dump(clf_k, f)
print(f"Le modèle a été sauvegardé sous : {model_filename}")
            




