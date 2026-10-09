import pickle
import pandas as pd
import numpy as np


def selected_FAO_features (model):
    """ Identify which FAO-derived features are selected by the model"""
    
    with open(model, "rb") as f:
        model = pickle.load(f)


    selector = model.named_steps["selectkbest"]
    mask = selector.get_support()
    print(mask)
    print('Selected features \nnhits_norm:',mask[0])
    print('pident_max:',mask[1])
    print('n6_norm:',mask[2])
    print('n7_norm:',mask[3])
    print('n8_norm:',mask[4])
    print('FAO_score:',mask[5],'\n\n')


def weight_of_features(model):
    """ Gives the most important features of the model
    """
    with open(model, "rb") as f:
        model = pickle.load(f)
    
    weight_feature = model.named_steps['xgbclassifier']
    importances = weight_feature.feature_importances_


    df = pd.DataFrame({
        "feature_index": np.arange(len(importances)),
        "importance": importances
    })

    df = df.sort_values("importance", ascending=False)
    print(df.head(10))



if __name__ == "__main__":
    model = 'Final_models/final_model_combi2_mutual_info_classif_XGBoost_median_k157.pkl'
    
    selected_FAO_features(model)
    weight_of_features(model)
    
    
    