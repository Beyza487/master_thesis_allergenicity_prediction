import pandas as pd


""" To get all the best models for each configuration of models:
    pooling_method-combination_of_features-selector-classifier
    The best model selected is the one having the highest MCC
    for the non-restricted threshold and for the restricted threshold
    """



selectors = ['f_classif', 'mutual_info_classif', 'RFE']
classifiers = ['GB','XGBoost','SVM']
pooling_methods = ['max', 'mean', 'median', 'max', 'norm', 'std']
combis=['combi1','combi2','combi3','combi4']




for combi in combis :
    max_MCCs = []
    max_MCCs_filtered = []
    
    for selector in selectors :
        for classifier in classifiers :
            for pooling in pooling_methods :
                source = f'{selector}_{classifier}_{pooling}'
                
                df = pd.read_csv(f'results_training/{combi}/{selector}_{classifier}/results_{selector}_{classifier}_{pooling}.csv')
                
                max_mcc = df.loc[df['mcc'].idxmax()].copy()
                max_mcc['model'] = source
                max_MCCs.append(max_mcc)
                
                # Filtering the database to only have the thresholds between 0.4 and 0.6
                if classifier == 'SVM' :
                    continue
                

                df_2 = df[df['threshold']>=0.39]
                df_2 = df_2[df_2['threshold']<=0.61]
                
                # filtered based on the threshold interval            
                max_mcc_filtered = df_2.loc[df_2['mcc'].idxmax()].copy()
                max_mcc_filtered['model'] = source
                max_MCCs_filtered.append(max_mcc_filtered)
                

    # Save results: 
    result_mcc = pd.DataFrame(max_MCCs)
    result_mcc_filtered = pd.DataFrame(max_MCCs_filtered)
    result_mcc.to_csv(f'results_training/{combi}/mcc_maximal.csv')
    result_mcc_filtered.to_csv(f'results_training/{combi}/mcc_maximal_filtered.csv')


