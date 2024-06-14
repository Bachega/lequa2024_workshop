import os
import numpy as np
import pandas as pd
from catboost import Pool, CatBoostRegressor

experiment_tables_path = './experiment_tables/'

def load_experiment_tables():
    exp_tables_dict = {key: None for key in ['CC', 'ACC', 'PACC', 'PCC', 'SMM', 'HDy', 'DyS', 'SORD', 'MS', 'MS2', 'MAX', 'X', 'T50']}
        
    for key in exp_tables_dict.keys():
        if os.path.isfile(experiment_tables_path + 'experiment_table_' + key + '.csv'):
            exp_tables_dict[key] = pd.read_csv(experiment_tables_path + 'experiment_table_' + key + '.csv')
        else:
            exp_tables_dict[key] = pd.DataFrame(columns=['dataset_name', 'alpha', 'sample_size', 'real_p', 'pred_p', 'abs_error', 'run_time'])
    
    return exp_tables_dict

def run():
    experiment_tables_dict = load_experiment_tables()
    df_dict = {key: None for key in list(experiment_tables_dict.keys())}


    for key in experiment_tables_dict:
        df_dict[key] = experiment_tables_dict[key].groupby('dataset_name')['abs_error'].aggregate('mean')

    meta_features_table = pd.read_csv('./metafeatures/meta-features-table.csv')

    algList = []
    tableList = []
    for counter in df_dict.keys():
        algList.append(counter)

        y = df_dict[counter].values

        X = meta_features_table.values
        np.nan_to_num(X, copy=False)
        
        row, column = np.where(X > np.finfo(np.float32).max)
        for i in range(len(row)):
            X[row[i]][column[i]] = np.finfo(np.float32).max

        tableList.append((X, y))

    cat_reg = CatBoostRegressor()

    instance_len = len(tableList[0][0])
    cat_results = {}

    j = 0
    for (X, y) in tableList:
        cat_results_list = []
        algName = algList[j]
        j += 1

        for i in range(0, len(X)):
            X_train = np.delete(X, i, 0)
            y_train = np.delete(y, i, 0)

            X_test = X[i]
            X_test = X_test.reshape(1, -1)
            y_test = y[i]

            cat_reg.fit(X_train, y_train)

            cat_abs_error = cat_reg.predict(X_test)

            cat_results_list.append([y_test, cat_abs_error[0]])

        cat_results[algName] = cat_results_list

    data = []
    cols = []
    for key in cat_results:
        cols.append('abs-error-'+key)
        cols.append('abs-error-'+key+'-predicted')
    cols.append('abs-error-ideal')
    cols.append('quantifier-ideal')
    cols.append('quantifier-ideal-num')
    cols.append('abs-error-recommended')
    cols.append('quantifier-recommended')
    cols.append('quantifier-recommended-num')
    i = 1
    for key in cat_results:
        cols.append('rank-' + str(i))
        i += 1

    i = 0
    for i in range(0, instance_len):
        abs_error_ideal = 2
        quantifier_ideal = 'NULL'
        quantifier_ideal_num = -1
        abs_error_recommended = 2
        quantifier_recommended = 'NULL'
        quantifier_recommended_num = -1
        row = []
        algNum = 0
        rank = {}

        for a in algList:
            row.append(cat_results[a][i][0])
            row.append(cat_results[a][i][1])

            rank[algNum] = cat_results[a][i][1]

            if cat_results[a][i][0] < abs_error_ideal:
                abs_error_ideal = cat_results[a][i][0]
                quantifier_ideal = a
                quantifier_ideal_num = algNum

            if cat_results[a][i][1] < abs_error_recommended:
                abs_error_recommended = cat_results[a][i][1]
                quantifier_recommended = a
                quantifier_recommended_num = algNum

            algNum += 1
        rank = sorted(rank.items(), key=lambda item: item[1])

        row.append(abs_error_ideal)
        row.append(quantifier_ideal)
        row.append(quantifier_ideal_num)
        row.append(abs_error_recommended)
        row.append(quantifier_recommended)
        row.append(quantifier_recommended_num)
        for key in rank:
            row.append(int(key[0]))

        data.append(row)
    cat_table = pd.DataFrame(data, columns = cols)
    cat_table.to_csv("./recommendation/recommendation_table_cat.csv", index = False)

if __name__ == '__main__':
    run()