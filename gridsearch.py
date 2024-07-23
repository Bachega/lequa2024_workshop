import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# This function generates the train and test partitions
# using holdout with test_size = 0.3 AND random_state = 42
# Also: data is normalized (MinMax Scale)
def generate_train_test_data(source_path):
    files = os.listdir(source_path)
    scaler = MinMaxScaler()

    for f in files:
        data = pd.read_csv(source_path + f)
        data = data.dropna()

        columns = data.columns
        y = data.pop(data.columns[-1])
        X = data
        X = scaler.fit_transform(X)
        X = np.c_[ X, y ]
        data = pd.DataFrame(data=X, columns=columns)

        train, test = train_test_split(data, test_size=0.3, random_state=42)
        train.to_csv('./train_data/' + str(f.split('.csv')[0]) + '-TRAIN.csv', index=False)
        test.to_csv('./test_data/' + str(f.split('.csv')[0]) + '-TEST.csv', index=False)

# Hyperparameters of LogisticRegression (LR) are tuned using the train set
# LR is used as a scorer for the quantifiers
def grid_search(X_train, y_train):
    clf = LogisticRegression(penalty='l2', random_state=42, max_iter=10000)

    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    class_weight = [None, 'balanced']
    
    grid = {'C': C,
            'class_weight': class_weight}
    
    search = GridSearchCV(estimator=clf,
                          param_grid=grid,
                          cv=3,
                          verbose=2,
                          n_jobs=-1)
    
    search.fit(X_train, y_train)
    return search.best_estimator_

def run():
    i = 0
    dataframe = None
    X = None
    y = None
    X_list = []
    y_list = []
    files = os.listdir('./train_data/')

    for f in files:
        if not f.endswith('.csv'):
            continue

        df = pd.read_csv('./train_data/' + f)
        df = df.dropna()

        y = df.pop(df.columns[-1])
        X = df

        y_list.append(y.to_numpy())
        X_list.append(X.to_numpy())

        i += 1
    i = 0

    file = open('log.txt', 'w')
    file.close()
    for i in range(0, len(X_list)):
        if os.path.isfile('./estimator_parameters/' + str(files[i].split('-TRAIN.csv')[0]) + '.joblib'):
            file = open('log.txt', 'a')
            file.write('Skipping ' + str(files[i]) + '\t\t : Already exists\n')
            file.close()
            print('Skipping ' + str(files[i]) + '\t\t : Already exists')
            continue

        try:
            clf = grid_search(X_list[i], y_list[i])
            print(clf.get_params())
            joblib.dump(clf, './estimator_parameters/' + str(files[i].split('-TRAIN.csv')[0]) + '.joblib', compress = 0)
        except Exception as e:
            file = open('log.txt', 'a')
            file.write('Skipping ' + str(i) + ' : ' + str(files[i]) + '...\t\t\t' + str(e) + '\n')
            file.close()

if __name__ == '__main__':
    # # # # # generate_train_test_data('./datasets/')
    # run()
    print('\n')