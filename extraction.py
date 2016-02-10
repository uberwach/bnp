import pandas as pd
import os
import numpy as np

def drop_categorical_from_df(df):
    return df.drop(get_categorical_columns(), axis=1)

def get_categorical_columns():
    return ['v3','v22', 'v24','v30','v31','v47','v52','v56','v66','v71','v74','v75',
            'v79','v91', 'v107','v110','v112','v113','v125']

# 39, 63, 73, 130 int <- could be categorical, too.
# the following functions returns the indices for X
#
def get_int_feature_columns():
    return [ 32,  53,  61, 109] + [2, 21, 23, 29, 30, 46, 51, 55, 65, 70, 73, 74, 78, 90, 106, 109, 111, 112, 124]

def extract_features(df):
    X = df.values
    return X

def drop_useless_columns_keep_ids(df):
    ids = df['ID'].values
    del df['ID']

    return ids

def prepare_data(path="./data", drop_categorical=True):
    df_train = pd.read_csv(os.path.join(path, "train.csv"))
    df_test = pd.read_csv(os.path.join(path, "test.csv"))

    # filter out useless columns
    drop_useless_columns_keep_ids(df_train)
    test_ids = drop_useless_columns_keep_ids(df_test)

    # extract labels
    y = df_train['target'].values
    del df_train['target']


    if drop_categorical:
        df_train = drop_categorical_from_df(df_train)
        df_test = drop_categorical_from_df(df_test)
        df_train.fillna(df_train.mean(), inplace=True) # TODO: consider replacing to global mean
        df_test.fillna(df_train.mean(), inplace=True)
    else:
        # extract categorical features, set nans to MISSING
        for (train_name, train_series), (test_name, test_series) in zip(df_train.iteritems(),df_test.iteritems()):
            if train_series.dtype == 'O':
                #for objects: factorize
                df_train[train_name], tmp_indexer = pd.factorize(df_train[train_name], na_sentinel=0)
                df_test[test_name].fillna(0, inplace=True)
                df_test[test_name] = tmp_indexer.get_indexer(df_test[test_name])
                #but now we have -1 values (NaN)
            else:
                #for int or float: fill NaN
                tmp_len = len(df_train[train_series.isnull()])
                if tmp_len>0:
                    df_train.loc[train_series.isnull(), train_name] = train_series.mean()
                #and Test
                tmp_len = len(df_test[test_series.isnull()])
                if tmp_len>0:
                    df_test.loc[test_series.isnull(), test_name] = train_series.mean()

    # bring the features into numpy form
    X = extract_features(df_train)
    X_test = extract_features(df_test)

    return X, y, X_test, test_ids

def load_extra_features():

    TRAIN_ROWS = 114321
    ROWS = 228714
    DIR_NAME = "./features/"
    file_names = filter(os.path.isfile, [DIR_NAME + name for name in os.listdir(DIR_NAME)])

    X = np.empty((ROWS, len(file_names)))

    for i, file in enumerate(file_names):
        print "Loading {}".format(file)
        x = np.fromfile(file)
        X[:, i] = x

    return X[:TRAIN_ROWS+1], X[TRAIN_ROWS+1:]