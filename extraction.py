import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import os

def drop_categorical_from_df(df):
    return df.drop(get_categorical_columns(df), axis=1)

def get_categorical_columns(df):
    return ['v3','v24','v30','v31','v47','v52','v56','v66','v71','v74','v75',
            'v79','v107','v110','v112','v113','v125']


# 39, 63, 73, 130 int <- could be categorical, too.
# the following functions returns the indices for X
def get_int_feature_columns():
    return [ 32,  53,  61, 109]

# transform categorical variables to numerical ones
def transform_cats(df):
    df_cat = df[get_categorical_columns(df)]
    dict_data = df_cat.T.to_dict().values()

    vectorizer = DictVectorizer(sparse=False)
    features = vectorizer.fit_transform(dict_data)

    df_vector = pd.DataFrame(features)
    df_vector.columns = vectorizer.get_feature_names()
    df_vector.index = df_cat.index

    return df_vector,

def extract_features(df):
    X = df.values
    return X

# 'v22', 'v91' <- one hot unviable, these needs to be dropped
def drop_useless_columns_keep_ids(df):
    ids = df['ID'].values
    del df['ID']
    del df['v22']
    del df['v91']

    return ids

def build_vector_df(df, dict_data, vectorizer):
    df_vector = pd.DataFrame(vectorizer.transform(dict_data))
    df_vector.columns = vectorizer.get_feature_names()
    df_vector.index = df.index
    df_vector.fillna(-1, inplace=True)

    return df_vector

def vectorize_dataframe(df, df_test):
     dict_data = df.T.to_dict().values()
     dict_test_data = df_test.T.to_dict().values()

     vectorizer = DictVectorizer(sparse=False)
     vectorizer.fit(dict_data)
     vectorizer.fit(dict_test_data)



     df_vector = build_vector_df(df, dict_data, vectorizer)
     df_vector_test = build_vector_df(df_test, dict_test_data, vectorizer)

     return df_vector, df_vector_test, vectorizer

def prepare_data(path="./data", drop_categorical=True):
    df = pd.read_csv(os.path.join(path, "train.csv"))
    df_test = pd.read_csv(os.path.join(path, "test.csv"))

    # filter out useless columns
    drop_useless_columns_keep_ids(df)
    test_ids = drop_useless_columns_keep_ids(df_test)

    # extract labels
    y = df['target'].values
    del df['target']

    if drop_categorical:
        df = drop_categorical_from_df(df)
        df_test = drop_categorical_from_df(df_test)
        df.fillna(df.mean(), inplace=True) # TODO: consider replacing to global mean
        df_test.fillna(df.mean(), inplace=True)
    else:
        # extract categorical features, set nans to MISSING
        df_cat = df[get_categorical_columns(df)]
        df_test_cat = df[get_categorical_columns(df_test)]
        df_cat.fillna("MISSING", inplace=True)
        df_test_cat.fillna("MISSING", inplace=True)

        # apply vectorizer on data
        df_cat_features, df_test_features, vectorizer = vectorize_dataframe(df_cat, df_test_cat)

        # drop categorical data from original
        df = drop_categorical_from_df(df)
        df_test = drop_categorical_from_df(df_test)

        # for the float feature columns insert the mean
        df.fillna(df.mean(), inplace=True) # TODO: consider replacing to global mean
        df_test.fillna(df.mean(), inplace=True)

        # join the two data frames together
        df = df.join(df_cat_features)
        df_test = df_test.join(df_test_features)

        df_test.fillna(df.mean(), inplace=True) # TODO: quickfix... some entries are NaN dunno why
        # probably the vectorizer...

    # bring the features into numpy form
    X = extract_features(df)
    X_test = extract_features(df_test)

    return X, y, X_test, test_ids

