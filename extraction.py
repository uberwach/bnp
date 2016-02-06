import pandas as pd

def drop_categorical_from_df(x):
    cat_ind = x.dropna().iloc[0].values
    cat_pos = [i for i in range(len(cat_ind)) if type(cat_ind[i]) == str]
    df = x.drop(x.columns[cat_pos], axis=1)
    return df

def get_categorical_column_idx(df):
    return [idx for idx, column in enumerate(df.columns) if df[type[df[column]] == str].any()]

# transform categorical variables to numerical ones
def transform_cats(x):
    cat_ind = x.dropna().iloc[0].values
    cat_pos = [i for i in range(len(cat_ind)) if type(cat_ind[i]) == str]
    pd.get_dummies(x[cat_pos].fillna("Missing"))
    return df

def extract_features(df):
    X = df.values
    return X

def prepare_data(path="./data/train.csv", drop_categorical=True):
    df = pd.read_csv(path)
    ids = df['ID'].values
    del df['ID']

    if drop_categorical:
        df_drop = drop_categorical_from_df(df)
        df_drop.fillna(0, inplace=True)


        if 'target' in df_drop.columns:
            y = df_drop['target'].values
            del df_drop['target']
        else:
            y = None

        X = extract_features(df_drop)

        return X, y, ids

    return None, None # not implemented yet
