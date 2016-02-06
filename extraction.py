import pandas as pd

df = pd.read_csv("./data/train.csv")

def drop_categorical_from_df(x):
    cat_ind = x.dropna().iloc[0].values
    cat_pos = [i for i in range(len(cat_ind)) if type(cat_ind[i]) == str]
    df = x.drop(x.columns[cat_pos], axis=1)
    return df

# df_drop = drop_categorical(df)

# transform categorical variables to numerical ones

def transform_cats(x):
    cat_ind = x.dropna().iloc[0].values
    cat_pos = [i for i in range(len(cat_ind)) if type(cat_ind[i]) == str]
    pd.get_dummies(x[cat_pos].fillna("Missing"))
    return df

# df_binary = transform_cats(df)

# df = df.join(df_binary)
# df.fillna(0, inplace=True)

def extract_feature_label(df):
    X = df[df.columns[2:]].values
    return X

def prepare_data(path="./data/train.csv", drop_categorical=True):
    df = pd.read_csv(path)

    if drop_categorical:
        df_drop = drop_categorical_from_df(df)
        df_drop.fillna(0, inplace=True)

        y = df_drop['target'].values if 'target' in df.columns else None
        X = extract_feature_label(df_drop)

        return X, y

    return None, None # not implemented yet
