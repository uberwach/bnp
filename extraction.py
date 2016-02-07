import pandas as pd

def drop_categorical_from_df(df):
    # cat_ind = x.dropna().iloc[0].values
    # cat_pos = [i for i in range(len(cat_ind)) if type(cat_ind[i]) == str]
    cat_columns = get_categorical_columns(df)
    df = df.drop(cat_columns, axis=1)
    return df

def get_categorical_columns(df):
    return [u'v3', u'v22', u'v24', u'v30', u'v31', u'v47', u'v52', u'v56', u'v66',
       u'v71', u'v74', u'v75', u'v79', u'v91', u'v107', u'v110', u'v112',
       u'v113', u'v125']

def get_one_hot_columns(df):
    #return [4, 25, 31, 32, 39, 48, 63, 57, 72, 73, 75, 76, 80, 92, 108, 111, 113, 114, 125, 130]
    return ['v3', 'v38', 'v62', 'v36', 'v129']
# 39, 63, 73, 130 int <- could be categorical, too.
# 23, 'v22' <- one hot unviable, this needs to be dropped

# transform categorical variables to numerical ones
def transform_cats(df):
    cat_names = get_one_hot_columns(df)

    # drop the categorical ones that we do not transform
    drop_cols = [x for x in get_categorical_columns(df) if x not in cat_names]
    df = df.drop(drop_cols, axis=1)

    # One hot encode the others
    for cat_column in cat_names:
        print df.columns
        print "One-hot encode {}".format(cat_column)
        df = pd.get_dummies(df, columns=cat_column, dummy_na=True)

    return df

def extract_features(df):
    X = df.values
    return X

def prepare_data(path="./data/train.csv", drop_categorical=True):
    df = pd.read_csv(path)
    ids = df['ID'].values
    del df['ID']

    if drop_categorical:
        df = drop_categorical_from_df(df)
    else:
        df = transform_cats(df)

    df.fillna(df.mean(), inplace=True)

    if 'target' in df.columns:
        y = df['target'].values
        del df['target']
    else:
         y = None

    X = extract_features(df)

    return X, y, ids

