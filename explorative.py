import pandas as pd

df = pd.read_csv("./data/train.csv")

def drop_categorical(x):
    cat_ind = x.dropna().iloc[0].values
    cat_pos = [i for i in range(len(cat_ind)) if type(cat_ind[i]) == str]
    df = x.drop(x.columns[cat_pos], axis=1)
    return df

df_drop = drop_categorical(df)

# transform categorical variables to numerical ones

def transform_cats(x):
    cat_ind = x.dropna().iloc[0].values
    cat_pos = [i for i in range(len(cat_ind)) if type(cat_ind[i]) == str]
    pd.get_dummies(x[cat_pos].fillna("Missing"))
    return df

df_binary = transform_cats()

df.join(test)

df.fillna(0, inplace=True)

def extract_feature_label(df):
    y = df['target'].values
    X = df[df.columns[1:]].values
    return X, y
"""
def get_categorical_columns(X):
    return [i for i, val in enumerate(X[0]) if type(val) == str]

X, y = extract_feature_label(df)
"""

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import numpy as np

X, y = extract_feature_label(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# pipeline = Pipeline(steps=[OneHotEncoding(categorical_features=get_categorical_columns(X)),
#                           RandomForestClassifier(n_jobs=-1)])

rf_clf = RandomForestClassifier(n_jobs=-1)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

print "Training Accuracy: {}".format(log_loss(y_train, rf_clf.predict(X_train)))
print "Test log-loss: {}".format(log_loss(y_test, y_pred))
