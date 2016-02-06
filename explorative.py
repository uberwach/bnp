import pandas as pd

df = pd.read_csv("./data/train.csv")

def extract_feature_label(df):
    del df['ID']

    y = df['target'].values
    X = df[df.columns[1:]].values

    return X, y

def get_categorical_columns(X):
    return [i for i, val in enumerate(X[0]) if type(val) == str]

X, y = extract_feature_label(df)

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoding
from sklearn.pipeline import Pipeline
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline(steps=[OneHotEncoding(categorical_features=get_categorical_columns(X)),
                RandomForestClassifier(n_jobs=-1)])

pipeline.fit(X_train, y_train)
print "Training Accuracy: {}".format(pipeline.score(X_train, y_train))
print "Test Accuracy: {}".format(pipeline.score(X_test, y_test))
