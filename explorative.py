import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from extraction import prepare_data


X, y = prepare_data("./data/train.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# pipeline = Pipeline(steps=[OneHotEncoding(categorical_features=get_categorical_columns(X)),
#                           RandomForestClassifier(n_jobs=-1)])

rf_clf = RandomForestClassifier(n_jobs=-1)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

print "Training log-loss: {}".format(log_loss(y_train, rf_clf.predict(X_train)))
print "Training accuracy: {}".format(rf_clf.score(X_train, y_train))
print "Test log-loss: {}".format(log_loss(y_test, y_pred))
print "Test accuracy: {}".format(rf_clf.score(X_test, y_test))


