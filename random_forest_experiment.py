import pandas as pd
import numpy as np

from time import time
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from extraction import prepare_data
from sklearn.grid_search import GridSearchCV

X, y, _ = prepare_data("./data/train.csv")

# Right now we look at an extra y_train, y_test to access the quality of our cv-estimates.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# pipeline = Pipeline(steps=[OneHotEncoding(categorical_features=get_categorical_columns(X)),
#                           RandomForestClassifier(n_jobs=-1)])


def find_cv_rf_model(X_train, y_train):

    params = {'max_depth': [3, None],
              'max_features': [10, 11, 12, 13],
              #'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']
    }

    # scoring='log_loss' does not perform that well, estimate totally off
    # Runtime Log (to estimate experiment length)
    # |params| = 8,  cv = 5,  estimators=100:  695 seconds  (cv too low)
    # |params| = 4,  cv = 10, estimators=100:   73 seconds  (poor score)
    # |params| = 16, cv = 10, estimators= 50:
    grid_cv = GridSearchCV(RandomForestClassifier(n_estimators=50),
                           param_grid=params,
                           scoring='accuracy',
                           n_jobs=-1,
                           cv=10)

    grid_cv.fit(X_train, y_train)

    return grid_cv

t0 = time()
grid_cv = find_cv_rf_model(X_train, y_train)
best_clf = grid_cv.best_estimator_
y_pred = best_clf.predict(X_test)
print "Done in %0.3fs" % (time() - t0)

print "Best params {}: ".format(grid_cv.best_params_)
print "Best CV score {}: ".format(grid_cv.best_score_)
print "Training log-loss: {}".format(log_loss(y_train, best_clf.predict(X_train)))
print "Training accuracy: {}".format(best_clf.score(X_train, y_train))
print "Test log-loss: {}".format(log_loss(y_test, y_pred))
print "Test accuracy: {}".format(best_clf.score(X_test, y_test))

# Prepare submission

X_holdout, _, ids = prepare_data("./data/test.csv")
y_submission = best_clf.predict_proba(X_holdout)[:, 1]

df_result = pd.DataFrame({"ID": ids, "PredictedProb": y_submission})
df_result.to_csv("submission.csv", index=False)

# Additional experiments

# figure out which n_estimators we need to hit sweet spot for runtime <-> quality tradeoff
# you want to see how the curve k -> f(k) and pick a k for which f'(k) is low.
def score_rf_model_k_estimators(k, X_train, y_train, X_test, y_test, hyper_params):

    rf_clf = RandomForestClassifier(n_estimators=k, n_jobs=-1, **hyper_params)
    rf_clf.fit(X_train, y_train)
    return rf_clf.score(X_test, y_test)
