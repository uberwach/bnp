import pandas as pd

from time import time
import os
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from extraction import prepare_data
from sklearn.grid_search import GridSearchCV
import util

# pipeline = Pipeline(steps=[OneHotEncoding(categorical_features=get_categorical_columns(X)),
#                           RandomForestClassifier(n_jobs=-1)])

NUM_ESTIMATORS=100

def find_cv_rf_model(X_train, y_train):

    params = {#'max_depth': [1, 2, 3, 4],
              'max_features': range(10, 21),
              #'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']
    }

    # scoring='log_loss' does not perform that well, estimate totally off
    # Runtime Log (to estimate experiment length)
    # |params| = 8,  cv = 5,  estimators=100:  695 seconds  (cv too low)
    # |params| = 4,  cv = 10, estimators=100:   73 seconds  (poor score)
    # |params| = 32, cv = 10, estimators= 50:
    grid_cv = GridSearchCV(RandomForestClassifier(n_estimators=NUM_ESTIMATORS),
                           param_grid=params,
                           scoring='roc_auc',
                           n_jobs=-1,
                           cv=10)

    grid_cv.fit(X_train, y_train)

    return grid_cv

if __name__ == "__main__":
    X, y, _ = prepare_data("./data/train.csv")

    # Right now we look at an extra y_train, y_test to assess the quality of our cv-estimates.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print "Run Random Forest with {} data points and {} features.".format(X_train.shape[0], X_train.shape[1])
    t0 = time()
    grid_cv = find_cv_rf_model(X_train, y_train)
    best_clf = grid_cv.best_estimator_
    y_pred = best_clf.predict_proba(X_test)
    print "Done in %0.3fs" % (time() - t0)

    print "Best params {}: ".format(grid_cv.best_params_)
    print "Best CV score {}: ".format(grid_cv.best_score_)
    print "Training log-loss: {}".format(log_loss(y_train, best_clf.predict_proba(X_train)))
    print "Training accuracy: {}".format(best_clf.score(X_train, y_train))
    print "Test log-loss: {}".format(log_loss(y_test, y_pred))
    print "Test accuracy: {}".format(best_clf.score(X_test, y_test))

    print "Fitting best model on whole data."
    rf_clf = RandomForestClassifier(10*NUM_ESTIMATORS, n_jobs=-1, **(grid_cv.best_params_))
    rf_clf.fit(X, y)

    submission_name = "submission_{}.csv".format(time())
    util.note_submission_info("Model: {}".format(rf_clf), submission_name)
    util.build_submission(rf_clf, submission_name)

