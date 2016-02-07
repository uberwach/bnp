import pandas as pd
from time import time
import os

from scipy.stats import randint
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from extraction import prepare_data
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import util

NUM_ESTIMATORS = 70


def find_cv_rf_model(X_train, y_train, grid=True):
    # scoring='log_loss' does not perform that well, estimate totally off
    # Runtime Log (to estimate experiment length)
    # |params| = 8,  cv = 5,  estimators=100:  695 seconds  (cv too low)
    # |params| = 4,  cv = 10, estimators=100:   73 seconds  (poor score)
    # |params| = 32, cv = 10, estimators= 50:
    if grid:
        params = {  # 'max_depth': [1, 2, 3, 4],
            'max_features': [26, 27, 28, 29],
            'max_leaf_nodes': range(10, 50, 5),
            # 'bootstrap': [True, False],
            'criterion': ['entropy']
        }

        search_clf = GridSearchCV(RandomForestClassifier(n_estimators=NUM_ESTIMATORS),
                                  param_grid=params,
                                  scoring='roc_auc',
                                  n_jobs=7,
                                  cv=10)
    else:
        param_dist = {"max_depth": [3, None],
                      "max_features": randint(20, 400),
                      "min_samples_split": randint(1, 100),
                      "min_samples_leaf": randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": ['entropy']}

        n_iter_search = 30
        search_clf = RandomizedSearchCV(RandomForestClassifier(n_estimators=NUM_ESTIMATORS),
                                        param_distributions=param_dist,
                                        scoring="roc_auc",
                                        cv=10,
                                        n_jobs=7,
                                        n_iter=n_iter_search)

    search_clf.fit(X_train, y_train)

    return search_clf


if __name__ == "__main__":
    X, y, X_holdout, ids = prepare_data("./data/", drop_categorical=False)

    # Right now we look at an extra y_train, y_test to assess the quality of our cv-estimates.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print "Run Random Forest with {} data points and {} features.".format(X_train.shape[0], X_train.shape[1])
    t0 = time()
    grid_cv = find_cv_rf_model(X_train, y_train, grid=False) # stochastic search now
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
    rf_clf = RandomForestClassifier(1000, n_jobs=7,
                                    **(grid_cv.best_params_))
    rf_clf.fit(X, y)

    submission_name = "submission_{}.csv".format(time())
    util.note_submission_info("Model: {}".format(rf_clf), submission_name)
    util.build_submission(rf_clf, X_holdout, ids, submission_name)
