from scipy.stats import randint
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from extraction import prepare_data, load_extra_features
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import util
from time import time
import numpy as np



def find_cv_rf_model(X_train, y_train, grid=True):
    NUM_ESTIMATORS = 50
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
                                  cv=10,
                                  verbose=1)
    else:
        param_dist = {#"max_depth": [3, None],
                      "max_features": randint(20, 30),
                      "min_samples_split": randint(1, 100),
                      "max_leaf_nodes": randint(5, 20),
                      # "min_samples_leaf": randint(1, 11),
                      # "bootstrap": [True, False],
                      "criterion": ['entropy']}

        n_iter_search = 5
        search_clf = RandomizedSearchCV(RandomForestClassifier(n_estimators=NUM_ESTIMATORS),
                                        param_distributions=param_dist,
                                        scoring="roc_auc",
                                        cv=10,
                                        n_jobs=7,
                                        n_iter=n_iter_search,
                                        verbose=1,
                                        random_state=42)

    search_clf.fit(X_train, y_train)

    return search_clf


def build_rf_submission():
    X, y, X_holdout, ids = prepare_data("./data/", drop_categorical=False)
    # Right now we look at an extra y_train, y_test to assess the quality of our cv-estimates.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print "Run Random Forest with {} data points and {} features.".format(X_train.shape[0], X_train.shape[1])
    t0 = time()
    grid_cv = find_cv_rf_model(X_train, y_train, grid=False)  # stochastic search now
    best_clf = grid_cv.best_estimator_
    y_pred = best_clf.predict_proba(X_test)
    print "Done in %0.3fs" % (time() - t0)
    print "Best params {}: ".format(grid_cv.best_params_)
    print "Best CV score {}: ".format(grid_cv.best_score_)
    print "Training log-loss: {}".format(log_loss(y_train, best_clf.predict_proba(X_train)))
    print "Training accuracy: {}".format(best_clf.score(X_train, y_train))
    print "Test log-loss: {}".format(log_loss(y_test, y_pred))
    print "Test accuracy: {}".format(best_clf.score(X_test, y_test))

    submission_name = "submission_{}.csv".format(time())
    util.note_submission_info("Model: {}".format(best_clf), submission_name)
    util.build_submission(best_clf, X_holdout, ids, submission_name)


def build_rf_features():
    X, y, X_holdout, ids = prepare_data("./data/", drop_categorical=False)

    X1, X2 = load_extra_features()
    X = np.hstack((X, X1))
    X_holdout = np.hstack((X_holdout, X2))

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

    t0 = time()
    for max_depth in range(1, 15):
        print "max depth {}".format(max_depth)
        rf_clf = RandomForestClassifier(n_estimators=200, max_depth=max_depth, criterion="entropy", n_jobs=-1)
        rf_clf.fit(X, y)

        print "Done in %0.3fs" % (time() - t0)
        print log_loss(y_test, rf_clf.predict_proba(X_test))
        #cv_scores = cross_val_score(rf_clf, X, y, scoring='log_loss',  n_jobs=1, cv=5, verbose=1)
        #print "CV log-loss {} (+- {})".format(cv_scores.mean(), cv_scores.std())
    # M = rf_clf.predict_proba(np.vstack((X, X_holdout)))[:, 1]
    # M.tofile("./features/rf_raw_features.npy")

    #submission_name = "submission_{}.csv".format(time())
    #util.note_submission_info("Model: {}".format(rf_clf), submission_name)
    #util.build_submission(rf_clf, X_holdout, ids, submission_name)
if __name__ == "__main__":
    build_rf_features()