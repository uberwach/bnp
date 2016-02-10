from time import time

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from extraction import prepare_data, get_int_feature_columns
from util import note_submission_info, build_submission
import numpy as np

SEED = 24
def find_svm_model(X_train, y_train):

    params = {
        "C" : [1, 2, 5, 0.05, 0.01],
        "kernel": ["linear", "rbf"]
    }

    grid_cv = GridSearchCV(SVC(probability=True, random_state=SEED),
                           param_grid=params,
                           scoring='log_loss',
                           n_jobs=-1,
                           cv=5,
                           verbose=1)

    grid_cv.fit(X_train, y_train)

    return grid_cv

if __name__ == "__main__":
    X, y, X_holdout, ids = prepare_data("./data", drop_categorical=False)

    # prepare the categorical columns
    cat_idx = get_int_feature_columns()
    encoder = OneHotEncoder(categorical_features=cat_idx, sparse=True, handle_unknown="ignore")
    n_rows = X.shape[0]

    X[:, cat_idx] = X[:, cat_idx] + 1
    X_holdout[:, cat_idx] = X_holdout[:, cat_idx] + 1

    X = encoder.fit_transform(X)
    X_holdout = encoder.transform(X_holdout)

    print "Run SVC with {} data points and {} features.".format(X.shape[0], X.shape[1])
    t0 = time()
    grid_cv = find_svm_model(X, y)
    best_clf = grid_cv.best_estimator_
    print "CV log-loss: {}".format(grid_cv.best_score_)
    best_clf.fit(X, y)
    print "Done in %0.3fs" % (time() - t0)

    M = np.hstack((best_clf.predict_proba(X)[:, 1], best_clf.predict_proba(X_holdout)[:, 1]))
    print "Probability output shape: {}".format(M.shape)  # just to be sure we do not mess up here
    M.tofile("./features/svm_features.npy")


    #submission_name = "submission_svm_{}.csv".format(time())
    #note_submission_info("Model: {}, Feature: {}".format(best_clf, feature_trafo), submission_name)
    #build_submission(best_clf, X_holdout, ids, submission_name, feature_trafo)

