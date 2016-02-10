from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import pipeline, cross_validation
import numpy as np
from extraction import prepare_data, get_int_feature_columns
from sklearn.grid_search import GridSearchCV
import util

# C = 0.05 performs the best for most experiments
def find_cv_rf_model(X_train, y_train):
    # params = {'C': [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]}
    params = {#'C': [10, 5, 1, 0.1],
              'C': [0.1],
              'penalty': ['l1']}

    grid_cv = GridSearchCV(LogisticRegression(),
                           param_grid=params,
                           scoring='log_loss',
                           n_jobs=7,
                           cv=2, #5,
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

    print "Run LogReg with {} data points and {} features.".format(X.shape[0], X.shape[1])
    t0 = time()
    #grid_cv = find_cv_rf_model(X, y)
    #best_clf = grid_cv.best_estimator_

    best_clf = LogisticRegression(C=0.25, penalty='l1')
    best_clf.fit(X, y)

    # CV log-loss -0.483927951479 (+- 0.00121618173757) for C = 0.1
    # CV log-loss -0.484415686138 (+- 0.00105883337937) for C = 0.5
    # CV log-loss -0.483759409049 (+- 0.00121414467795) for C = 0.25
    print "Done in %0.3fs" % (time() - t0)
    cv_scores = cross_validation.cross_val_score(best_clf, X, y, scoring='log_loss',  n_jobs=7, cv=7, verbose=1)
    print "CV log-loss {} (+- {})".format(cv_scores.mean(), cv_scores.std())

    M = np.hstack((best_clf.predict_proba(X)[:, 1], best_clf.predict_proba(X_holdout)[:, 1]))
    print "Probability output shape: {}".format(M.shape) # just to be sure we do not mess up here
    M.tofile("./features/logreg_features.npy")

    submission_name = "submission_lr_{}.csv".format(time())
    util.note_submission_info("Model: {}\n".format(best_clf), submission_name)
    util.build_submission(best_clf, X_holdout, ids, submission_name)