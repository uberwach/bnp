from extraction import prepare_data, get_int_feature_columns
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

import numpy as np
from time import time


SEED = 42
if __name__ == "__main__":
    X, y, X_holdout, ids = prepare_data("./data", drop_categorical=False)

    cat_idx = get_int_feature_columns()
    X, X_holdout = X[:, cat_idx], X_holdout[:, cat_idx]
    X = X + 1
    X_holdout = X_holdout + 1

    mnb = MultinomialNB(alpha=0.0)
    t0 = time()
    print "Fitting multinomial naive-Bayes."
    mnb.fit(X, y)
    print "Done in %0.3fs" % (time() - t0)
    cv_scores = cross_validation.cross_val_score(mnb, X, y, scoring='log_loss', n_jobs=7, cv=7, verbose=1)
    print "CV log-loss {} (+- {})".format(cv_scores.mean(), cv_scores.std())

    print "Predicting: "
    M = np.hstack((mnb.predict_proba(X)[:, 1], mnb.predict_proba(X_holdout)[:, 1]))
    print "Probability output shape: {}".format(M.shape)  # just to be sure we do not mess up here
    M.tofile("./features/naivebayes_features.npy")
