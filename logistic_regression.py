import pandas as pd

from time import time
import os
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from extraction import prepare_data
from sklearn.grid_search import GridSearchCV
import util

# pipeline = Pipeline(steps=[OneHotEncoding(categorical_features=get_categorical_columns(X)),
#                           RandomForestClassifier(n_jobs=-1)])


def find_cv_rf_model(X_train, y_train):

    params = {'C': [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    }

    grid_cv = GridSearchCV(LogisticRegression(),
                           param_grid=params,
                           scoring='log_loss',
                           n_jobs=-1,
                           cv=10)

    grid_cv.fit(X_train, y_train)

    return grid_cv

if __name__ == "__main__":
    X, y, _ = prepare_data("./data/train.csv")

    print "Run LogReg with {} data points and {} features.".format(X.shape[0], X.shape[1])
    t0 = time()
    grid_cv = find_cv_rf_model(X, y)
    best_clf = grid_cv.best_estimator_
    y_pred = best_clf.predict_proba(X)
    print "Done in %0.3fs" % (time() - t0)

    print "Best params {}: ".format(grid_cv.best_params_)
    print "Best CV score {}: ".format(grid_cv.best_score_)
    print "Training log-loss: {}".format(log_loss(y, best_clf.predict_proba(X)))
    print "Training accuracy: {}".format(best_clf.score(X, y))

    submission_name = "submission_lr_{}.csv".format(time())
    util.note_submission_info("Model: {}\n\nCV: {}".format(best_clf, grid_cv), submission_name)
    util.build_submission(best_clf, submission_name)

