from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import pipeline
from extraction import prepare_data, get_int_feature_columns
from sklearn.grid_search import GridSearchCV
import util

# C = 0.05 performs the best for most experiments
def find_cv_rf_model(X_train, y_train, pipe):
    # params = {'C': [0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]}
    params = {'C': [0.05]}
    grid_cv = GridSearchCV(LogisticRegression(penalty="l1"),
                           param_grid=params,
                           scoring='log_loss',
                           n_jobs=-1,
                           cv=10)

    grid_cv.fit(X_train, y_train)

    return grid_cv

if __name__ == "__main__":
    X, y, X_holdout, ids = prepare_data("./data", drop_categorical=False)

    # hack to extract the 4 columns that are indeed
    cat_idx = get_int_feature_columns()
    encoder = OneHotEncoder(categorical_features=cat_idx, sparse=False)
    X = encoder.fit_transform(X, y)

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

    clf_pipe = pipeline.Pipeline([("One hot encoding", encoder),
                                  ("Classifier", best_clf)])

    submission_name = "submission_lr_{}.csv".format(time())
    util.note_submission_info("Model: {}\n\nCV: {}".format(clf_pipe, grid_cv), submission_name)
    util.build_submission(clf_pipe, X_holdout, ids, submission_name)