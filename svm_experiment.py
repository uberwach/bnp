from time import time

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import pipeline
from extraction import prepare_data
from util import note_submission_info, build_submission



def find_svm_model(X_train, y_train):

    params = {
        "C" : [0.05, 0.01, 0.005, 0.001],
        "loss" : ["hinge", "squared_hinge"],
        "penalty": ["l2"]
    }

    grid_cv = GridSearchCV(LinearSVC(),
                           param_grid=params,
                           scoring='accuracy',
                           n_jobs=1,
                           cv=5)

    grid_cv.fit(X_train, y_train)

    return grid_cv

def get_feature_transform():
    return pipeline.Pipeline([('Scaling', StandardScaler()),
                  ('Polynomial Features', PolynomialFeatures(2))])

if __name__ == "__main__":
    X, y, X_holdout, ids = prepare_data("./data/train.csv")
    feature_trafo = get_feature_transform()
    X = feature_trafo.fit_transform(X, y)

    # Right now we look at an extra y_train, y_test to access the quality of our cv-estimates.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print "Run Linear SVC with {} data points and {} features.".format(X_train.shape[0], X_train.shape[1])
    t0 = time()
    # grid_cv = find_svm_model(X_train, y_train)
    best_clf = LinearSVC() # grid_cv.best_estimator_
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    print "Done in %0.3fs" % (time() - t0)

    #print "Best params {}: ".format(grid_cv.best_params_)
    #print "Best CV score {}: ".format(grid_cv.best_score_)
    print "Training log-loss: {}".format(log_loss(y_train, best_clf.predict(X_train)))
    print "Training accuracy: {}".format(best_clf.score(X_train, y_train))
    print "Test log-loss: {}".format(log_loss(y_test, y_pred))
    print "Test accuracy: {}".format(best_clf.score(X_test, y_test))

    submission_name = "submission_svm_{}.csv".format(time())
    note_submission_info("Model: {}, Feature: {}".format(best_clf, feature_trafo), submission_name)
    build_submission(best_clf, X_holdout, ids, submission_name, feature_trafo)

