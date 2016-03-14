from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss
from extraction import prepare_data, get_cat_columns, get_multivariate_bernoulli_features


def get_oob_predictions(clf, X, y, k=5):
    n_samples = X.shape[0]
    X_oob = np.zeros((n_samples, 1))

    kfold = StratifiedKFold(y, k, random_state=42)
    scores = []

    for idx, (train, oob) in enumerate(kfold):
        print "Calculating OOB predictions of {}-th fold".format(idx)
        clf.fit(X[train], y[train])
        X_oob[oob, 0] = clf.predict_proba(X[oob])[:, 1]

        train_score = log_loss(y[train], clf.predict_proba(X[train])[:, 1])
        oob_score = log_loss(y[oob], X_oob[oob, 0])

        print "OOB-loss {}\t train-loss {}".format(oob_score, train_score)
        scores.append(oob_score)

    scores = np.asarray(scores)
    print "Log-Loss: {} (+- {})".format(scores.mean(), scores.std())

    return X_oob


def build_base_features(clf, X, X_test, y, k=5):
    X_oob = get_oob_predictions(clf, X, y, k)
    clf.fit(X, y)
    X_holdout = clf.predict_proba(X_test)[:, 1]

    return X_oob, X_holdout.reshape(X_holdout.shape[0], 1)

def build_knn_features():
    X, y, X_holdout, _ = prepare_data("./data", drop_categorical=True)

    n_rows = X.shape[0]

    scaler = StandardScaler()
    Z = np.vstack((X, X_holdout))
    Z = scaler.fit_transform(np.vstack((X, X_holdout )))

    X = Z[:n_rows]
    X_test = Z[n_rows:]

    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        print "Getting OOB from KNN for k={}".format(k)
        clf = KNeighborsClassifier(k, n_jobs=-1)
        X_1, X_2 = build_base_features(clf, X, X_test, y, 10)

        M = np.vstack((X_1, X_2))
        M.tofile('./features/knn_oob_{}.npy'.format(k))

def build_svm_features():
    X, y, X_test, _ = get_sparse_onehot_features()

    print "Getting OOB predictions from linear SVM"
    clf = SVC(kernel="linear", probability=True)
    X_1, X_2 = build_base_features(clf, X, X_test, y, 5)
    np.vstack((X_1, X_2)).tofile('./features/linear_svm_oob.npy')

    print "Getting OOB predictions from rbf SVM"
    clf = SVC(kernel="rbf", probability=True)
    X_1, X_2 = build_base_features(clf, X, X_test, y, 5)
    np.vstack((X_1, X_2)).tofile('./features/rbf_svm_oob.npy')



def build_logreg_features():
    X, y, X_test, _ = get_sparse_onehot_features()

    print "Getting OOB predictions for LogReg"
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=0.25, penalty='l1'))
    X_1, X_2 = build_base_features(clf, X, X_test, y, 10)
    np.vstack((X_1, X_2)).tofile('./features/logreg_oob.npy')

def build_extratrees_features():
    X, y, X_holdout, _ = prepare_data("./data", drop_categorical=False)

    print "Getting OOB predictions from ExtraTreesClassifier"
    clf = ExtraTreesClassifier(n_estimators=500, max_features= 50,criterion= 'entropy',min_samples_split= 5,
                               max_depth= 50, min_samples_leaf= 5, n_jobs=4)
    X_1, X_2 = build_base_features(clf, X, X_holdout, y, 10)
    np.vstack((X_1, X_2)).tofile('./features/extra_trees_oob.npy')


def build_random_forest_features_on_bernoulli():
    X, y, X_holdout, _ = get_multivariate_bernoulli_features()

    print "Getting OOB predictions for RF classifier"

    clf = RandomForestClassifier(n_estimators=200, max_depth=5, criterion="entropy", n_jobs=7)
    X_1, X_2 = build_base_features(clf, X, X_holdout, y, 5)
    np.vstack((X_1, X_2)).tofile('./features/rf_bf_oob.npy')

def get_sparse_onehot_features():
    X, y, X_holdout, ids = prepare_data("./data", drop_categorical=False)
    cat_idx = get_cat_columns()
    encoder = OneHotEncoder(categorical_features=cat_idx, sparse=True, handle_unknown="ignore")

    X[:, cat_idx] = X[:, cat_idx] + 1
    X_holdout[:, cat_idx] = X_holdout[:, cat_idx] + 1
    X = encoder.fit_transform(X)
    X_holdout = encoder.transform(X_holdout)

    return X.tocsr(), y, X_holdout.tocsr(), ids

if __name__ == "__main__":
    # build_knn_features()
    # build_svm_features()
    # build_logreg_features()
    # build_extratrees_features()
    build_random_forest_features_on_bernoulli()