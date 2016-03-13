from sklearn.cross_validation import StratifiedKFold
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from extraction import prepare_data


def get_oob_predictions(clf, X, y, k=5):
    n_samples = X.shape[0]
    X_oob = np.zeros((n_samples, 1))

    kfold = StratifiedKFold(y, k, random_state=42)

    for train, oob in kfold:
        clf.fit(X[train], y[train])
        X_oob[oob, 0] = clf.predict_proba(X[oob])[:, 1]

    return X_oob



def build_base_features(clf, X, X_test, y, k=5):
    X_oob = get_oob_predictions(clf, X, y, k)
    clf.fit(X, y)
    X_holdout = clf.predict_proba(X_test)[:, 1]

    return X_oob, X_holdout

def build_knn_features():
    X, y, X_holdout, ids = prepare_data("./data", drop_categorical=True)

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

if __name__ == "__main__":
    build_knn_features()