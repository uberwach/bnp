from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from extraction import prepare_data
import numpy as np

SEED = 42
if __name__ == "__main__":
    X, y, X_holdout, ids = prepare_data("./data", drop_categorical=True)

    n_rows = X.shape[0]

    scaler = StandardScaler()
    Z = np.vstack((X, X_holdout))
    Z = scaler.fit_transform(np.vstack((X, X_holdout )))

    X = Z[:n_rows]

    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        print "Fitting KNN k = {}".format(k)
        knn = KNeighborsClassifier(k, n_jobs=-1)
        knn.fit(X, y)
        print "Predicting"
        x_knn = knn.predict_proba(Z)[:, 1]

        x_knn.tofile("./features/knn_{}.npy".format(k))