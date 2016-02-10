from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from extraction import prepare_data, get_int_feature_columns
import numpy as np

SEED = 42
if __name__ == "__main__":
    X, y, X_holdout, ids = prepare_data("./data", drop_categorical=True)

    # prepare the categorical columns
    #cat_idx = get_int_feature_columns()
    #encoder = OneHotEncoder(categorical_features=cat_idx, sparse=True)

    #X = encoder.fit_transform(X)

    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        print "Fitting KNN k = {}".format(k)
        knn = KNeighborsClassifier(k, n_jobs=-1)
        knn.fit(X, y)
        print "Predicting"
        x_knn = np.vstack((knn.predict_proba(X)[1], knn.predict_proba(X_holdout)[1]))

        x_knn.tofile("./features/knn_{}.npy".format(k))