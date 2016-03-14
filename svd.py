from sklearn.decomposition import TruncatedSVD
import numpy as np
from extraction import prepare_data

if __name__ == "__main__":
    X, _, X_holdout, _ = prepare_data("./data/", drop_categorical=False)

    A = np.vstack((X, X_holdout))

    print "Applying SVD"
    svd = TruncatedSVD(20)
    B = svd.fit_transform(A)
    print B.shape

    for col in xrange(B.shape[1]):
        B[:, col].tofile("./features/svd_{}.npy".format(col))
