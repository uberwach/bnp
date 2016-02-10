from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np


from extraction import prepare_data

if __name__ == "__main__":
    X, _, X_holdout, _ = prepare_data("./data/", drop_categorical=False)

    A = np.vstack((X, X_holdout))

    print "Applying SVD"
    svd = TruncatedSVD(50)
    B = svd.fit_transform(A)
    print B.shape
    print "Applying TSNE"
    tsne = TSNE(2)
    tsne.fit(B)

    for col in M.shape[1]:
        M[:, col].tofile("./features/tsne_{}.npy".format(col))



