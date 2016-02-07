from time import time

import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from extraction import prepare_data
from visualization.learning_curve import plot_learning_curve

if __name__ == "__main__":
    X, y, _, _ = prepare_data("../data")

    cat_mask = (X[0] == 0.0) | (X[0] == 1.0)
    encoder = OneHotEncoder(categorical_features=cat_mask, sparse=False)
    X = encoder.fit_transform(X, y)

    plt = plot_learning_curve(estimator=LogisticRegression(C=0.05),
                              title="Learning Curves of LogReg with logloss",
                              X=X, y=y,
                              cv=cross_validation.ShuffleSplit(X.shape[0], n_iter=5, test_size=0.2, random_state=0),
                              n_jobs=7,
                              scoring="log_loss")

    plt.savefig("../images/learning_curve_logreg_{}.png".format(time()))
    plt.show()