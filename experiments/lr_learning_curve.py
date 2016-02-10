from time import time

import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from extraction import prepare_data, get_int_feature_columns
from visualization.learning_curve import plot_learning_curve

if __name__ == "__main__":
    X, y, _, _ = prepare_data("../data", drop_categorical=False)

    cat_idx = get_int_feature_columns()
    encoder = OneHotEncoder(categorical_features=cat_idx, sparse=True)
    X = encoder.fit_transform(X, y)


    plt = plot_learning_curve(estimator=LogisticRegression(C=0.1, penalty='l1'),
                              title="Learning Curves of LogReg with logloss",
                              X=X, y=y,
                              cv=5,
                              n_jobs=7,
                              scoring="log_loss")

    plt.savefig("../images/learning_curve_logreg_{}.png".format(time()))
    plt.show()