import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from extraction import prepare_data

# figure out which n_estimators we need to hit sweet spot for runtime <-> quality tradeoff
# you want to see how the curve k -> f(k) and pick a k for which f'(k) is low.
def score_rf_model_k_estimators(k, X_train, y_train, X_test, y_test, hyper_params):
    rf_clf = RandomForestClassifier(n_estimators=k, n_jobs=-1, **hyper_params)
    rf_clf.fit(X_train, y_train)
    return rf_clf.score(X_test, y_test)

if __name__ == "__main__":
    X, y, _ = prepare_data("../data/train.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    hyper_params = {}

    ks = range(1, 200)
    scores = [score_rf_model_k_estimators(k, X_train, y_train, X_test, y_test, hyper_params) for k in ks]


    plt.plot(ks, scores)
    plt.title("Random Forest Performance by number of estimators")
    plt.xlabel("num_estimators")
    plt.ylabel("score")
    plt.savefig("../rf_performance.png")
    plt.show()
