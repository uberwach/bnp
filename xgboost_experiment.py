from extraction import prepare_data, load_extra_features
import xgboost as xgb
import numpy as np
import pandas as pd
from time import time
from sklearn import cross_validation

if __name__ == "__main__":
    X, y, X_holdout, ids = prepare_data("./data/", drop_categorical=False)
    X_extra, X_holdout_extra = load_extra_features()

    params = {
       "objective"  : "binary:logistic",
       "eval_metric" : "logloss",
       "eta" : 0.005, # 0.01
       "subsample" : 0.8,
       "colsample_bytree" : 0.8,
       "min_child_weight" : 1,
       "max_depth" : 10
    }

    xg_train = xgb.DMatrix(np.hstack((X, X_extra)), label=y)
    xg_test = xgb.DMatrix(np.hstack((X_holdout, X_holdout_extra)))

    #xg_train = xgb.DMatrix(X, label=y)

    xgb_clf = xgb.train(params, xg_train, num_boost_round=2500, verbose_eval=True, maximize=False)

    y_pred = xgb_clf.predict(xg_test)# ,ntree_limit=xgb_clf.best_iteration)
    #cv_scores = xgb.cv(params, xg_train, num_boost_round=100, nfold=5, metrics="logloss", seed=42, early_stopping_rounds=5)
    #print cv_scores
    df = pd.DataFrame({'ID': ids, 'PredictedProb': y_pred})
    df.to_csv("submission_xgb_ensemble_{}.csv".format(time()), index=False)