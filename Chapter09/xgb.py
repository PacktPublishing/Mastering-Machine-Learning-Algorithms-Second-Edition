import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
import xgboost as xgb
import shap

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    shap.initjs()

    # Load the dataset
    wine = load_wine()
    X, Y = wine["data"], wine["target"]
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y,
                         test_size=0.15,
                         random_state=1000)

    # Create the D-Matrices
    dall = xgb.DMatrix(X, label=Y,
                       feature_names=wine['feature_names'])
    dtrain = xgb.DMatrix(X_train, label=Y_train,
                         feature_names=wine['feature_names'])
    dtest = xgb.DMatrix(X_test, label=Y_test,
                        feature_names=wine['feature_names'])

    params = {
        'max_depth': 2,
        'eta': 1,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 3,
        'lambda': 1.0,
        'seed': 1000,
        'nthread': joblib.cpu_count(),
    }

    evals = [(dtest, 'eval'), (dtrain, 'train')]
    nb_rounds = 20

    # Perform a CV evaluation
    cv_model = xgb.cv(params, dall, nb_rounds, nfold=10, seed=1000)
    print(cv_model.describe())

