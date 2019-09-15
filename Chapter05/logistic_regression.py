import numpy as np
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Load the dataset
    data = load_breast_cancer()
    X = data["data"]
    Y = data["target"]

    # Normalize the dataset
    rs = RobustScaler(quantile_range=(15.0, 85.0))
    X = rs.fit_transform(X)

    # Perform a cross validation
    cvs = cross_val_score(
        LogisticRegression(C=0.1, penalty="l1", solver="saga",
                           max_iter=5000, random_state=1000),
        X, Y, cv=10, n_jobs=joblib.cpu_count())

    print(cvs)

    # Train a Lasso Logistic Regression
    lr = LogisticRegression(C=0.1, penalty="l1",
                            solver="saga",
                            max_iter=5000,
                            random_state=1000)
    lr.fit(X, Y)

    for i, p in enumerate(np.squeeze(lr.coef_)):
        print("{} = {:.2f}".
              format(data['feature_names'][i], p))

    # Create a description of the model
    model = "logit(risk) = {:.2f}".format(-lr.intercept_[0])

    for i, p in enumerate(np.squeeze(lr.coef_)):
        if p != 0:
            model += " + ({:.2f}*{}) ".\
                format(-p, data['feature_names'][i])

    print("Model:\n")
    print(model)



