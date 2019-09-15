import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error


# Set random seed for reproducibility
np.random.seed(1000)


def u_scores(y_true, y_pred):
    a = np.sum(np.power(y_true - y_pred, 2))
    b = np.sum(np.power(y_true, 2))
    u = np.sqrt(a / b)

    d_true = y_true[:y_true.shape[0] - 1] - y_true[1:]
    d_pred = y_pred[:y_pred.shape[0] - 1] - y_true[1:]
    c = np.sum(np.power(d_true - d_pred, 2))
    d = np.sum(np.power(d_true, 2))
    ud = np.sqrt(c / d)

    return u, ud


if __name__ == "__main__":
    # Load the dataset
    data = load_diabetes()

    X = data['data']
    Y = data['target']

    # Compute the condition number
    XTX = np.linalg.inv(X.T @ X)
    print("k = {:.2f}".format(np.linalg.cond(XTX)))

    # Compute the correlation matrix
    cm = np.corrcoef(X.T)

    sns.set()
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(cm, annot=True, fmt=".1f",
                xticklabels=data['feature_names'], yticklabels=data['feature_names'], ax=ax)

    plt.show()

    # Perform a CV to find the optimal alpha
    rcv = RidgeCV(alphas=np.arange(0.1, 1.0, 0.01),
                  normalize=True)
    rcv.fit(X, Y)

    print("Alpha: {:.2f}".format(rcv.alpha_))

    # Compute the new condition numbers
    print("k(0.1): {:.2f}".format(
        np.linalg.cond(X.T @ X +
                       0.1 * np.eye(X.shape[1]))))
    print("k(0.25): {:.2f}".format(
        np.linalg.cond(X.T @ X +
                       0.25 * np.eye(X.shape[1]))))
    print("k(0.5): {:.2f}".format(
        np.linalg.cond(X.T @ X +
                       0.5 * np.eye(X.shape[1]))))

    # Perform a Ridge regression
    lrr = Ridge(alpha=0.25, normalize=True,
                random_state=1000)
    lrr.fit(X, Y)

    print("R2 = {:.2f}".format(
        r2_score(Y, lrr.predict(X))))
    print("MAE = {:.2f}".format(
        mean_absolute_error(Y, lrr.predict(X))))

    print("U = {:.2f}, UD = {:.2f}".
          format(*u_scores(Y, lrr.predict(X))))

    # Create a polynomial expansion
    pf = PolynomialFeatures(degree=3,
                            interaction_only=True)
    Xp = pf.fit_transform(X)

    lrr = Ridge(alpha=0.25, normalize=True,
                random_state=1000)
    lrr.fit(Xp, Y)

    print("R2 = {:.2f}".
          format(r2_score(Y, lrr.predict(Xp))))
    print("MAE = {:.2f}".
          format(mean_absolute_error(Y, lrr.predict(Xp))))
    print("U = {:.2f}, UD = {:.2f}".
          format(*u_scores(Y, lrr.predict(Xp))))

