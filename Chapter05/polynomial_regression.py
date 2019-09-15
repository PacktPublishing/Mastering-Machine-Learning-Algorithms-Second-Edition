import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

from sklearn.linear_model import Ridge
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
    # Create the dataset
    x = np.expand_dims(np.arange(-50, 60, 0.1), axis=1)
    y = 0.1 * np.power(x + np.random.normal(0.0, 2.5, size=x.shape), 3) + \
        3.0 * np.power(x - 2 + np.random.normal(0.0, 1.5, size=x.shape), 2) - \
        5.0 * (x + np.random.normal(0.0, 0.5, size=x.shape))

    y = (y - np.min(y)) / (np.abs(np.min(y)) + np.max(y))

    # Train a Ridge model
    lr = Ridge(alpha=0.1, normalize=True, random_state=1000)
    lr.fit(x, y)

    # Evaluate the Ridge model
    print("R2 = {:.2f}".format(
        r2_score(y, lr.predict(x))))
    print("MAE = {:.2f}".format(
        mean_absolute_error(y, lr.predict(x))))

    # Compute the U scores
    print("U = {:.2f}, UD = {:.2f}".
          format(*u_scores(y, lr.predict(x))))

    # Generate the polynomial features
    pf5 = PolynomialFeatures(degree=5)
    xp5 = pf5.fit_transform(x)

    pf3 = PolynomialFeatures(degree=3)
    xp3 = pf3.fit_transform(x)

    pf2 = PolynomialFeatures(degree=2)
    xp2 = pf2.fit_transform(x)

    # Train the models
    lrp5 = Ridge(alpha=0.1, normalize=True, random_state=1000)
    lrp5.fit(xp5, y)
    yp5 = lrp5.predict(xp5)

    lrp3 = Ridge(alpha=0.1, normalize=True, random_state=1000)
    lrp3.fit(xp3, y)
    yp3 = lrp3.predict(xp3)

    lrp2 = Ridge(alpha=0.1, normalize=True, random_state=1000)
    lrp2.fit(xp2, y)
    yp2 = lrp2.predict(xp2)

    # Compute the U scores
    print("2. U = {:.2f}, UD = {:.2f}".
          format(*u_scores(y, yp2)))
    print("3. U = {:.2f}, UD = {:.2f}".
          format(*u_scores(y, yp3)))
    print("5. U = {:.2f}, UD = {:.2f}".
          format(*u_scores(y, yp5)))

    # Show the results
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(x, y, '-', linewidth=0.5, label="Data")
    ax.plot(x, yp2, c='g', linewidth=2.5, label="Polynomial regression (d=2)")
    ax.plot(x, yp3, c='y', linewidth=2.5, label="Polynomial regression (d=3)")
    ax.plot(x, yp5, c='r', linewidth=2.5, label="Polynomial regression (d=5)")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    l = mlines.Line2D([-50, 60], [lr.intercept_ - 50 * lr.coef_, lr.intercept_ + 60 * lr.coef_], c='r', linewidth=2.5,
                      linestyle="dashed", label="Linear regression")
    ax.add_line(l)
    ax.legend(fontsize=16)

    plt.show()