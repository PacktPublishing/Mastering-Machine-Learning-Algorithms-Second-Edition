import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Create the dataset
    x = np.arange(0, 60, 0.1)
    y = 0.1 * np.power(x + np.random.normal(0.0, 1.0, size=x.shape), 3) + \
        3.0 * np.power(
        x - 2 + np.random.normal(0.0, 0.5, size=x.shape), 2) - \
        5.0 * (x + np.random.normal(0.0, 0.5, size=x.shape))

    y = (y - np.min(y)) / (np.abs(np.min(y)) + np.max(y))

    # Fit a linear regression
    lr = LinearRegression()
    lr.fit(np.expand_dims(x, axis=1), y)

    # Fit a isotonic regression
    ir = IsotonicRegression()
    ir.fit(x, y)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(x, y, '-', linewidth=0.5, label="Data")
    ax.plot(x, ir.predict(np.squeeze(x)), '-', c='g', linewidth=3.5, label="Isotonic regression")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    l = mlines.Line2D([0, 60], [lr.intercept_, lr.intercept_ + 60 * lr.coef_], c='r', linewidth=2.5, linestyle="dashed",
                      label="Linear regression")
    ax.add_line(l)
    ax.legend(fontsize=16)

    plt.show()

