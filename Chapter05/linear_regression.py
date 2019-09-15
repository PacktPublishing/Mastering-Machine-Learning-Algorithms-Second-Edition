import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Create the dataset
    x_ = np.expand_dims(np.arange(0, 10, 0.1), axis=1)
    y_ = 0.8 * x_ + np.random.normal(0.0, 0.75, size=x_.shape)
    x = np.concatenate([x_, np.ones_like(x_)], axis=1)

    # Fit the estimator
    theta = (np.linalg.inv(x.T @ x) @ x.T) @ y_

    print("y = {:.2f} + {:.2f}x".
          format(theta[1, 0], theta[0, 0]))

    # Compute the variance
    sigma2 = (1. / float(x_.shape[0] - 1)) * \
             np.sum(np.power(np.squeeze(y_) - np.squeeze(x_) *
                             theta[0, 0], 2))
    variance = np.squeeze(
        np.linalg.inv(x_.T @ x_) * sigma2)

    print("theta ~ N(0.8, {:.5f})".
          format(variance))

    # Compute R^2
    sst = np.sum(np.power(np.squeeze(y_) -
                          np.mean(y_), 2))
    ssr = np.sum(np.power(np.squeeze(y_) -
                          np.squeeze(x_) * theta[0, 0], 2))
    print("R^2 = {:.3f}".format(1 - ssr / sst))

    # Train an OLS with Statsmodels and print a complete summary
    df = pd.DataFrame(data=np.concatenate((x_, y_), axis=1),
                      columns=("x", "y"))

    slr = smf.ols("y ~ x", data=df)
    r = slr.fit()

    print(r.summary())

    # Show the results
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(x[:, 0], y_, 'o-', label="Data")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    l = mlines.Line2D([0, 10], [theta[1, 0], 10 * theta[0, 0]], c='r', linewidth=2.5, label="Regression line")
    ax.add_line(l)
    ax.legend(fontsize=16)

