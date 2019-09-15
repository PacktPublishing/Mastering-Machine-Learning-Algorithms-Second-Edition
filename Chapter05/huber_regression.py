import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_error


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Create the dataset
    x = np.expand_dims(np.arange(0, 10, 0.1), axis=1)
    y = 0.8 * x + np.random.normal(0.0, 0.75, size=x.shape)
    y[65:75] *= 5.0

    # Fit a linear regression
    lr = LinearRegression()
    lr.fit(x, y)

    print("Linear: {:.2f}".
          format(mean_absolute_error(y, lr.predict(x))))

    print("Mean Y[0:50] = {:.2f}".
          format(np.mean(y[0:50] - 0.8*x[0:50])))
    print("Std Y[0:50] = {:.2f}".
          format(np.std(y[0:50] - 0.8*x[0:50])))

    # Fit a Huber regression
    hr = HuberRegressor(epsilon=1.2)
    hr.fit(x, y.ravel())

    print("Huber: {:.2f}".
          format(mean_absolute_error(y, hr.predict(x))))

    # Show the results
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(x, y, 'o-', label="Data")
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    l1 = mlines.Line2D([0, 10], [lr.intercept_, 10 * lr.coef_], c='r', linewidth=2.5, linestyle="dashed",
                       label="Regression line")
    l2 = mlines.Line2D([0, 10], [hr.intercept_, 10 * hr.coef_], c='g', linewidth=2.5, label="Huber regression line")
    ax.add_line(l1)
    ax.add_line(l2)
    ax.legend(fontsize=16)

    plt.show()



