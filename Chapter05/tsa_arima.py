import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Create the time-series
    x = np.arange(0, 50, 0.5)
    y = np.sin(5. * x) + np.random.normal(0.0, 0.5, size=x.shape)
    y += x/10.

    # Show the time-series
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(y, 'o-', label="Data")
    ax.set_xlabel("t", fontsize=16)
    ax.set_ylabel("Measure", fontsize=16)
    ax.legend(fontsize=16)

    plt.show()

    # Train the ARIMA model
    y_train = y[0:90]
    y_test = y[90:]

    arima = ARIMA(y_train, order=(6, 1, 2), missing="drop").\
        fit(transparams=True, maxiter=500, trend="c")

    y_pred_arima = arima.predict(start=90, end=99)

    # Show the results
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(y_test, linewidth=1.0, color="r", label="Data")
    arima.plot_predict(start=90, end=99, plot_insample=False, dynamic=True, ax=ax)

    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Measure", fontsize=16)
    ax.set_title("ARMA(6, 1, 2) prediction", fontsize=16)
    ax.legend(fontsize=16)

    plt.show()
