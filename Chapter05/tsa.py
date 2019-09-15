import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tsa.arima_model import ARMA, ARIMA


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == "__main__":
    # Create the time-series
    x = np.arange(0, 50, 0.5)
    y = np.sin(5. * x) + np.random.normal(0.0, 0.5, size=x.shape)

    # Show the time-series
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(y, 'o-', label="Data")
    ax.set_xlabel("t", fontsize=16)
    ax.set_ylabel("Measure", fontsize=16)
    ax.legend(fontsize=16)

    plt.show()

    # Compute and plot the auto-correlation function
    fig = plt.figure(figsize=(22, 14))

    ax0 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(y, lags=30, ax=ax0)
    ax0.set_title("Autocorrelation", fontsize=16)

    plt.show()

    # Train AR, MA, and ARMA models
    y_train = y[0:90]
    y_test = y[90:]

    ar = ARMA(y_train, order=(15, 0), missing="drop").\
        fit(transparams=True, trend="nc")

    arma = ARMA(y_train, order=(6, 4), missing="drop").\
        fit(transparams=True, maxiter=500, trend="nc")

    ma = ARMA(y_train, order=(0, 15), missing="drop").\
        fit(transparams=True, maxiter=500, trend="nc")

    # Compute the predictions
    y_pred_ar = ar.predict(start=90, end=99)
    y_pred_ma = ma.predict(start=90, end=99)
    y_pred_arma = arma.predict(start=90, end=99)

    print("MSE AR: {:.2f}".
          format(0.1*np.sum(np.power(y_test - y_pred_ar, 2))))
    print("MSE MA: {:.2f}".
          format(0.1*np.sum(np.power(y_test - y_pred_ma, 2))))
    print("MSE ARMA: {:.2f}".
          format(0.1*np.sum(np.power(y_test - y_pred_arma, 2))))

    # Show the results for AR and MA
    fig, ax = plt.subplots(2, 1, figsize=(18, 20), sharex=True)

    ax[0].plot(y_test, linewidth=1.0, color="r", label="Data")
    ar.plot_predict(start=90, end=99, plot_insample=False, ax=ax[0])

    ax[1].plot(y_test, linewidth=1.0, color="r", label="Data")
    ma.plot_predict(start=90, end=99, plot_insample=False, ax=ax[1])

    ax[0].set_title("AR(15) prediction", fontsize=16)
    ax[1].set_title("MA(15) prediction", fontsize=16)
    ax[1].set_xlabel("Time", fontsize=16)
    ax[0].set_ylabel("Measure", fontsize=16)
    ax[1].set_ylabel("Measure", fontsize=16)
    ax[0].legend(fontsize=16)
    ax[1].legend(fontsize=16)

    plt.show()

    # Show the result for ARMA
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(y_test, linewidth=1.0, color="r", label="Data")
    arma.plot_predict(start=90, end=99, plot_insample=False, ax=ax)

    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Measure", fontsize=16)
    ax.set_title("ARMA(6, 4) prediction", fontsize=16)
    ax.legend(fontsize=16)

    plt.show()

