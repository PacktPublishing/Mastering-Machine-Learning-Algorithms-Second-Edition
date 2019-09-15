import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set random seed for reproducibility
np.random.seed(1000)

# Download the dataset from: https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
data_file = "energydata_complete.csv"


if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv(data_file, header=0, index_col="date")

    # Perform the smoothing
    Y = df["T1"].values
    l1 = 0.995
    l2 = 0.999

    skt = np.zeros((Y.shape[0], 2))
    skt[0, 0] = Y[0]
    skt[0, 1] = Y[0]

    for i in range(1, skt.shape[0]):
        skt[i, 0] = ((1 - l1) * Y[i]) + (l1 * skt[i - 1, 0])
        skt[i, 1] = ((1 - l2) * Y[i]) + (l2 * skt[i - 1, 1])

    # Show the results
    sns.set()

    fig, ax = plt.subplots(figsize=(18, 8))

    df["T1"].plot(label="Data", linewidth=1.)
    ax.plot(skt[:, 0], linewidth=3.0, linestyle="dashed", color="r", label=r"Smoothing $\lambda=0.995$")
    ax.plot(skt[:, 1], linewidth=3.0, linestyle="dashed", color="g", label="Smoothing $\lambda=0.999$")

    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Kitchen temperature (°C)", fontsize=16)
    ax.legend(fontsize=16)

    plt.show()