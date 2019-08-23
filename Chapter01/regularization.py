import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=500, n_classes=2, n_features=10, n_informative=5,
                               n_redundant=3, n_clusters_per_class=2, random_state=1000)

    # Scale the dataset
    ss = StandardScaler()
    X_s = ss.fit_transform(X)

    # Train two logistic regressions with L2 and L1 penalties
    lr_l2 = LogisticRegression(solver='saga', penalty='l2', C=0.25, random_state=1000)
    lr_l2.fit(X_s, Y)

    lr_l1 = LogisticRegression(solver='saga', penalty='l1', C=0.25, random_state=1000)
    lr_l1.fit(X_s, Y)

    # Plot the coefficients
    df = pd.DataFrame(np.array([np.abs(lr_l1.coef_[0]), np.abs(lr_l2.coef_[0])]).T, columns=['L1', 'L2'])
    sns.set()

    fig, ax = plt.subplots(1, 2, figsize=(22, 8))

    sns.barplot(x=df.index + 1, y='L1', data=df, ax=ax[0])
    sns.barplot(x=df.index + 1, y='L2', data=df, ax=ax[1])

    ax[0].set_title('L1 regularization', fontsize=18)
    ax[0].set_xlabel('Parameter', fontsize=18)
    ax[0].set_ylabel(r'$|\theta_i|$', fontsize=18)

    ax[1].set_title('L2 regularization', fontsize=18)
    ax[1].set_xlabel('Parameter', fontsize=16)
    ax[1].set_ylabel('')

    plt.show()