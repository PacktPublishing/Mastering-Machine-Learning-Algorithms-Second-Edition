import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=500, n_classes=5, n_features=50, n_informative=10,
                               n_redundant=5, n_clusters_per_class=3, random_state=1000)

    # Scale the dataset
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # Perform a CV with 10 folds and a Logistic Regression
    lr = LogisticRegression(solver="lbfgs", multi_class="auto", random_state=1000)

    splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=1000)
    train_sizes = np.linspace(0.1, 1.0, 20)

    # Compute the learning curves
    lr_train_sizes, lr_train_scores, lr_test_scores = learning_curve(lr, X, Y, cv=splits, train_sizes=train_sizes,
                                                                     n_jobs=joblib.cpu_count(), scoring="accuracy",
                                                                     shuffle=True, random_state=1000)

    # Plot the scores
    sns.set()

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(lr_train_sizes, np.mean(lr_train_scores, axis=1), "o-", label="Training")
    ax.plot(lr_train_sizes, np.mean(lr_test_scores, axis=1), "o-", label="Test")
    ax.set_xlabel('Training set size', fontsize=18)
    ax.set_ylabel('Average accuracy', fontsize=18)
    ax.set_xticks(lr_train_sizes)
    ax.grid(True)
    ax.legend(fontsize=16)

    plt.show()