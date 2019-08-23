import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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

    # Perform the evaluation for different number of folds
    mean_scores = []
    cvs = [x for x in range(5, 100, 10)]

    for cv in cvs:
        score = cross_val_score(LogisticRegression(solver="lbfgs", multi_class="auto", random_state=1000),
                                X, Y, scoring="accuracy", n_jobs=joblib.cpu_count(), cv=cv)
        mean_scores.append(np.mean(score))

    # Plot the scores
    sns.set()

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(cvs, mean_scores, 'o-')
    ax.set_xlabel('Number of folds / Training set size', fontsize=16)
    ax.set_ylabel('Average accuracy', fontsize=16)
    ax.set_xticks(cvs)
    ax.set_xticklabels(['{} / {}'.format(x, int(500 * (x - 1) / x)) for x in cvs], fontsize=15)
    ax.grid(True)

    plt.show()