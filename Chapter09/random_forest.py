import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    wine = load_wine()
    X, Y = wine["data"], wine["target"]
    ss = StandardScaler()
    Xs = ss.fit_transform(X)

    # Test Logistic regression
    lr = LogisticRegression(max_iter=5000,
                            solver='lbfgs',
                            multi_class='auto',
                            random_state=1000)
    scores_lr = cross_val_score(lr, Xs, Y, cv=10,
                                n_jobs=joblib.cpu_count())
    print("Avg. Logistic Regression CV Score: {:.3f}".
          format(np.mean(scores_lr)))

    # Test Decision Tree
    dt = DecisionTreeClassifier(criterion='entropy',
                                max_depth=5,
                                random_state=1000)
    scores_dt = cross_val_score(dt, Xs, Y, cv=10,
                                n_jobs=joblib.cpu_count())
    print("Avg. Decision Tree CV Score: {:.3f}".
          format(np.mean(scores_dt)))

    # Test Polynomial SVM
    svm = SVC(kernel='rbf', gamma='scale',
              random_state=1000)
    scores_svm = cross_val_score(svm, Xs, Y, cv=10,
                                 n_jobs=joblib.cpu_count())
    print("Avg. SVM CV Score: {:.3f}".
          format(np.mean(scores_svm)))

    # Plot CV scores
    sns.set()
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(scores_lr, 'o-', label='Logistic Regression')
    ax.plot(scores_dt, 'd-', label='Decision Tree')
    ax.plot(scores_svm, 's-', label='Polynomial SVM')
    ax.set_xlabel('Fold', fontsize=22)
    ax.set_ylabel('10-fold Cross-Validation Accuracy', fontsize=22)
    plt.ylim([0.5, 1.05])
    ax.grid(True)
    ax.legend(fontsize=22)
    plt.show()

    # Show the average CV score for different number of trees
    scores_nt = []

    for nt in range(1, 150, 5):
        rf = RandomForestClassifier(n_estimators=nt,
                                    criterion='entropy',
                                    n_jobs=joblib.cpu_count(),
                                    random_state=1000)
        scores_nt.append(np.mean(
            cross_val_score(rf, Xs, Y, cv=10,
                                 n_jobs=joblib.cpu_count())))

    # Plot CV scores
    sns.set()
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(scores_nt, 'o-')
    ax.set_xlabel('Number of trees', fontsize=22)
    ax.set_ylabel('10-fold Average Cross-Validation Accuracy', fontsize=22)
    plt.ylim([0.5, 1.05])
    plt.xticks(np.arange(30), np.arange(1, 150, 5, dtype=np.int))
    ax.grid(True)
    plt.show()

    # Test Random Forest
    rf = RandomForestClassifier(n_estimators=150,
                                criterion='entropy',
                                n_jobs=joblib.cpu_count(),
                                random_state=1000)
    scores = cross_val_score(rf, Xs, Y, cv=10,
                             n_jobs=joblib.cpu_count())
    print("Avg. Random Forest CV score: {:.3f}".
          format(np.mean(scores)))

    # Plot CV scores
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(scores, 'o-')
    ax.set_xlabel('Fold', fontsize=22)
    ax.set_ylabel('10-fold Cross-Validation Accuracy', fontsize=22)
    plt.ylim([0.5, 1.05])
    ax.grid(True)
    plt.show()

    # Show feature importances
    rf.fit(X, Y)

    features = [wine['feature_names'][x] for x in np.argsort(rf.feature_importances_)][::-1]

    fig, ax = plt.subplots(figsize=(22, 10))

    ax.bar([i for i in range(13)], np.sort(rf.feature_importances_)[::-1], align='center')
    ax.set_ylabel('Feature Importance', fontsize=22)
    plt.xticks([i for i in range(len(features))], features, rotation=60, fontsize=22)
    plt.show()

    # Select the most important features
    sfm = SelectFromModel(estimator=rf,
                          prefit=True,
                          threshold=0.02)
    X_sfm = sfm.transform(X)

    print('Feature selection shape: {}'.
          format(X_sfm.shape))

