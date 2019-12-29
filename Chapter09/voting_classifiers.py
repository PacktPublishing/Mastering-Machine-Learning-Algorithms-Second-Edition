import numpy as np
import joblib

from sklearn.datasets import load_wine
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    wine = load_wine()
    X, Y = wine["data"], wine["target"]
    ss = StandardScaler()
    X = ss.fit_transform(X)

    # Test Decision Tree
    svm = SVC(kernel='rbf',
              gamma=0.01,
              random_state=1000)
    print('SVM score: {:.3f}'.format(
        np.mean(cross_val_score(svm, X, Y,
                                n_jobs=joblib.cpu_count(),
                                cv=10))))

    # Test Logistic Regression
    lr = LogisticRegression(C=2.0,
                            max_iter=5000,
                            solver='lbfgs',
                            multi_class='auto',
                            random_state=1000)
    print('Logistic Regression score: {:.3f}'.format(
        np.mean(cross_val_score(lr, X, Y,
                                n_jobs=joblib.cpu_count(),
                                cv=10))))

    # Create a soft voting classifier
    vc = VotingClassifier(estimators=[
        ('LR', LogisticRegression(C=2.0,
                                  max_iter=5000,
                                  solver='lbfgs',
                                  multi_class='auto',
                                  random_state=1000)),
        ('SVM', SVC(kernel='rbf',
                    gamma=0.01,
                    probability=True,
                    random_state=1000))],
        voting='soft',
        weights=(0.5, 0.5))

    print('Voting classifier score: {:.3f}'.format(
        np.mean(cross_val_score(vc, X, Y,
                                n_jobs=joblib.cpu_count(),
                                cv=10))))