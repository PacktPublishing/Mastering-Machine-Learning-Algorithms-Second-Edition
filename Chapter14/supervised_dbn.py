import numpy as np

# To install the DBN package: pip install git+git://github.com/albertbup/deep-belief-network.git
# Further information: https://github.com/albertbup/deep-belief-network
from dbn import SupervisedDBNClassification

from sklearn.datasets import load_wine
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load and normalize the dataset
    wine = load_wine()

    ss = StandardScaler()
    X = ss.fit_transform(wine['data'])
    Y = wine['target']

    # Create train and test sets
    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y,
                         test_size=0.25,
                         random_state=1000)

    # Train the model
    classifier = SupervisedDBNClassification(
        hidden_layers_structure=[16, 8],
        learning_rate_rbm=0.001,
        learning_rate=0.01,
        n_epochs_rbm=20,
        n_iter_backprop=100,
        batch_size=16,
        activation_function='relu',
        dropout_p=0.1)

    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    print(classification_report(Y_test, Y_pred))

