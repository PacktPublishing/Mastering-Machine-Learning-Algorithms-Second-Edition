import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns

from sklearn.cluster.bicluster import SpectralBiclustering


# Set random seed for reproducibility
np.random.seed(1000)


nb_users = 100
nb_products = 100


if __name__ == "__main__":
    # Create the dataset
    items = [i for i in range(nb_products)]

    transactions = []
    ratings = np.zeros(shape=(nb_users, nb_products),
                       dtype=np.int)

    for i in range(nb_users):
        n_items = np.random.randint(2, 60)
        transaction = tuple(
            np.random.choice(items,
                             replace=False,
                            size=n_items))
        transactions.append(
            list(map(lambda x: "P{}".format(x + 1),
                     transaction)))

        for t in transaction:
            rating = np.random.randint(1, 11)
            ratings[i, t] = rating

    # Show the initial matrix
    sns.set()

    fig, ax = plt.subplots(figsize=(12, 12))

    sns.heatmap(ratings, linewidths=0.2, xticklabels=False, yticklabels=False, ax=ax)
    ax.set_xlabel("Products", fontsize=22)
    ax.set_ylabel("Users", fontsize=22)

    plt.show()

    # Perform the biclustering
    sbc = SpectralBiclustering(n_clusters=10, n_best=5,
                               svd_method="arpack",
                               n_jobs=joblib.cpu_count(),
                               random_state=1000)
    sbc.fit(ratings)

    # Show the mix of user/products that have rated/received the rating 8
    print("Users: {}".format(
        np.where(sbc.rows_[8, :] == True)))
    print("Product: {}".format(
        np.where(sbc.columns_[8, :] == True)))

    # Show the final matrix
    rc = np.outer(np.sort(sbc.row_labels_) + 1,
                  np.sort(sbc.column_labels_) + 1)

    fig, ax = plt.subplots(figsize=(12, 12))

    sns.heatmap(rc, linewidths=0.2, xticklabels=False, yticklabels=False, ax=ax)
    ax.set_xlabel("Products", fontsize=22)
    ax.set_ylabel("Users", fontsize=22)

    plt.show()