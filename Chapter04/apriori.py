import numpy as np

# Install the library using: pip install -U efficient-apriori
from efficient_apriori import apriori


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

    _, rules = apriori(transactions,
                       min_support=0.15,
                       min_confidence=0.75,
                       max_length=3,
                       verbosity=1)

    print("No. rules: {}".format(len(rules)))

    for r in rules:
        print(r)