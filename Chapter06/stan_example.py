import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Install using pip install pystan
# It requires a C/C++ compiler
import pystan


# Set random seed for reproducibility
np.random.seed(1000)

# Number of observations
nb_samples = 10


if __name__ == "__main__":
    # Create the observations
    departure_delay = np.random.exponential(0.5, size=nb_samples)
    travel_time = np.random.normal(2.0, 0.2, size=nb_samples)
    arrival_delay = np.random.exponential(0.1, size=nb_samples)
    arrival_time = np.random.normal(departure_delay +
                                    travel_time +
                                    arrival_delay,
                                    0.5, size=nb_samples)

    # Define the Stan model
    code = """
    data {
        int<lower=0> num;
        vector[num] departure_delay;
        vector[num] travel_time;
        vector[num] arrival_delay;
        vector[num] arrival_time;
    }
    parameters {
        real beta_a;
        real beta_b;
        real mu_t;
        real sigma_t;
        real sigma_a;
    }
    model {
        departure_delay ~ exponential(beta_a);
        travel_time ~ normal(mu_t, sigma_t);
        arrival_delay ~ exponential(beta_b);
        arrival_time ~ normal(departure_delay + 
                              travel_time + 
                              arrival_delay, 
                              sigma_a);
    }
    """

    # Compile the model
    model = pystan.StanModel(model_code=code)

    # Define the observation dataset
    data = {
        "num": nb_samples,
        "departure_delay": departure_delay,
        "arrival_time": arrival_time,
        "travel_time": travel_time,
        "arrival_delay": arrival_delay
    }

    # Fit the model
    fit = model.sampling(data=data, iter=10000,
                         refresh=10000, warmup=1000,
                         chains=2, seed=1000)

    # Show a fit summary
    print(fit)

    # Sample some parameters from the posterior distribution
    ext = fit.extract()
    beta_a = ext["beta_a"]
    beta_b = ext["beta_b"]
    mu_t = ext["mu_t"]
    sigma_t = ext["sigma_t"]

    # Show the density estimations
    sns.set()

    fig, ax = plt.subplots(2, 2, figsize=(22, 12))

    sns.distplot(beta_a, kde_kws={"shade": True}, ax=ax[0, 0])
    sns.distplot(beta_b, kde_kws={"shade": True}, ax=ax[0, 1])
    sns.distplot(mu_t, kde_kws={"shade": True}, ax=ax[1, 0])
    sns.distplot(sigma_t, kde_kws={"shade": True}, ax=ax[1, 1])

    ax[0, 0].set_title(r"$\beta_0$", fontsize=22)
    ax[0, 1].set_title(r"$\beta_1$", fontsize=22)
    ax[1, 0].set_title(r"$\mu_t$", fontsize=22)
    ax[1, 1].set_title(r"$\sigma_t$", fontsize=22)

    plt.show()

