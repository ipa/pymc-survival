import warnings
import numpy as np
from numpy.random import default_rng
import pymc as pm
import arviz as az
import time

warnings.simplefilter("ignore")


def synthetic_data_random(n_samples=100):
    rng = default_rng(seed=0)
    x1 = rng.standard_normal((n_samples, 1)) + 3
    x2 = rng.standard_normal((n_samples, 1)) + 3
    x3 = rng.integers(low=0, high=2, size=(n_samples, 1))
    X = np.hstack((x1, x2, x3))

    y_time = 100 + x1 * 2 + x2 * 1.5 + x3 * 3.4
    y_event = ((x3 == 1) & (y_time > 110))  # rng.choice([1, 0], size=(n_samples, 1), p=[0.2, 0.8]).astype(np.int_)
    y = np.hstack((y_time, y_event))
    return X, y


def main():
    X, y = synthetic_data_random()
    censor = (y[:, 1] == 0)
    time_obs = y[:, 0]
    print(np.min(time_obs), np.max(time_obs))

    print(X.dtype)
    print(y.dtype)
    print(time_obs.dtype, time_obs.shape)
    print(censor.dtype, censor.shape)

    model = pm.Model()

    start = time.time()
    with model:
        lambda_intercept = pm.Normal("lambda_intercept", mu=0, sigma=10, shape=(1,))
        k_intercept = pm.Normal('k_intercept', mu=0, sigma=10, shape=(1,))

        lambda_coefs = pm.Normal('lambda_coefs', mu=0, sigma=10, shape=(3,))
        k_coefs = 0.0
        # k_coefs = pm.Normal('k_coefs', mu=0, sigma=10, shape=(3,))

        lambda_ = pm.Deterministic("lambda_", pm.math.exp(lambda_intercept + (lambda_coefs * X).sum(axis=1)))
        k_ = pm.Deterministic("k_", pm.math.exp(k_intercept + (k_coefs * X).sum(axis=1)))
        print(k_.shape.eval(), lambda_.shape.eval(), time_obs.shape)

        y = pm.Weibull("y", alpha=k_[~censor], beta=lambda_[~censor], observed=time_obs[~censor])

        def weibull_lccdf(x, alpha, beta):
            """ Log complementary cdf of Weibull distribution. """
            return -((x / beta) ** alpha)

        y_cens = pm.Potential("y_cens", weibull_lccdf(time_obs[censor], alpha=k_[censor], beta=lambda_[censor]))  # noqa:F841

    end = time.time()
    print('compile: ', end - start)

    start = time.time()
    with model:
        trace = pm.sample(draws=2000, tune=1000, chains=2, cores=1, progressbar=True)
    end = time.time()
    print('sample: ', end - start)

    print(az.summary(trace, var_names=["lambda_coefs", "lambda_intercept", "k_intercept", 'k_coefs'], filter_vars='like'))


if __name__ == "__main__":
    main()
