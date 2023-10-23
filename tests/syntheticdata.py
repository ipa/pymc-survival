import warnings
import pandas as pd
import numpy as np
from numpy.random import default_rng
from scipy.stats import expon, weibull_min

warnings.simplefilter("ignore")


def synthetic_data_random(n_samples=100, as_dataframe=True):
    coefficients = [2, 4, 6]
    rng = default_rng(seed=0)
    x1 = rng.standard_normal((n_samples, 1)) + 3
    x2 = rng.standard_normal((n_samples, 1)) + 3
    x3 = rng.integers(low=0, high=2, size=(n_samples, 1))
    X = np.hstack((x1, x2, x3))

    rate = 1 / (coefficients[0]*x1 + coefficients[1]*x2 + coefficients[2]*x3)
    y_time = expon.rvs(rate)
    y_event = rng.choice([1, 0], size=(n_samples, 1), p=[0.2, 0.8]).astype(np.int_)
    y = np.hstack((y_time, y_event))

    if as_dataframe:
        X = pd.DataFrame(X, columns=['A', 'B', 'C'])

    return X, y


def synthetic_data_weibull(lam_ctrl, lam_trt, k, n_samples=100, as_dataframe=True):
    X = np.stack((np.zeros((int(n_samples / 2))), np.ones((int(n_samples / 2))))).flatten()
    X = np.expand_dims(X, axis=1)

    y_time_trt = weibull_min.rvs(c=np.exp(k), scale=np.exp(lam_trt), size=int(n_samples / 2)).T.flatten()
    y_time_ctrl = weibull_min.rvs(c=np.exp(k), scale=np.exp(lam_ctrl), size=int(n_samples / 2)).T.flatten()
    y_time = np.stack((y_time_ctrl, y_time_trt)).flatten()

    y_event = y_time < 15  # rng.choice([1, 0], size=(n_samples), p=[0.5, 0.5]).astype(np.int_)
    y = np.vstack((y_time, y_event)).T

    if as_dataframe:
        X = pd.DataFrame(X, columns=['A'])

    return X, y


def synthetic_data_intercept_only(lam, k, n_samples=100, as_dataframe=True):
    X = np.zeros((n_samples, 1))
    y_time = weibull_min.rvs(c=np.exp(k), scale=np.exp(lam), size=(n_samples)).T.flatten()
    y_event = y_time < 10  # rng.choice([1, 0], size=(n_samples), p=[0.5, 0.5]).astype(np.int_)
    y = np.vstack((y_time, y_event)).T

    if as_dataframe:
        X = pd.DataFrame(X, columns=['A'])

    return X, y
