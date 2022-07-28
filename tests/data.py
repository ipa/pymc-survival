import warnings
warnings.simplefilter("ignore")
import numpy as np
from numpy.random import default_rng
from scipy.stats import expon

def synthetic_data_random(n_samples=100):
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
    return X, y
