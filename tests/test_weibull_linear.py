import warnings

import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
import os
import tempfile
import unittest

from pmsurv.models.weibull_linear import WeibullModelLinear
import pmsurv.utils
from data import synthetic_data_random

class TestWeibullLinear(unittest.TestCase):

    def test_setup(self):
        print("test_setup")
        priors = {
            'lambda_mu': 0,
            'lambda_sd': 5,
            'k_mu': 0,
            'k_sd': 5,
            'coefs_mu': 0,
            'coefs_sd': 0.5
        }
        included_features = ['a', 'b']
        wb_model = WeibullModelLinear()
        self.assertIsNotNone(wb_model)

    def test_fit(self):
        print("test_fit_1")
        included_features = ['a', 'b', 'c']
        X, y = synthetic_data_random()
        print(X.shape, y.shape)
        fit_args = {'draws': 100, 'tune': 100, 'chains': 1, 'cores': 1, 'return_inferencedata': True}
        wb_model = WeibullModelLinear()
        wb_model.fit(X, y, inference_args=fit_args, column_names=included_features)


    # def test_save(self):
    #     print("test_save")
    #     included_features = ['a', 'b', 'c']
    #     X, y = synthetic_data_random()
    #     print(X.shape, y.shape)
    #     fit_args = {'draws': 100, 'tune': 100, 'chains': 1, 'cores': 1, 'return_inferencedata': True}
    #     wb_model = WeibullModelLinear()
    #     wb_model.fit(X, y, inference_args=fit_args, column_names=included_features)
    #     wb_model.save(os.path.join(tempfile.gettempdir(), 'test'))
    #
    #     wb_model2 = WeibullModelLinear()
    #     wb_model2.load(os.path.join(tempfile.gettempdir(), 'test'))
    #
    # def test_save_and_load(self):
    #     print("test_save")
    #     included_features = ['a', 'b', 'c']
    #     X, y = synthetic_data_random()
    #     print(X.shape, y.shape)
    #     fit_args = {'draws': 100, 'tune': 100, 'chains': 1, 'cores': 1, 'return_inferencedata': True}
    #     wb_model = WeibullModelLinear()
    #     wb_model.fit(X, y, inference_args=fit_args, column_names=included_features)
    #     wb_model.save(os.path.join(tempfile.gettempdir(), 'test'))
    #
    #     wb_model2 = WeibullModelLinear()
    #     wb_model2.load(os.path.join(tempfile.gettempdir(), 'test'))
    #
    #     X_test = X[1:20, :]
    #     y_test = y[1:20, :]
    #     c_index = wb_model2.score(X_test, y_test)
    #     print(f"c-index = {c_index}")
    #     self.assertGreater(c_index, 0.99)


    # def test_score(self):
    #     print("test_fit_1")
    #     included_features = ['a', 'b', 'c']
    #     X, y = synthetic_data_random(n_samples=500)
    #     print(X.shape, y.shape)
    #     fit_args = {'draws': 1000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True, 'progressbar': True}
    #     wb_model = WeibullModelLinear()
    #     wb_model.fit(X, y, inference_type='nuts', inference_args=fit_args, column_names=included_features)
    #
    #     X_test = X[1:20, :]
    #     y_test = y[1:20, :]
    #     c_index = wb_model.score(X_test, y_test)
    #     print(f"c-index = {c_index}")
    #     self.assertGreater(c_index, 0.99)

    # def test_score_advi(self):
    #     print("test_fit_1")
    #     included_features = ['a', 'b', 'c']
    #     X, y = synthetic_data_random(n_samples=500)
    #     print(X.shape, y.shape)
    #     fit_args = {'n': 20000, 'draws': 1000, 'progressbar': True}
    #     wb_model = WeibullModelLinear()
    #     wb_model.fit(X, y, inference_type='advi', inference_args=fit_args, column_names=included_features)
    #
    #     X_test = X[1:20, :]
    #     y_test = y[1:20, :]
    #     c_index = wb_model.score(X_test, y_test)
    #     print(f"c-index = {c_index}")
    #     self.assertGreater(c_index, 0.99)
    #
    # def test_predict(self):
    #     print("test_predict")
    #     included_features = ['a', 'b', 'c']
    #     X, y = synthetic_data_random(n_samples=500)
    #     print(X.shape, y.shape)
    #     fit_args = {'draws': 1000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True, 'progressbar': True}
    #     wb_model = WeibullModelLinear()
    #     wb_model.fit(X, y, inference_args=fit_args, column_names=included_features)
    #
    #     X_test = X[1:20, :]
    #     print(X_test.shape)
    #     mu, lower, upper = wb_model.predict(X_test, resolution=10)
    #
    #     t_plot = pdeepsurv.utils.get_time_axis(0, wb_model.max_time, 10)
    #     print(mu.shape, t_plot.shape)
    #
    #     # import matplotlib.pyplot as plt
    #     # print(t_plot.shape, mu.shape, lower.shape, upper.shape)
    #     # plt.plot(t_plot, mu[1, :])
    #     # plt.show()
    #     self.assertEqual(19, mu.shape[0])

if __name__ == '__main__':
    unittest.main()
