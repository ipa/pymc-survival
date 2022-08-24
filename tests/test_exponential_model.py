import warnings
import os
import tempfile
import unittest
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

from pmsurv.models.exponential_model import ExponentialModel
import data

warnings.simplefilter("ignore")

class TestExponentialModel(unittest.TestCase):

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
        wb_model = ExponentialModel()
        self.assertIsNotNone(wb_model)

    def test_create_model(self):
        print("test_create_model")
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = data.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        wb_model = ExponentialModel()
        fit_args = {'draws': 1000, 'tune': 500, 'target_accept': 0.85, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True}
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])
        print(summary)

        self.assertIsNotNone(wb_model)

    def test_fit(self):
        print("test_fit")
        included_features = ['a']
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = data.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse
        print(X.shape, y.shape)
        fit_args = {'draws': 1000, 'tune': 500, 'target_accept': 0.85, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True}
        wb_model = ExponentialModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])
        print(summary)
        self.assertAlmostEqual(summary['mean']['lambda_intercept'], lam_ctrl, 0)
        self.assertAlmostEqual(summary['mean']['lambda_A'], lam_trt - lam_ctrl, 0)

    def test_fit_intercept_only(self):
        print("test_fit_intercept_only")
        lam = 1.5
        k = 1
        X, y = data.synthetic_data_intercept_only(lam=lam, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'cores': 1, 'return_inferencedata': True}
        wb_model = ExponentialModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])
        print(summary)
        self.assertAlmostEqual(summary['mean']['lambda_intercept'], lam, 0)

    def test_save_and_load(self):
        print("test_save_and_load")
        X, y = data.synthetic_data_random()
        print(X.shape, y.shape)
        fit_args = {'draws': 2000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True}
        wb_model = ExponentialModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary_1 = az.summary(wb_model.trace, filter_vars='like', var_names=["~lambda_det"])

        file = os.path.join(tempfile.gettempdir(), 'test.yaml')
        print('saving to ', file)
        wb_model.save(file)

        wb_model2 = ExponentialModel()
        wb_model2.load(file)

        summary_2 = az.summary(wb_model2.trace, filter_vars='like', var_names=["~lambda_det"])

        self.assertAlmostEqual(summary_1['mean']['lambda_intercept'], summary_2['mean']['lambda_intercept'])
        self.assertAlmostEqual(summary_1['mean']['lambda_coefs_A'], summary_2['mean']['lambda_coefs_A'])
        self.assertAlmostEqual(summary_1['mean']['lambda_coefs_B'], summary_2['mean']['lambda_coefs_B'])
        self.assertAlmostEqual(summary_1['mean']['lambda_coefs_C'], summary_2['mean']['lambda_coefs_C'])

    def test_score(self):
        print("test_fit_1")
        included_features = ['a']
        X, y = data.synthetic_data_weibull(lam_ctrl=1, lam_trt=2.5, k=1)
        y[:, 1] = 1 - y[:, 1]  # inverse
        print(X.shape, y.shape)
        fit_args = {'draws': 1000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True,
                    'progressbar': True}
        wb_model = ExponentialModel()
        wb_model.fit(X, y, inference_args=fit_args)

        c_index = wb_model.score(X, y)
        print(f"c-index = {c_index}")
        self.assertGreater(c_index, 0.725)


if __name__ == '__main__':
    unittest.main()
