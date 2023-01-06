import warnings
import os
import tempfile
import unittest
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

from pmsurv.models.gaussian_process import GaussianProcessModel
import tests.syntheticdata

warnings.simplefilter("ignore")

class TestGaussianProcessModelModel(unittest.TestCase):

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
        wb_model = GaussianProcessModel()
        self.assertIsNotNone(wb_model)

    def test_create_model(self):
        print("test_create_model")
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        wb_model = GaussianProcessModel()
        fit_args = {'draws': 100, 'tune': 50, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True, 'type': 'blackjax' }
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~f"])
        print(summary)

        self.assertIsNotNone(wb_model)

    def test_fit(self):
        print("test_fit")
        included_features = ['a']
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse
        print(X.shape, y.shape)
        fit_args = {'draws': 100, 'tune': 50, 'target_accept': 0.8, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True, 'type': 'map', 'progressbar': True }
        wb_model = GaussianProcessModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~f"])
        print(summary)
        c_index = wb_model.score(X[:10,:], y[:10,:])
        print(c_index)
        # self.assertAlmostEqual(np.exp(summary['mean']['lambda_intercept']), lam_ctrl, 0)
        # self.assertAlmostEqual(np.exp(summary['mean']['lambda_det_pred']), lam_trt - lam_ctrl, 0)

        # plt.plot(wb_model.approx.hist)
        # plt.show()


    def test_save_and_load(self):
        print("test_save_and_load")
        X, y = tests.syntheticdata.synthetic_data_random()
        print(X.shape, y.shape)
        fit_args = {'draws': 200, 'tune': 100, 'chains': 2, 'cores': 1, 'return_inferencedata': True, 'type': 'blackjax'}
        wb_model = GaussianProcessModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary_1 = az.summary(wb_model.trace, filter_vars='like', var_names=["~f"])

        pmsurv_dir = os.path.join(tempfile.gettempdir(), "pmsurv")
        Path(pmsurv_dir).mkdir(parents=True, exist_ok=True)

        file = os.path.join(pmsurv_dir, 'test.yaml')
        print('saving to ', file)
        wb_model.save(file)

        wb_model2 = GaussianProcessModel()
        wb_model2.load(file)

        summary_2 = az.summary(wb_model2.trace, filter_vars='like', var_names=["~f"])
        print(summary_1)
        print(summary_2)
        # self.assertAlmostEqual(summary_1['mean']['lambda_intercept'], summary_2['mean']['lambda_intercept'])
        # self.assertAlmostEqual(summary_1['mean']['lambda_det'], summary_2['mean']['lambda_det'])

    def test_score(self):
        print("test_fit_1")
        included_features = ['a']
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=1, lam_trt=2.5, k=1)
        y[:, 1] = 1 - y[:, 1]  # inverse
        print(X.shape, y.shape)
        fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'cores': 1, 'return_inferencedata': True,
                    'progressbar': True, 'type':'blackjax' }
        wb_model = GaussianProcessModel()
        wb_model.fit(X, y, inference_args=fit_args)

        c_index = wb_model.score(X[:10,:], y[:10,:])
        print(f"c-index = {c_index}")
        self.assertGreater(c_index, 0.725)


if __name__ == '__main__':
    unittest.main()
