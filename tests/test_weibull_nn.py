import warnings
import os
import tempfile
import unittest
import arviz as az

from pmsurv.models.weibull_nn import WeibullModelNN
import data
from pathlib import Path

warnings.simplefilter("ignore")

class TestWeibullNN(unittest.TestCase):

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
        wb_model = WeibullModelNN()
        self.assertIsNotNone(wb_model)

    def test_create_model(self):
        print("test_create_model")
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = data.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse
        print(X)
        wb_model = WeibullModelNN()
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
        # fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'type': 'blackjax'}
        wb_model = WeibullModelNN()
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like',
                             var_names=["~w_", "~b_" "~k_", "~lambda_"
                                        "~k_det", "~lambda_det"])
        print(summary)

    def test_save_and_load(self):
        print("test_save_and_load")
        included_features = ['a', 'b', 'c']
        X, y = data.synthetic_data_random()
        print(X.shape, y.shape)
        fit_args = {'draws': 2000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True}
        wb_model = WeibullModelNN()
        wb_model.fit(X, y, inference_args=fit_args)

        summary_1 = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])

        pmsurv_dir = os.path.join(tempfile.gettempdir(), "pmsurv")
        Path(pmsurv_dir).mkdir(parents=True, exist_ok=True)

        file = os.path.join(pmsurv_dir, 'test.yaml')
        print('saving to ', file)
        score_1 = wb_model.score(X, y)
        wb_model.save(file)

        wb_model2 = WeibullModelNN()
        wb_model2.load(file)
        score_2 = wb_model2.score(X, y)
        summary_2 = az.summary(wb_model2.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])

        self.assertAlmostEqual(score_1, score_2, 2)
        print(summary_2)

        print(score_1, score_2)

    def test_score(self):
        print("test_fit_1")
        included_features = ['a']
        X, y = data.synthetic_data_weibull(lam_ctrl=1, lam_trt=2.5, k=1)
        y[:, 1] = 1 - y[:, 1]  # inverse
        print(X.shape, y.shape)
        fit_args = {'draws': 1000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True,
                    'progressbar': True}
        wb_model = WeibullModelNN()
        wb_model.fit(X, y, inference_args=fit_args)

        c_index = wb_model.score(X, y)
        print(f"c-index = {c_index}")
        self.assertGreater(c_index, 0.725)


if __name__ == '__main__':
    unittest.main()
