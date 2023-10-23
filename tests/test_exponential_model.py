import os
import logging
import tempfile
import unittest
import arviz as az
from pathlib import Path
from pmsurv.models.exponential_model import ExponentialModel
import tests.syntheticdata

logger = logging.getLogger(__name__)


class TestExponentialModel(unittest.TestCase):

    def test_setup(self):
        logger.info("test_setup")
        model = ExponentialModel()
        self.assertIsNotNone(model)

    def test_create_model(self):
        logger.info("test_create_model")
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        wb_model = ExponentialModel()
        fit_args = {'draws': 1000, 'tune': 500, 'target_accept': 0.85, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True}
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])
        logger.info(summary)

        self.assertIsNotNone(wb_model)

    def test_fit(self):
        logger.info("test_fit")
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        fit_args = {'draws': 1000, 'tune': 500, 'target_accept': 0.85, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True}
        wb_model = ExponentialModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])
        logger.info(summary)
        self.assertAlmostEqual(summary['mean']['lambda_intercept'], lam_ctrl, 0)
        self.assertAlmostEqual(summary['mean']['lambda_A'], lam_trt - lam_ctrl, 0)

    def test_fit_intercept_only(self):
        logger.info("test_fit_intercept_only")
        lam = 1.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_intercept_only(lam=lam, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'cores': 1, 'return_inferencedata': True}
        wb_model = ExponentialModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])
        logger.info(summary)
        self.assertAlmostEqual(summary['mean']['lambda_intercept'], lam, 0)

    def test_save_and_load(self):
        logger.info("test_save_and_load")
        X, y = tests.syntheticdata.synthetic_data_random()

        fit_args = {'draws': 2000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True}
        wb_model = ExponentialModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary_1 = az.summary(wb_model.trace, filter_vars='like', var_names=["~lambda_det"])

        pmsurv_dir = os.path.join(tempfile.gettempdir(), "pmsurv")
        Path(pmsurv_dir).mkdir(parents=True, exist_ok=True)

        file = os.path.join(pmsurv_dir, 'test.yaml')
        logger.info('saving to ', file)
        wb_model.save(file)

        wb_model2 = ExponentialModel()
        wb_model2.load(file)

        summary_2 = az.summary(wb_model2.trace, filter_vars='like', var_names=["~lambda_det"])

        self.assertAlmostEqual(summary_1['mean']['lambda_intercept'], summary_2['mean']['lambda_intercept'])
        self.assertAlmostEqual(summary_1['mean']['lambda_A'], summary_2['mean']['lambda_A'])
        self.assertAlmostEqual(summary_1['mean']['lambda_B'], summary_2['mean']['lambda_B'])
        self.assertAlmostEqual(summary_1['mean']['lambda_C'], summary_2['mean']['lambda_C'])

    def test_score(self):
        logger.info("test_fit_1")
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=1, lam_trt=2.5, k=1)
        y[:, 1] = 1 - y[:, 1]  # inverse

        fit_args = {'draws': 1000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True}
        wb_model = ExponentialModel()
        wb_model.fit(X, y, inference_args=fit_args)

        c_index = wb_model.score(X, y)
        logger.info(f"c-index = {c_index}")
        self.assertGreater(c_index, 0.725)


if __name__ == '__main__':
    unittest.main()
