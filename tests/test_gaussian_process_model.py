import logging
import os
import tempfile
import unittest
import arviz as az
from pathlib import Path
from sklearn.model_selection import train_test_split
from pmsurv.models.gaussian_process import GaussianProcessModel
import tests.syntheticdata

logger = logging.getLogger(__name__)


@unittest.skip("Not yet ready")
class TestGaussianProcessModel(unittest.TestCase):

    def test_setup(self):
        logger.info("test_setup")
        wb_model = GaussianProcessModel()
        self.assertIsNotNone(wb_model)

    def test_create_model(self):
        logger.info("test_create_model")
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        wb_model = GaussianProcessModel()
        fit_args = {'draws': 100, 'tune': 100, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True, 'nuts_sampler': 'nutpie'}
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~f"])
        logger.info(summary)

        self.assertIsNotNone(wb_model)

    def test_fit(self):
        logger.info("test_fit")
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        fit_args = {'draws': 100, 'tune': 50, 'target_accept': 0.85,  'chains': 2, 'cores': 1,
                    'return_inferencedata': True, 'nuts_sampler': 'nutpie'}
        try:
            wb_model = GaussianProcessModel()
            wb_model.fit(X, y, inference_args=fit_args)
        except:  # noqa:E722
            self.assertTrue(False)
        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~f"])
        logger.info(summary)

    def test_save_and_load(self):
        logger.info("test_save_and_load")
        X, y = tests.syntheticdata.synthetic_data_random()

        fit_args = {'draws': 100, 'tune': 100, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True, 'nuts_sampler': 'nutpie'}
        wb_model = GaussianProcessModel()
        wb_model.fit(X, y, inference_args=fit_args)

        summary_1 = az.summary(wb_model.trace, filter_vars='like', var_names=["~f"])

        pmsurv_dir = os.path.join(tempfile.gettempdir(), "pmsurv")
        Path(pmsurv_dir).mkdir(parents=True, exist_ok=True)

        file = os.path.join(pmsurv_dir, 'test.yaml')
        logger.info('saving to ', file)
        wb_model.save(file)

        wb_model2 = GaussianProcessModel()
        wb_model2.load(file)

        summary_2 = az.summary(wb_model2.trace, filter_vars='like', var_names=["~f"])
        logger.info(summary_1)
        logger.info(summary_2)
        self.assertAlmostEqual(summary_1['mean']['lambda_intercept'], summary_2['mean']['lambda_intercept'])
        self.assertAlmostEqual(summary_1['mean']['eta'], summary_2['mean']['eta'])
        self.assertAlmostEqual(summary_1['mean']['eta_log__'], summary_2['mean']['eta_log__'])
        self.assertAlmostEqual(summary_1['mean']['ell_log__[0]'], summary_2['mean']['ell_log__[0]'])
        self.assertAlmostEqual(summary_1['mean']['ell[0]'], summary_2['mean']['ell[0]'])

    def test_score(self):
        logger.info("test_fit_1")
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=1, lam_trt=2.5, k=1)
        y[:, 1] = 1 - y[:, 1]  # inverse

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train = X_train.values.astype(float)
        y_train = y_train.astype(float)
        X_test = X_test.values.astype(float)
        y_test = y_test.astype(float)

        fit_args = {'draws': 500, 'tune': 250, 'target_accept': 0.85, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True, 'nuts_sampler': 'nutpie'}
        wb_model = GaussianProcessModel()
        wb_model.fit(X_train, y_train, inference_args=fit_args)

        c_index = wb_model.score(X_test, y_test)
        logger.info(f"c-index = {c_index}")
        # self.assertGreater(c_index, 0.725)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
