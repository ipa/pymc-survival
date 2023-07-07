import warnings
import os
import tempfile
import unittest
import arviz as az
import logging
from pmsurv.models.weibull_nn import WeibullModelNN
import tests.syntheticdata
from pathlib import Path

warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)


class TestWeibullNN(unittest.TestCase):

    def test_setup(self):
        logger.info("test_setup")
        wb_model = WeibullModelNN()
        self.assertIsNotNone(wb_model)

    def test_create_model(self):
        logger.info("test_create_model")
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse

        wb_model = WeibullModelNN()
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
        # fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'type': 'blackjax'}
        wb_model = WeibullModelNN()
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like',
                             var_names=["~w_", "~b_" "~k_", "~lambda_"
                                        "~k_det", "~lambda_det"])
        logger.info(summary)

    def test_save_and_load(self):
        logger.info("test_save_and_load")
        X, y = tests.syntheticdata.synthetic_data_random()

        fit_args = {'draws': 1000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True}
        wb_model = WeibullModelNN()
        wb_model.fit(X, y, inference_args=fit_args)

        pmsurv_dir = os.path.join(tempfile.gettempdir(), "pmsurv")
        Path(pmsurv_dir).mkdir(parents=True, exist_ok=True)

        file = os.path.join(pmsurv_dir, 'test.yaml')
        logger.info('saving to ', file)
        score_1 = wb_model.score(X, y)
        wb_model.save(file)

        wb_model2 = WeibullModelNN()
        wb_model2.load(file)
        score_2 = wb_model2.score(X, y)
        summary_2 = az.summary(wb_model2.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])

        self.assertAlmostEqual(score_1, score_2, 2)
        logger.info(summary_2)

        logger.info(score_1, score_2)

    def test_score(self):
        logger.info("test_fit_1")
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=1, lam_trt=2.5, k=1)
        y[:, 1] = 1 - y[:, 1]  # inverse

        fit_args = {'draws': 1000, 'tune': 1000, 'chains': 2, 'cores': 1, 'return_inferencedata': True,
                    'progressbar': False}
        wb_model = WeibullModelNN()
        wb_model.fit(X, y, inference_args=fit_args)

        c_index = wb_model.score(X, y)
        logger.info(f"c-index = {c_index}")
        self.assertGreater(c_index, 0.7)


if __name__ == '__main__':
    unittest.main()
