import warnings
import unittest
import logging
from pmsurv.models.exponential_model import ExponentialModel
import tests.syntheticdata

warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)


class TestSamplers(unittest.TestCase):
    def get_data():
        lam_ctrl = 1
        lam_trt = 2.5
        k = 1
        X, y = tests.syntheticdata.synthetic_data_weibull(lam_ctrl=lam_ctrl, lam_trt=lam_trt, k=k)
        y[:, 1] = 1 - y[:, 1]  # inverse
        return X, y

    def test_fit_pymc(self):
        logger.info("test_fit_pymc")

        X, y = TestSamplers.get_data()

        fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'cores': 1,
                    'nuts_sampler': 'pymc',
                    'return_inferencedata': True}
        model = ExponentialModel()
        model.fit(X, y, inference_args=fit_args)

        self.assertTrue(True)

    def test_fit_nutpie(self):
        logger.info("test_fit_pymc")

        X, y = TestSamplers.get_data()

        fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'cores': 1,
                    'nuts_sampler': 'nutpie',
                    'return_inferencedata': True}
        model = ExponentialModel()
        model.fit(X, y, inference_args=fit_args)

        self.assertTrue(True)

    @unittest.skip("Requires modification for JAX upgrade")
    def test_fit_blackjax(self):
        logger.info("test_fit_blackjax")

        X, y = TestSamplers.get_data()

        fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'cores': 1,
                    'nuts_sampler': 'blackjax',
                    'return_inferencedata': True}
        model = ExponentialModel()
        model.fit(X, y, inference_args=fit_args)

        self.assertTrue(True)

    @unittest.skip("Requires modification for JAX upgrade")
    def test_fit_numpyro(self):
        logger.info("test_fit_numpyro")

        X, y = TestSamplers.get_data()

        fit_args = {'draws': 1000, 'tune': 500, 'chains': 2, 'cores': 1,
                    'nuts_sampler': 'numpyro',
                    'return_inferencedata': True}
        model = ExponentialModel()
        model.fit(X, y, inference_args=fit_args)

        self.assertTrue(True)
