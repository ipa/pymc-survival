import unittest
import logging
import pandas as pd
from pmsurv.models.weibull_linear import WeibullModelLinear
from pmsurv.models.exponential_model import ExponentialModel
import arviz as az

logger = logging.getLogger(__name__)


class TestVeteranDataset(unittest.TestCase):
    def test_weibull_linear_veteran(self):
        logger.info("test_weibull_linear_veteran")
        data = pd.read_csv("tests/data/veteran.csv")
        X = data[['age', 'celltype', 'trt']]
        X['celltype_1'] = data['celltype'] == 1
        X['celltype_2'] = data['celltype'] == 2
        X['celltype_3'] = data['celltype'] == 3
        X['celltype_4'] = data['celltype'] == 4
        X['trt'] = data['trt'] == 2

        y = data[['time', 'status']].values
        y[:, 1] = 1 - y[:, 1]  # inverse

        wb_model = WeibullModelLinear()
        fit_args = {'draws': 4000, 'tune': 2000, 'target_accept': 0.85, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True}
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])
        logger.info(summary)

        c_index = wb_model.score(X, y)
        logger.info(c_index)

    def test_exponential_linear_veteran(self):
        logger.info("test_exponential_linear_veteran")
        data = pd.read_csv("tests/data/veteran.csv")
        X = data[['age']]
        X['celltype_2'] = data['celltype'] == 2
        X['celltype_3'] = data['celltype'] == 3
        X['celltype_4'] = data['celltype'] == 4
        X['celltype_1:trt'] = (data['celltype'] == 1) & (data['trt'] == 2)
        X['celltype_2:trt'] = (data['celltype'] == 2) & (data['trt'] == 2)
        X['celltype_3:trt'] = (data['celltype'] == 3) & (data['trt'] == 2)
        X['celltype_4:trt'] = (data['celltype'] == 4) & (data['trt'] == 2)
        X['trt'] = data['trt'] == 2
        y = data[['time', 'status']].values
        y[:, 1] = 1 - y[:, 1]  # inverse

        wb_model = ExponentialModel()
        fit_args = {'draws': 4000, 'tune': 2000, 'target_accept': 0.85, 'chains': 2, 'cores': 1,
                    'return_inferencedata': True}
        wb_model.fit(X, y, inference_args=fit_args)

        summary = az.summary(wb_model.trace, filter_vars='like', var_names=["~k_det", "~lambda_det"])
        logger.info(summary)

        c_index = wb_model.score(X, y)
        logger.info(c_index)


if __name__ == '__main__':
    unittest.main()
