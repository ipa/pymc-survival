import logging
import numpy as np
import scipy.stats as st
import lifelines
import pandas as pd
import pymc as pm
from pmsurv.models.base import BayesianModel
import pmsurv
import pmsurv.utils
from pmsurv.exc import PyMCModelsError


class WeibullModelBase(BayesianModel):

    def __init__(self):
        super(WeibullModelBase, self).__init__()
        self.priors = None

    def create_model(self, X=None, y=None, priors=None):
        raise NotImplementedError

    def fit(self, X, y, inference_args=None, priors=None):
        self.num_training_samples, self.num_pred = X.shape
        if isinstance(X, pd.DataFrame):
            self.column_names = list(X.columns.values)
        else:
            self.column_names = ["column_%i" % i for i in range(0, self.num_pred)]

        self.max_time = int(np.max(y[:, 0]))

        if y.ndim != 1:
            logging.debug('squeeze')
            y = np.squeeze(y)

        if not inference_args:
            inference_args = BayesianModel._get_default_inference_args()

        if self.cached_model is None:
            logging.info('create from fit')
            self.cached_model = self.create_model(X, y, priors=priors)

        with self.cached_model:
            pm.set_data({
                'model_input': X,
                'time_censor': y[y[:, 1] == 1, 0],
                'time_uncensor': y[y[:, 1] == 0, 0],
                'censor': y[:, 1].astype(np.int32)
            })

        self._inference(inference_args)
        print('fitted')

    def predict(self, X, return_std=True, num_ppc_samples=1000, resolution=10):
        if self.trace is None:
            raise PyMCModelsError('Run fit on the model before predict.')

        num_samples = X.shape[0]

        if self.cached_model is None:
            logging.info('create from predict')
            self.cached_model = self.create_model()

        with self.cached_model:
            pm.set_data({
                'model_input': X
            })

        ppc = pm.sample_posterior_predictive(self.trace, model=self.cached_model, return_inferencedata=False,
                                             random_seed=0, progressbar=False, var_names=['lambda_det', 'k_det'])
        logging.info("")

        t_plot = pmsurv.utils.get_time_axis(0, self.max_time, resolution)

        logging.info(ppc.keys())
        pp_lambda = np.mean(ppc['lambda_det'].reshape(-1, X.shape[0]), axis=0)
        pp_k = np.mean(ppc['k_det'].reshape(-1, X.shape[0]), axis=0)
        pp_lambda_std = np.std(ppc['lambda_det'].reshape(-1, X.shape[0]), axis=0)
        pp_k_std = np.std(ppc['k_det'].reshape(-1, X.shape[0]), axis=0)

        t_plot_rep = np.repeat(t_plot, num_samples).reshape((len(t_plot), -1))

        pp_surv_mean = 1 - st.weibull_min.cdf(t_plot_rep, c=pp_k, scale=pp_lambda).T
        pp_surv_lower = 1 - st.weibull_min.cdf(t_plot_rep, c=pp_k - pp_k_std, scale=pp_lambda - pp_lambda_std).T
        pp_surv_upper = 1 - st.weibull_min.cdf(t_plot_rep, c=pp_k + pp_k_std, scale=pp_lambda + pp_lambda_std).T
        pp_surv_lower = np.nan_to_num(pp_surv_lower, 0)
        pp_surv_upper = np.nan_to_num(pp_surv_upper, 1)

        return pp_surv_mean, pp_surv_lower, pp_surv_upper

    def score(self, X, y, num_ppc_samples=1000):
        surv_prob, _, _ = self.predict(X, num_ppc_samples=num_ppc_samples)
        surv_prob_median = np.median(surv_prob, axis=1)

        c_index = lifelines.utils.concordance_index(y[:, 0], surv_prob_median, 1 - y[:, 1])
        return c_index
