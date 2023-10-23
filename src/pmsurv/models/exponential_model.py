import logging
import lifelines
import pandas as pd
import scipy.stats as st
import numpy as np
import pymc as pm
from pmsurv.exc import PyMCModelsError
from pmsurv.models.base import BayesianModel
import pmsurv.utils

logger = logging.getLogger(__name__)


class ExponentialModel(BayesianModel):
    def __init__(self, priors_sd=1.0):
        super(ExponentialModel, self).__init__()
        self.column_names = None
        self.max_time = None
        self.priors = None
        self.fit_args = None
        self.num_training_samples = None
        self.num_pred = None
        self.priors_sd = priors_sd

    def _get_priors(self):
        return {
            'lambda_mu': 1.0,
            'lambda_sd': self.priors_sd,
            'lambda_coefs_mu': 0.0,
            'lambda_coefs_sd': self.priors_sd
        }

    def __str__(self):
        str_output = "ExponentialModel \n\r"
        str_output += str(self.column_names) + "\r\n"
        str_output += str(self.priors) + "\r\n"
        str_output += str(self.fit_args) + "\r\n"
        return str_output

    def create_model(self, X=None, y=None, priors=None):
        if self.priors is None:
            self.priors = priors
        if self.priors is None:
            self.priors = self._get_priors()

        model = pm.Model()
        with model:
            if X is None:
                X = np.zeros([self.num_training_samples, self.num_pred])

            model_input = pm.MutableData("model_input", X)
            if y is not None:
                time_censor_ = pm.MutableData("time_censor", y[y[:, 1] == 1, 0]).astype('float32')
                time_uncensor_ = pm.MutableData("time_uncensor", y[y[:, 1] == 0, 0]).astype('float32')
                censor_ = pm.MutableData("censor", y[:, 1].astype(np.int8))

        with model:
            logger.info("Priors: {}".format(str(self.priors)))

            lambda_intercept = pm.Normal("lambda_intercept",
                                         mu=self.priors['lambda_mu'] if 'lambda_intercept_mu' not in self.priors else self.priors['lambda_intercept_mu'],
                                         sigma=self.priors['lambda_sd'] if 'lambda_intercept_sd' not in self.priors else self.priors['lambda_intercept_sd']).astype('float32')

            lambda_coefs = []
            for i, cn in enumerate(self.column_names):
                feature_name = f'lambda_{cn}'
                lambda_coef = pm.Normal(feature_name,
                                        mu=self.priors['lambda_coefs_mu'] if f'{feature_name}_mu' not in self.priors else self.priors[f'{feature_name}_mu'],
                                        sigma=self.priors['lambda_coefs_sd'] if f'{feature_name}_sd' not in self.priors else self.priors[f'{feature_name}_sd']).astype('float32')
                lambda_coefs.append(model_input[:, i] * lambda_coef)
            lambda_det = pm.Deterministic("lambda_det", pm.math.exp(lambda_intercept + sum(lambda_coefs))).astype('float32')

            if y is not None:
                censor_ = pm.math.eq(censor_, 1)
                lambda_det_uncensor = lambda_det[~censor_]
                y = pm.Exponential("y", pm.math.ones_like(time_uncensor_) / lambda_det_uncensor,
                                   observed=time_uncensor_)

                def exponential_lccdf(lam, time):
                    """ Log complementary cdf of Exponential distribution. """
                    return -(lam * time)

                lambda_det_censor = lambda_det[censor_]
                y_cens = pm.Potential(  # noqa:F841
                    "y_cens", exponential_lccdf(pm.math.ones_like(time_censor_) / lambda_det_censor, time_censor_)
                )

        return model

    def fit(self, X, y, inference_args=None, priors=None):
        self.num_training_samples, self.num_pred = X.shape
        if isinstance(X, pd.DataFrame):
            self.column_names = list(X.columns.values)
        else:
            self.column_names = ["column_%i" % i for i in range(0, self.num_pred)]

        self.max_time = int(np.max(y[:, 0]))

        if y.ndim != 1:
            logger.info('squeeze')
            y = np.squeeze(y)

        if not inference_args:
            inference_args = BayesianModel._get_default_inference_args()

        if self.cached_model is None:
            logger.info('create from fit')
            self.cached_model = self.create_model(X, y, priors=priors)

        with self.cached_model:
            pm.set_data({
                'model_input': X,
                'time_censor': y[y[:, 1] == 1, 0],
                'time_uncensor': y[y[:, 1] == 0, 0],
                'censor': y[:, 1].astype(np.int32)
            })

        self._inference(inference_args)

        return self

    def predict(self, X, return_std=True, num_ppc_samples=1000, resolution=10):
        if self.trace is None:
            raise PyMCModelsError('Run fit on the model before predict.')

        num_samples = X.shape[0]

        if self.cached_model is None:
            logger.info('create from predict')
            self.cached_model = self.create_model()

        with self.cached_model:
            pm.set_data({
                'model_input': X,
            })

        ppc = pm.sample_posterior_predictive(self.trace, model=self.cached_model, return_inferencedata=False,
                                             random_seed=0, progressbar=False, var_names=['lambda_det'])
        logger.info("")

        t_plot = pmsurv.utils.get_time_axis(0, self.max_time, resolution)

        logger.info(ppc.keys())
        pp_lambda = np.mean(ppc['lambda_det'].reshape(-1, X.shape[0]), axis=0)
        pp_lambda_std = np.std(ppc['lambda_det'].reshape(-1, X.shape[0]), axis=0)

        t_plot_rep = np.repeat(t_plot, num_samples).reshape((len(t_plot), -1))

        pp_surv_mean = st.expon.sf(t_plot_rep / pp_lambda).T
        pp_surv_lower = st.expon.sf(t_plot_rep / (pp_lambda - pp_lambda_std)).T
        pp_surv_upper = st.expon.sf(t_plot_rep / (pp_lambda + pp_lambda_std)).T
        pp_surv_lower = np.nan_to_num(pp_surv_lower, 0)
        pp_surv_upper = np.nan_to_num(pp_surv_upper, 1)

        return pp_surv_mean, pp_surv_lower, pp_surv_upper

    def score(self, X, y, num_ppc_samples=1000):
        surv_prob, _, _ = self.predict(X, num_ppc_samples=num_ppc_samples)
        surv_prob_median = np.median(surv_prob, axis=1)

        c_index = lifelines.utils.concordance_index(y[:, 0], surv_prob_median, 1 - y[:, 1])
        return c_index

    def save(self, file_prefix, **kwargs):
        custom_params = {
            'column_names': self.column_names,
            'priors': self.priors,
            # 'inference_args': self.inference_args,
            'max_observed_time': self.max_time,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }
        logger.info('store: ', self.priors)

        super(ExponentialModel, self).save(file_prefix, custom_params)

    def load(self, file_prefix, **kwargs):
        params = super(ExponentialModel, self).load(file_prefix)
        logger.info('load: ', params['priors'])
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
        self.column_names = params['column_names']
        self.priors = params['priors']
        self.max_time = params['max_observed_time']
