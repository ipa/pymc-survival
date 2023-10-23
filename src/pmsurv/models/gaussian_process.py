import lifelines
import scipy.stats as st
import numpy as np
import pandas as pd
import pymc as pm
from pmsurv.exc import PyMCModelsError
from pmsurv.models.base import BayesianModel
import pmsurv.utils


class GaussianProcessModel(BayesianModel):
    def __init__(self):
        super(GaussianProcessModel, self).__init__()
        from warnings import warn
        warn('This model is not yet supported and under development!', UserWarning)
        self.column_names = None
        self.max_time = None
        self.priors = None
        self.fit_args = None
        self.num_training_samples = None
        self.num_pred = None
        self.gp = None

    @staticmethod
    def _get_default_priors():
        return {
            'lambda_mu': 0,
            'lambda_sd': 1,
            'lambda_coefs_mu': 0,
            'lambda_coefs_sd': 1
        }

    def __str__(self):
        str_output = "Gaussian Process \n\r"
        str_output += str(self.column_names) + "\r\n"
        str_output += str(self.priors) + "\r\n"
        str_output += str(self.fit_args) + "\r\n"
        return str_output

    def create_model(self, X=None, y=None, priors=None):
        if self.priors is None:
            self.priors = priors
        if self.priors is None:
            self.priors = GaussianProcessModel._get_default_priors()

        model = pm.Model()
        with model:
            if X is None:
                X = np.zeros([self.num_training_samples, self.num_pred])

            model_input = pm.MutableData("model_input", X)
            if y is not None:
                time_censor_ = pm.MutableData("time_censor", y[y[:, 1] == 1, 0])
                time_uncensor_ = pm.MutableData("time_uncensor", y[y[:, 1] == 0, 0])
                censor_ = pm.MutableData("censor", y[:, 1].astype(np.int8))

            ell = pm.InverseGamma("ell",
                                  mu=1.0,
                                  sigma=0.5,
                                  shape=(len(self.column_names)))
            eta = pm.Exponential("eta",
                                 lam=1.0)
            cov = eta ** 2 * pm.gp.cov.ExpQuad(input_dim=len(self.column_names), ls=ell)

            self.gp = pm.gp.Latent(cov_func=cov)
            f = self.gp.prior("f", X=model_input)

            lambda_intercept = pm.Normal("lambda_intercept",
                                         mu=self.priors['lambda_mu'] if 'lambda_intercept_mu' not in self.priors else self.priors['lambda_intercept_mu'],
                                         sigma=self.priors['lambda_sd'] if 'lambda_intercept_sd' not in self.priors else self.priors['lambda_intercept_sd'])

            # lambda_det = pm.Deterministic("lambda_det", pm.math.exp(lambda_intercept + f))
            lambda_det = pm.math.exp(lambda_intercept + f)

            if y is not None:
                censored_ = pm.math.eq(censor_, 1)
                y_ = pm.Exponential("y", pm.math.ones_like(time_uncensor_) / lambda_det[~censored_],   # noqa:F841
                                    observed=time_uncensor_)

                def exponential_lccdf(lam, time):
                    """ Log complementary cdf of Exponential distribution. """
                    return -(lam * time)

                y_cens = pm.Potential(  # noqa:F841
                    "y_cens",
                    exponential_lccdf(pm.math.ones_like(time_censor_) / lambda_det[censored_],
                                      time_censor_)
                )

        return model

    def fit(self, X, y, inference_args=None, priors=None):
        self.num_training_samples, self.num_pred = X.shape
        if isinstance(X, pd.DataFrame):
            self.column_names = list(X.columns.values)
        else:
            self.column_names = ["column_%i" % i for i in range(0, self.num_pred)]

        self.max_time = int(np.max(y))

        if y.ndim != 1:
            print('squeeze')
            y = np.squeeze(y)

        if not inference_args:
            inference_args = BayesianModel._get_default_inference_args()
            inference_args['draws'] = int(inference_args['draws'] / 8)
            inference_args['tune'] = int(inference_args['tune'] / 8)
            inference_args['type'] = 'blackjax'

        if self.cached_model is None:
            print('create from fit')
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
            print('create from predict')
            self.cached_model = self.create_model()

        with self.cached_model:
            pm.set_data({
                'model_input': X
            })

            f_pred = self.gp.conditional("f_pred", X, jitter=1e-4)
            lambda_intercept = self.cached_model.named_vars['lambda_intercept']
            p_pred = pm.Deterministic("lambda_det_pred", pm.math.exp(lambda_intercept + f_pred))   # noqa:F841

        ppc = pm.sample_posterior_predictive(self.trace, model=self.cached_model, return_inferencedata=False,
                                             random_seed=0, var_names=["f_pred", "lambda_det_pred"])
        print()

        t_plot = pmsurv.utils.get_time_axis(0, self.max_time, resolution)

        print(ppc.keys())
        pp_lambda = np.mean(ppc['lambda_det_pred'].reshape(-1, X.shape[0]), axis=0)
        pp_lambda_std = np.std(ppc['lambda_det_pred'].reshape(-1, X.shape[0]), axis=0)

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
        print(c_index)
        return c_index

    def save(self, file_prefix, **kwargs):
        custom_params = {
            'column_names': self.column_names,
            'priors': self.priors,
            'max_observed_time': self.max_time,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }
        print('store: ', self.priors)

        super(GaussianProcessModel, self).save(file_prefix, custom_params)

    def load(self, file_prefix, **kwargs):
        params = super(GaussianProcessModel, self).load(file_prefix)
        print('load: ', params['priors'])
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
        self.column_names = params['column_names']
        self.priors = params['priors']
        self.max_time = params['max_observed_time']
