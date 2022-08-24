# import warnings
# warnings.simplefilter("ignore")
import lifelines
import scipy.stats as st
import numpy as np
from numpy.random import default_rng
import pymc as pm
from pmsurv.exc import PyMCModelsError
from pmsurv.models.base import BayesianModel
import pmsurv.utils
import aesara.tensor as at


class ExponentialModel(BayesianModel):
    def __init__(self):
        super(ExponentialModel, self).__init__()
        self.column_names = None
        self.max_time = None
        self.priors = None
        self.fit_args = None
        self.num_training_samples = None
        self.num_pred = None

    @staticmethod
    def _get_default_priors():
        return {
            'lambda_mu': 1,
            'lambda_sd': 1,
            'lambda_coefs_mu': 0,
            'lambda_coefs_sd': 1
        }

    @staticmethod
    def _get_default_inference_args():
        return {
            'num_samples': 1000,
            'warmup_ratio': 1,
            'num_chains': 1
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
            self.priors = ExponentialModel._get_default_priors()

        model = pm.Model()
        with model:
            if X is None:
                print("create cached with empty data")
                model_input = pm.MutableData("model_input", np.zeros([self.num_training_samples, self.num_pred]))
                time_censor_ = pm.MutableData("time_censor",
                                              np.zeros(np.ceil(self.num_training_samples / 2).astype(int)))
                time_uncensor_ = pm.MutableData("time_uncensor",
                                                np.zeros(np.floor(self.num_training_samples / 2).astype(int)))
                censor_ = pm.MutableData("censor", default_rng(seed=0).choice([1, 0], size=(self.num_training_samples),
                                                                              p=[0.5, 0.5]).astype(np.int32))
            else:
                print("create cached with real data")
                model_input = pm.MutableData("model_input", X)
                time_censor_ = pm.MutableData("time_censor", y[y[:, 1] == 1, 0])
                time_uncensor_ = pm.MutableData("time_uncensor", y[y[:, 1] == 0, 0])
                censor_ = pm.MutableData("censor", y[:, 1].astype(np.int8))
                # print(censor_.dtype)

        with model:
            print("Priors: {}".format(str(self.priors)))

            lambda_intercept = pm.Normal("lambda_intercept",
                                         mu=self.priors['lambda_mu'], sigma=self.priors['lambda_sd'])

            lambda_coefs = []
            for i, cn in enumerate(self.column_names):
                lambda_coef = pm.Normal(f'lambda_{cn}',
                                        mu=self.priors['lambda_coefs_mu'],
                                        sigma=self.priors['lambda_coefs_sd'])
                lambda_coefs.append(model_input[:, i] * lambda_coef)
            lambda_det = pm.Deterministic("lambda_det", pm.math.exp(lambda_intercept + sum(lambda_coefs)))

            print(lambda_det.shape.eval(), time_censor_.shape.eval(), time_uncensor_.shape.eval())

            censor_ = at.eq(censor_, 1)
            y = pm.Exponential("y", at.ones_like(time_uncensor_) / lambda_det[~censor_],
                               observed=time_uncensor_)

            def exponential_lccdf(lam, time):
                """ Log complementary cdf of Exponential distribution. """
                return -(lam * time)

            y_cens = pm.Potential(
                "y_cens", exponential_lccdf(at.ones_like(time_censor_) / lambda_det[censor_], time_censor_)
            )

        return model

    def fit(self, X, y, inference_args=None, priors=None):
        self.num_training_samples, self.num_pred = X.shape
        self.column_names = list(X.columns.values)
        self.inference_args = inference_args if inference_args is not None else ExponentialModel._get_default_inference_args()

        self.max_time = int(np.max(y))

        if y.ndim != 1:
            print('squeeze')
            y = np.squeeze(y)

        if not inference_args:
            inference_args = self.__set_default_inference_args()

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
                'model_input': X,
                'time_uncensor': np.zeros(num_samples).astype(np.int32),
                'censor': np.zeros(num_samples).astype(np.int32)
            })

        ppc = pm.sample_posterior_predictive(self.trace, model=self.cached_model, return_inferencedata=False,
                                             random_seed=0, var_names=['y', 'lambda_det'])
        print()

        t_plot = pmsurv.utils.get_time_axis(0, self.max_time, resolution)

        print(ppc.keys())
        pp_lambda = np.mean(ppc['lambda_det'], axis=0)
        pp_lambda_std = np.std(ppc['lambda_det'], axis=0)

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
            'inference_args': self.inference_args,
            'max_observed_time': self.max_time,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }
        print('store: ', self.priors)

        super(ExponentialModel, self).save(file_prefix, custom_params)

    def load(self, file_prefix, **kwargs):
        params = super(ExponentialModel, self).load(file_prefix)
        print('load: ', params['priors'])
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
        self.column_names = params['column_names']
        self.priors = params['priors']
        self.inference_args = params['inference_args']
        self.max_time = params['max_observed_time']
