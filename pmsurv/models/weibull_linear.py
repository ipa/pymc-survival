import numpy as np
from numpy.random import default_rng
import pymc as pm
from pmsurv.models.base import WeibullModelBase
import aesara.tensor as at


class WeibullModelLinear(WeibullModelBase):
    def __init__(self):
        super(WeibullModelLinear, self).__init__()
        self.column_names = None
        self.max_time = None
        # self.priors = None
        self.fit_args = None
        self.num_training_samples = None
        self.num_pred = None

    @staticmethod
    def _get_default_priors():
        return {
            'k_coefs': False,
            'uncertainty': [],
            'lambda_mu': 1,
            'lambda_sd': 1,
            'k_mu': 1,
            'k_sd': 1,
            'lambda_coefs_mu': 0,
            'lambda_coefs_sd': 1,
            'k_coefs_mu': 0,
            'k_coefs_sd': 1
        }


    def __str__(self):
        str_output = "WeibullModelTreatment \n\r"
        str_output += str(self.column_names) + "\r\n"
        str_output += str(self.priors) + "\r\n"
        str_output += str(self.fit_args) + "\r\n"
        return str_output

    def create_model(self, X=None, y=None, priors=None):
        if self.priors is None:
            self.priors = priors
        if self.priors is None:
            self.priors = WeibullModelLinear._get_default_priors()

        model = pm.Model()
        with model:
            if X is None:
                X = np.zeros([self.num_training_samples, self.num_pred])

            model_input = pm.MutableData("model_input", X)
            if y is not None:
                time_censor_ = pm.MutableData("time_censor", y[y[:, 1] == 1, 0])
                time_uncensor_ = pm.MutableData("time_uncensor", y[y[:, 1] == 0, 0])
                censor_ = pm.MutableData("censor", y[:, 1].astype(np.int8))

            lambda_intercept = pm.Normal("lambda_intercept",
                                         mu=self.priors['lambda_mu'],
                                         sigma=self.priors['lambda_sd'])

            k_intercept = pm.Normal('k_intercept',
                                    mu=self.priors['k_mu'],
                                    sigma=self.priors['k_sd'])

            lambda_coefs = []
            for i, cn in enumerate(self.column_names):
                lambda_coef = pm.Normal(f'lambda_{cn}',
                                        mu=self.priors['lambda_coefs_mu'],
                                        sigma=self.priors['lambda_coefs_sd'])
                lambda_coefs.append(model_input[:, i] * lambda_coef)
            lam = pm.math.sum(lambda_coefs, axis=0)

            if self.priors['k_coefs']:
                k_coefs = []
                for i, cn in enumerate(self.column_names):
                    k_coef = pm.Normal(f'k_{cn}',
                                            mu=self.priors['k_coefs_mu'],
                                            sigma=self.priors['k_coefs_sd'])
                    k_coefs.append(model_input[:, i] * k_coef)
                k = pm.math.sum(k_coefs, axis=0)
            else:
                k = pm.math.zeros_like(lam)

            lambda_ = pm.Deterministic("lambda_det",
                                       pm.math.exp(lambda_intercept + lam))
            k_ = pm.Deterministic("k_det", pm.math.exp(k_intercept + k))

            if y is not None:
                censored_ = at.eq(censor_, 1)
                y = pm.Weibull("y", alpha=k_[~censored_], beta=lambda_[~censored_],
                               observed=time_uncensor_)

                def weibull_lccdf(x, alpha, beta):
                    """ Log complementary cdf of Weibull distribution. """
                    return -((x / beta) ** alpha)

                y_cens = pm.Potential("y_cens", weibull_lccdf(time_censor_, alpha=k_[censored_], beta=lambda_[censored_]))

        return model

    def save(self, file_prefix, **kwargs):
        custom_params = {
            'column_names': self.column_names,
            'priors': self.priors,
            # 'inference_args': self.inference_args,
            'max_observed_time': self.max_time,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }
        print('store: ', self.priors)

        super(WeibullModelLinear, self).save(file_prefix, custom_params)

    def load(self, file_prefix, **kwargs):
        params = super(WeibullModelLinear, self).load(file_prefix)
        print('load: ', params['priors'])
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
        self.column_names = params['column_names']
        self.priors = params['priors']
        # self.inference_args = params['inference_args']
        self.max_time = params['max_observed_time']
