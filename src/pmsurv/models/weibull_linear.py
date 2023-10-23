import logging
import numpy as np
import pymc as pm
from pmsurv.models.weibull_base import WeibullModelBase

logger = logging.getLogger(__name__)


class WeibullModelLinear(WeibullModelBase):
    def __init__(self, k_constant=True, priors_sd=1):
        super(WeibullModelLinear, self).__init__()
        self.column_names = None
        self.max_time = None
        self.fit_args = None
        self.num_training_samples = None
        self.num_pred = None
        self.k_constant = k_constant
        self.priors_sd = priors_sd

    def _get_priors(self):
        return {
            'k_constant': True,
            'lambda_mu': 1,
            'lambda_sd': self.priors_sd,
            'k_mu': 1,
            'k_sd': self.priors_sd,
            'lambda_coefs_mu': 0,
            'lambda_coefs_sd': self.priors_sd,
            'k_coefs_mu': 0,
            'k_coefs_sd': self.priors_sd
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
            self.priors = self._get_priors()
        self.priors['k_constant'] = self.k_constant

        model = pm.Model()
        with model:
            if X is None:
                X = np.zeros([self.num_training_samples, self.num_pred])
            model_input = pm.MutableData("model_input", X).astype('float32')

            if y is not None:
                time_censor_ = pm.MutableData("time_censor", y[y[:, 1] == 1, 0]).astype('float32')
                time_uncensor_ = pm.MutableData("time_uncensor", y[y[:, 1] == 0, 0]).astype('float32')
                censor_ = pm.MutableData("censor", y[:, 1].astype(np.int8))

        with model:
            logger.info("Priors: {}".format(str(self.priors)))
            lambda_intercept = pm.Normal("lambda_intercept",
                                         mu=self.priors['lambda_mu'] if 'lambda_intercept_mu' not in self.priors else self.priors['lambda_intercept_mu'],
                                         sigma=self.priors['lambda_sd'] if 'lambda_intercept_sd' not in self.priors else self.priors['lambda_intercept_sd'])  # .astype('float32')

            k_intercept = pm.Normal('k_intercept',
                                    mu=self.priors['k_mu'] if 'k_intercept_mu' not in self.priors else self.priors['k_intercept_mu'],
                                    sigma=self.priors['k_sd'] if 'k_intercept_sd' not in self.priors else self.priors['k_intercept_sd'])  # .astype('float32')

            lambda_coefs = []
            for i, cn in enumerate(self.column_names):
                feature_name = f'lambda_{cn}'
                lambda_coef = pm.Normal(feature_name,
                                        mu=self.priors['lambda_coefs_mu'] if f'{feature_name}_mu' not in self.priors else self.priors[f'{feature_name}_mu'],
                                        sigma=self.priors['lambda_coefs_sd'] if f'{feature_name}_sd' not in self.priors else self.priors[f'{feature_name}_sd'])  # .astype('float32')
                lambda_coefs.append(model_input[:, i] * lambda_coef)
            lambda_det = pm.Deterministic("lambda_det", pm.math.exp(lambda_intercept + sum(lambda_coefs)))

            if not self.priors['k_constant']:
                k_coefs = []
                for i, cn in enumerate(self.column_names):
                    feature_name = f'k_{cn}'
                    k_coef = pm.Normal(feature_name,
                                       mu=self.priors['k_coefs_mu'] if f'{feature_name}_mu' not in self.priors else self.priors[f'{feature_name}_mu'],
                                       sigma=self.priors['k_coefs_sd'] if f'{feature_name}_sd' not in self.priors else self.priors[f'{feature_name}_sd'])  # .astype('float32')
                    k_coefs.append(model_input[:, i] * k_coef)
                k = pm.math.sum(k_coefs, axis=0)
            else:
                k = pm.math.zeros_like(lambda_det)
            k_det = pm.Deterministic("k_det", pm.math.exp(k_intercept + k))

            if y is not None:
                censored_ = pm.math.eq(censor_, 1)
                y = pm.Weibull("y", alpha=k_det[~censored_], beta=lambda_det[~censored_],
                               observed=time_uncensor_)

                def weibull_lccdf(x, alpha, beta):
                    """ Log complementary cdf of Weibull distribution. """
                    return -((x / beta) ** alpha)

                y_cens = pm.Potential("y_cens", weibull_lccdf(time_censor_, alpha=k_det[censored_], beta=lambda_det[censored_]))  # noqa:F841

        return model

    def save(self, file_prefix, **kwargs):
        custom_params = {
            'column_names': self.column_names,
            'priors': self.priors,
            'max_observed_time': self.max_time,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }
        logger.info('store: ', self.priors)

        super(WeibullModelLinear, self).save(file_prefix, custom_params)

    def load(self, file_prefix, **kwargs):
        params = super(WeibullModelLinear, self).load(file_prefix)
        logger.info('load: ', params['priors'])
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
        self.column_names = params['column_names']
        self.priors = params['priors']
        self.max_time = params['max_observed_time']
