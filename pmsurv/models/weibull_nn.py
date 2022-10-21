import copy
import warnings
warnings.simplefilter("ignore")
import lifelines
import scipy.stats as st
import numpy as np
from numpy.random import default_rng
import pymc as pm
from pmsurv.exc import PyMCModelsError
import pmsurv.utils as putils
from pmsurv.models.base import BayesianModel
import aesara.tensor as at

# https://github.com/pyro-ppl/numpyro/issues/534
# http://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.mcmc.MCMC.post_warmup_state


class WeibullModelNN(BayesianModel):
    def __init__(self):
        super(WeibullModelNN, self).__init__()
        self.column_names = None
        self.max_time = None
        self.priors = None
        self.fit_args = None
        self.num_training_samples = None
        self.num_pred = None
        self.layers = []
  
    def __str__(self):
        str_output = ""
        for idx, l in enumerate(self.layers):
            str_output += f'Layer %i:\t %s \n' % (idx, str(l))
        return str_output

    @staticmethod
    def _get_default_priors():
        return {
            'k_coefs': False,
            'n_hidden_layers': [5, 5, 5],
            'lambda_mu': 1,
            'lambda_sd': 10,
            'k_mu': 1,
            'k_sd': 10,
            'coefs_mu': 0,
            'coefs_sd': 0.25
        }

    @staticmethod
    def _get_default_inference_args():
        return {
            'num_samples': 1000,
            'warmup_ratio': 1,
            'num_chains': 1
        }

    def create_model(self, X=None, y=None, priors=None):
        if self.priors is None:
            self.priors = priors
        if self.priors is None:
            self.priors = WeibullModelNN._get_default_priors()

        print(self.priors['n_hidden_layers'])
        n_hidden_layers = copy.deepcopy(self.priors['n_hidden_layers'])
        n_hidden_layers.insert(0, self.num_pred)
        n_hidden_layers.append(2)
        print(n_hidden_layers)

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
                x_hidden = model_input
                for idx_layer in range(len(n_hidden_layers)-1):
                    layer_input_dim = n_hidden_layers[idx_layer]
                    layer_output_dim = n_hidden_layers[idx_layer + 1]
                    self.layers.append((layer_input_dim, layer_output_dim))
                    weights = pm.Normal(f"w_{idx_layer}", self.priors['coefs_mu'], sigma=self.priors['coefs_sd'], shape=(layer_input_dim, layer_output_dim))
                    biases = pm.Normal(f"b_{idx_layer}", self.priors['coefs_mu'], sigma=self.priors['coefs_sd'], shape=(layer_output_dim))

                    x_hidden = pm.math.dot(x_hidden, weights) + biases

                    if idx_layer < len(n_hidden_layers) - 2:
                        x_hidden = pm.math.tanh(x_hidden)

                lambda_intercept = pm.Normal("lambda_intercept",
                                             mu=self.priors['lambda_mu'],
                                             sigma=self.priors['lambda_sd'],
                                             shape=(1))
                k_intercept = pm.Normal('k_intercept',
                                        mu=self.priors['k_mu'],
                                        sigma=self.priors['k_sd'],
                                        shape=(1))

                lambda_ = pm.Deterministic("lambda_", pm.math.exp(lambda_intercept + x_hidden[:, 0]))

                print('With k %s' % self.priors['k_coefs'])
                if self.priors['k_coefs']:
                    k_ = pm.Deterministic("k_", pm.math.exp(k_intercept + x_hidden[:, 1]))
                else:
                    k_constant = 0.0
                    k_ = pm.Deterministic("k_", pm.math.exp(k_intercept + (x_hidden[:,1] * k_constant)))

                censored = at.eq(censor_, 1)
                y = pm.Weibull("y", alpha=k_[~censored], beta=lambda_[~censored],
                               observed=time_uncensor_)

                def weibull_lccdf(x, alpha, beta):
                    """ Log complementary cdf of Weibull distribution. """
                    return -((x / beta) ** alpha)

                y_cens = pm.Potential("y_cens", weibull_lccdf(time_censor_, alpha=k_[censored],
                                                          beta=lambda_[censored]))

        return model


    def fit(self, X, y, inference_type='nuts', inference_args=None, priors=None, column_names=None):
        self.num_training_samples, self.num_pred = X.shape
        self.column_names = column_names
        self.inference_args = inference_args

        self.max_time = int(np.max(y))

        if y.ndim != 1:
            print('squeeze')
            y = np.squeeze(y)

        if not inference_args:
            inference_args = self.__set_default_inference_args()

        if self.cached_model is None:
            print('create from fit')
            self.cached_model = self.create_model(priors=priors)

        with self.cached_model:
            print(self)
            pm.set_data({
                'model_input': X,
                'time_censor': y[y[:, 1] == 1, 0],
                'time_uncensor': y[y[:, 1] == 0, 0],
                'censor': y[:, 1].astype(np.int32)
            })

        self._inference(inference_args)

        return self

    def predict(self, X, return_std=True, return_curve=True, num_ppc_samples=1000, resolution=10):
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
                                             random_seed=0, var_names=['y', 'lambda_', 'k_'])
        print()

        t_plot = putils.get_time_axis(0, self.max_time, resolution)

        pp_lambda = np.mean(ppc['lambda_'], axis=0)
        pp_k = np.mean(ppc['k_'], axis=0)
        pp_lambda_std = np.std(ppc['lambda_'], axis=0)
        pp_k_std = np.std(ppc['k_'], axis=0)

        t_plot_rep = np.repeat(t_plot, num_samples).reshape((len(t_plot), -1))
        pp_surv_mean = 1 - st.weibull_min.cdf(t_plot_rep, pp_k, scale=pp_lambda).T
        pp_surv_lower = 1 - st.weibull_min.cdf(t_plot_rep, pp_k - pp_k_std, scale=pp_lambda - pp_lambda_std).T
        pp_surv_upper = 1 - st.weibull_min.cdf(t_plot_rep, pp_k + pp_k_std, scale=pp_lambda + pp_lambda_std).T
        pp_surv_lower = np.nan_to_num(pp_surv_lower, 0)
        pp_surv_upper = np.nan_to_num(pp_surv_upper, 1)

        return pp_surv_mean, pp_surv_lower, pp_surv_upper

    def score(self, X, y, num_ppc_samples=1000):
        surv_prob, _, _ = self.predict(X, num_ppc_samples=num_ppc_samples)
        # print(surv_prob)
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
            'num_training_samples': self.num_training_samples,
        }

        super(WeibullModelNN, self).save(file_prefix, custom_params)

    def load(self, file_prefix, **kwargs):
        params = super(WeibullModelNN, self).load(file_prefix)

        self.column_names = params['column_names']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']
        self.priors = params['priors']
        self.inference_args = params['inference_args']
        self.max_time = params['max_observed_time']

    def __show_graph(self):
        try:
            print("show graph")
            import matplotlib.pyplot as plt
            g = pm.model_to_graphviz(self.cached_model)
            g.render("graphname", format="png")
        except ImportError:
            print("graphviz not available")
