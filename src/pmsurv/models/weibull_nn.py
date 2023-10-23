import copy
import numpy as np
from numpy.random import default_rng
import pymc as pm
from pmsurv.models.weibull_base import WeibullModelBase

# https://github.com/pyro-ppl/numpyro/issues/534
# http://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.mcmc.MCMC.post_warmup_state


class WeibullModelNN(WeibullModelBase):
    def __init__(self, k_constant=True, n_hidden_layers=1, priors_sd=0.25):
        super(WeibullModelNN, self).__init__()
        self.column_names = None
        self.max_time = None
        self.priors = None
        self.fit_args = None
        self.num_training_samples = None
        self.num_pred = None
        self.n_hidden_layers = n_hidden_layers
        self.layers = []
        self.k_constant = k_constant
        self.priors_sd = priors_sd

    def __str__(self):
        str_output = ""
        for idx, l in enumerate(self.layers):
            str_output += 'Layer %i:\t %s \n' % (idx, str(l))
        return str_output

    def _get_priors(self):
        return {
            'k_constant': False,
            'n_hidden_layers': [5, 5, 5],
            'lambda_mu': 1,
            'lambda_sd': self.priors_sd * 40,
            'k_mu': 1,
            'k_sd': self.priors_sd * 40,
            'coefs_mu': 0,
            'coefs_sd': self.priors_sd,
            'lambda_coefs_mu': 0,
            'lambda_coefs_sd': self.priors_sd
        }

    def create_model(self, X=None, y=None, priors=None):
        if self.priors is None:
            self.priors = priors
        if self.priors is None:
            self.priors = self._get_priors()
        self.priors['n_hidden_layers'] = [self.num_pred for i in range(self.n_hidden_layers)]
        self.priors['k_constant'] = self.k_constant

        print(self.priors['n_hidden_layers'])
        n_hidden_layers = copy.deepcopy(self.priors['n_hidden_layers'])
        n_hidden_layers.insert(0, self.num_pred)
        output_dims = 1 if self.priors['k_constant'] else 2
        n_hidden_layers.append(output_dims)
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
                                             sigma=self.priors['lambda_sd'])

                k_intercept = pm.Normal('k_intercept',
                                        mu=self.priors['k_mu'],
                                        sigma=self.priors['k_sd'])

                lambda_ = pm.Deterministic("lambda_det", pm.math.exp(lambda_intercept + x_hidden[:, 0]))

                print('Keep k constant = %s' % self.priors['k_constant'])
                if self.priors['k_constant']:
                    k = pm.math.ones_like(x_hidden[:, 1])
                    k_ = pm.Deterministic("k_det", pm.math.exp(k_intercept * k))  # + (x_hidden[:, 1] * k_constant)
                else:
                    k_ = pm.Deterministic("k_det", pm.math.exp(k_intercept + x_hidden[:, 1]))

                censored = pm.math.eq(censor_, 1)
                y = pm.Weibull("y", alpha=k_[~censored], beta=lambda_[~censored],
                               observed=time_uncensor_)

                def weibull_lccdf(x, alpha, beta):
                    """ Log complementary cdf of Weibull distribution. """
                    return -((x / beta) ** alpha)

                y_cens = pm.Potential("y_cens", weibull_lccdf(time_censor_, alpha=k_[censored],  # noqa:F841
                                                              beta=lambda_[censored]))

        return model

    def save(self, file_prefix, **kwargs):
        custom_params = {
            'column_names': self.column_names,
            'priors': self.priors,
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
        self.max_time = params['max_observed_time']

    def __show_graph(self):
        try:
            print("show graph")
            g = pm.model_to_graphviz(self.cached_model)
            g.render("graphname", format="png")
        except ImportError:
            print("graphviz not available")
