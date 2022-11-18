# Adaptet from pymc-lean: https://github.com/pymc-learn/pymc-learn/blob/master/pmlearn/base.py
import logging

from sklearn.base import BaseEstimator
import pymc as pm
import arviz as az
import yaml
import numpy as np
import pmsurv
import pmsurv.utils
from pmsurv.exc import PyMCModelsError
import scipy.stats as st
import lifelines
import pandas as pd
logger = logging.getLogger(__name__)


class BayesianModel(BaseEstimator):
    """
    Bayesian model base class
    """

    def __init__(self):
        self.cached_model = None
        self.num_pred = None
        self.trace = None
        self.params = {}

    def create_model(self):
        raise NotImplementedError

    def _inference(self, inference_args=None):
        """
        Calls internal methods for two types of inferences.
        Raises an error if the inference_type is not supported.

        Parameters
        ----------
        inference_type : str (defaults to 'advi')
            specifies which inference method to call
            Currently, only 'advi' and 'nuts' are supported.

        inference_args : dict (defaults to None)
            arguments to be passed to the inference methods
            Check the PyMC3 docs to see what is permitted.

        num_advi_sample_draws : int (defaults to 10000)
            Number of samples to draw from ADVI approximation after it has been fit;
            not used if inference_type != 'advi'
        """

        self.__nuts_inference(inference_args)

    def __nuts_inference(self, inference_args):
        """
        Runs NUTS inference.

        Parameters
        ----------
        inference_args : dict
            arguments to be passed to the PyMC3 sample method
            See PyMC3 doc for permissible values.
        """
        with self.cached_model:
            if 'type' not in inference_args:
                self.trace = pm.sample(**inference_args, random_seed=0)
            elif inference_args['type'] == 'nuts':
                inference_args.pop('type')
                self.trace = pm.sample(**inference_args, random_seed=0)
            elif inference_args['type'] == 'blackjax':
                inference_args.pop('type')
                inference_args.pop('cores')
                import pymc.sampling_jax
                if 'progressbar' in inference_args:
                    inference_args.pop('progressbar')
                if 'return_inferencedata' in inference_args:
                    inference_args.pop('return_inferencedata')

                self.trace = pm.sampling_jax.sample_blackjax_nuts(**inference_args)

    @staticmethod
    def _get_default_inference_args():
        """
        Set default values for inference arguments if none are provided, dependent on inference type.

        ADVI Default Parameters
        -----------------------
        callbacks : list
            contains a parameter stopping check.

        n : int (defaults to 200000)
            number of iterations for ADVI fit

        NUTS Default Parameters
        -----------------------
        draws : int (defaults to 2000)
            number of samples to draw
        """
        inference_args = {
            'draws': 1000,
            'tune': 500,
            'target_accept': 0.8,
            'chains': 2,
            'cores': 1,
            'return_inferencedata': True,
            'progressbar': False,
            'type': 'nuts'
        }
        return inference_args

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def score(self, X, y, **kwargs):
        raise NotImplementedError

    def print_summary(self):
        print(az.summary(self.trace))

    def save(self, file, custom_params=None):
        """
        Saves the trace and custom params to files with the given file_prefix.

        Parameters
        ----------
        file : str
            path and prefix used to identify where to save the trace for this model,
            e.g. given file_prefix = 'path/to/file/'
            This will attempt to save to 'path/to/file/trace.pickle'.

        custom_params : dict (defaults to None)
            Custom parameters to save
        """
        trace_file = f"{file}.netcdf"
        az.to_netcdf(self.trace, trace_file)

        self.params['custom_params'] = custom_params
        self.params['trace_file'] = trace_file
        with open(file, 'w') as f:
            yaml.dump(self.params, f)

    def load(self, file):
        """
        Loads a saved version of the trace, and custom param files with the given file_prefix.

        Parameters
        ----------
        file_prefix : str
            path and prefix used to identify where to load the saved trace for this model,
            e.g. given file_prefix = 'path/to/file/'
            This will attempt to load 'path/to/file/trace.pickle'.

        load_custom_params : bool (defaults to False)
            flag to indicate whether custom parameters should be loaded

        Returns
        ----------
        custom_params : Dictionary of custom parameters
        """

        with open(file, 'r') as f:
            self.params = yaml.safe_load(f)

        self.trace = az.from_netcdf(self.params['trace_file'])
        custom_params = self.params['custom_params']
        return custom_params


class WeibullModelBase(BayesianModel):

    def __init__(self):
        super(WeibullModelBase, self).__init__()
        self.priors = None

    def create_model(self, X=None, y=None, priors=None):
        raise NotImplementedError

    def fit(self, X, y, inference_args=None, priors=None):
        self.num_training_samples, self.num_pred = X.shape
        if X is pd.DataFrame:
            self.column_names = X.columns.values
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
