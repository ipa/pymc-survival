# Adaptet from pymc-lean: https://github.com/pymc-learn/pymc-learn/blob/master/pmlearn/base.py
import logging
import os
from sklearn.base import BaseEstimator
import pymc as pm
import arviz as az
import yaml

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
            Check the PyMC docs to see what is permitted.

        num_advi_sample_draws : int (defaults to 10000)
            Number of samples to draw from ADVI approximation after it has been fit;
            not used if inference_type != 'advi'
        """
        if 'type' not in inference_args or 'nuts_sampler' in inference_args:
            self.__nuts_inference(inference_args)
        elif inference_args['type'] == 'nuts':
            inference_args.pop('type')
            self.__nuts_inference(inference_args)
        elif inference_args['type'] == 'map':
            logger.info('fit with MAP')
            with self.cached_model:
                self.trace = pm.find_MAP()

    def __nuts_inference(self, inference_args):
        """
        Runs NUTS inference.

        Parameters
        ----------
        inference_args : dict
            arguments to be passed to the PyMC sample method
            See PyMC doc for permissible values.
        """
        with self.cached_model:
            self.trace = pm.sample(**inference_args, random_seed=0)

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
        nuts_sampler = os.environ.get('PMSURV_SAMPLER')
        progress_bar = os.environ.get('PMSURV_PROGRESSBAR')

        inference_args = {
            'draws': 2000,
            'tune': 1000,
            'target_accept': 0.8,
            'chains': 2,
            'cores': 1,
            'return_inferencedata': True,
            'progressbar': False if progress_bar is None else progress_bar,
            'nuts_sampler': 'pymc' if nuts_sampler is None else nuts_sampler
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

        trace_file = f"{file}.netcdf"
        self.trace = az.from_netcdf(trace_file)
        custom_params = self.params['custom_params']
        return custom_params
