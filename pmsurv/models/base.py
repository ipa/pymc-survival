# Adaptet from pymc-lean: https://github.com/pymc-learn/pymc-learn/blob/master/pmlearn/base.py
from sklearn.base import BaseEstimator
import pymc as pm


class BayesianModel(BaseEstimator):
    """
    Bayesian model base class
    """
    def __init__(self):
        self.cached_model = None
        self.num_pred = None
        self.summary = None
        self.trace = None

    def create_model(self):
        raise NotImplementedError

    def _inference(self, inference_type='advi', inference_args=None, num_advi_sample_draws=10000):
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
            if 'nuts_kwargs' in inference_args:
                nuts_kwargs = inference_args.pop('nuts_kwargs')
            else:
                nuts_kwargs = {}
            step = pm.NUTS(**nuts_kwargs)
            self.trace = pm.sample(step=step, **inference_args, random_seed=0)

        # self.summary = pm.summary(self.trace)

    def __set_default_inference_args(self):
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
            'draws': 2000
        }
        return inference_args

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    #
    # def save(self, file_prefix, custom_params=None):
    #     """
    #     Saves the trace and custom params to files with the given file_prefix.
    #
    #     Parameters
    #     ----------
    #     file_prefix : str
    #         path and prefix used to identify where to save the trace for this model,
    #         e.g. given file_prefix = 'path/to/file/'
    #         This will attempt to save to 'path/to/file/trace.pickle'.
    #
    #     custom_params : dict (defaults to None)
    #         Custom parameters to save
    #     """
    #     fileObject = open(file_prefix + 'trace.pickle', 'wb')
    #     joblib.dump(self.trace, fileObject)
    #     fileObject.close()
    #
    #     if custom_params:
    #         fileObject = open(file_prefix + 'params.pickle', 'wb')
    #         joblib.dump(custom_params, fileObject)
    #         fileObject.close()
    #
    # def load(self, file_prefix, load_custom_params=False):
    #     """
    #     Loads a saved version of the trace, and custom param files with the given file_prefix.
    #
    #     Parameters
    #     ----------
    #     file_prefix : str
    #         path and prefix used to identify where to load the saved trace for this model,
    #         e.g. given file_prefix = 'path/to/file/'
    #         This will attempt to load 'path/to/file/trace.pickle'.
    #
    #     load_custom_params : bool (defaults to False)
    #         flag to indicate whether custom parameters should be loaded
    #
    #     Returns
    #     ----------
    #     custom_params : Dictionary of custom parameters
    #     """
    #     self.trace = joblib.load(file_prefix + 'trace.pickle')
    #
    #     custom_params = None
    #     if load_custom_params:
    #         custom_params = joblib.load(file_prefix + 'params.pickle')
    #
    #     return custom_params
