from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from pmsurv.models.weibull_linear import WeibullModelLinear
from skopt.space import Real, Categorical, Integer
import utils


def preprocess_data(dataset, config):
    X = dataset[config['features']]
    y = dataset[[config['outcome']['time'], config['outcome']['event']]].values
    y[:, 1] = 1 - y[:, 1]  # inverse to censored
    return X, y


def train_model(X_train, y_train, config, train_kwargs):
    pipeline = Pipeline(
        [
            ('selector', SelectKBest(utils.mutual_info_surv)),
            ('model', WeibullModelLinear())
        ]
    )

    parameters = {
        'selector__k': Integer(1, X_train.shape[1]),
    }

    fit_params = {
        'inference_args': {'draws': 2000,
                           'tune': 1000,
                           'target_accept': 0.9,
                           'chains': 2,
                           'cores': 1,
                           'return_inferencedata': True,
                           'type': 'nuts'}
    }

    return pipeline, parameters, fit_params


def train_model_k(X_train, y_train, config, train_kwargs):
    pipeline = Pipeline(
        [
            ('selector', SelectKBest(utils.mutual_info_surv)),
            ('model', WeibullModelLinear(with_k=True))
        ]
    )

    parameters = {
        'selector__k': Integer(1, X_train.shape[1]),
    }

    fit_params = {
        'inference_args': {'draws': 2000,
                           'tune': 1000,
                           'target_accept': 0.9,
                           'chains': 2,
                           'cores': 1,
                           'return_inferencedata': True,
                           'type': 'nuts'}
    }

    return pipeline, parameters, fit_params
