import os
import joblib
import yaml
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from pmsurv.models.gaussian_process import GaussianProcessModel
from skopt.space import Real, Categorical, Integer
import utils
import arviz as az


def preprocess_data(dataset, config):
    X = dataset[config['features']]
    y = dataset[[config['outcome']['time'], config['outcome']['event']]].values
    y[:, 1] = 1 - y[:, 1]  # inverse to censored
    # y[:, 0] = y[:, 0] / 30.25
    return X, y


def train_model(X_train, y_train, config, train_kwargs):
    pipeline = Pipeline(
        [
            ('selector', SelectKBest(utils.mutual_info_surv)),
            ('model', GaussianProcessModel())
        ]
    )

    parameters = {
        'selector__k': Integer(1, X_train.shape[1]),
    }

    fit_params = { }

    return pipeline, parameters, fit_params


# def get_priors(summary, config):
#     new_priors = ExponentialModel._get_default_priors()
#     for feature in config['features']:
#         feature_mu = summary['mean'][f'lambda_{feature}']
#         feature_sd = summary['sd'][f'lambda_{feature}']
#         new_priors[f'lambda_{feature}_mu'] = feature_mu
#         new_priors[f'lambda_{feature}_sd'] = feature_sd / 2
#
#     new_priors[f'lambda_intercept_mu'] = summary['mean'][f'lambda_intercept']
#     new_priors[f'lambda_intercept_sd'] = summary['sd'][f'lambda_intercept'] / 2
#     return new_priors

#
# def retrain_model(X_train, y_train, config, train_kwargs, prior_model=None):
#     model = ExponentialModel()
#     if prior_model is None:
#         model.fit(X_train, y_train)
#     else:
#         new_priors = get_priors(az.summary(prior_model.trace), config)
#         model.fit(X_train, y_train, priors=new_priors)
#
#     return model


def save(save_dir, model, c_index, params, data):
    os.makedirs(save_dir, exist_ok=True)

    model['model'].save(os.path.join(save_dir, "model.yaml"))
    joblib.dump(model['selector'], os.path.join(save_dir, "selector.pkl"))
    joblib.dump(data, os.path.join(save_dir, "data.pkl"))
    metadata = {
        'c_index': float(c_index),
        'params': dict(params),
    }
    with open(os.path.join(save_dir, "params.yaml"), 'w') as f:
        yaml.dump(metadata, f)

