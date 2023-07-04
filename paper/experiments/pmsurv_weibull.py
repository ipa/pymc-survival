from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from pmsurv.models.weibull_linear import WeibullModelLinear
from skopt.space import Real, Categorical, Integer
import utils
import arviz as az

def preprocess_data(dataset, config):
    X = dataset[config['features']]
    y = dataset[[config['outcome']['time'], config['outcome']['event']]].values
    y[:, 1] = 1 - y[:, 1]  # inverse to censored
    return X, y


def train_model(X_train, y_train, config, train_kwargs):
    pipeline = Pipeline(
        [
            ('selector', SelectKBest(utils.mutual_info_surv)),
            ('model', WeibullModelLinear(k_constant=True))
        ]
    )

    parameters = {
        'selector__k': Integer(1, X_train.shape[1]),
        'model__priors_sd': Real(0.1, 2, prior='uniform')
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

def get_priors(summary, config):
    new_priors = WeibullModelLinear(k_constant=True)._get_priors()
    for feature in config['features']:
        new_priors[f'lambda_{feature}_mu'] = summary['mean'][f'lambda_{feature}']
        new_priors[f'lambda_{feature}_sd'] = summary['sd'][f'lambda_{feature}']

        if new_priors['k_constant'] ==  False:
            new_priors[f'k_{feature}_mu'] = summary['mean'][f'k_{feature}']
            new_priors[f'k_{feature}_sd'] = summary['sd'][f'k_{feature}']

    new_priors[f'lambda_intercept_mu'] = summary['mean'][f'lambda_intercept']
    new_priors[f'lambda_intercept_sd'] = summary['sd'][f'lambda_intercept']

    new_priors[f'k_intercept_mu'] = summary['mean'][f'k_intercept']
    new_priors[f'k_intercept_sd'] = summary['sd'][f'k_intercept']
    return new_priors


def retrain_model(X_train, y_train, config, train_kwargs, prior_model=None):
    model = WeibullModelLinear(k_constant=True)
    if prior_model is None:
        model.fit(X_train, y_train)
    else:
        new_priors = get_priors(az.summary(prior_model.trace), config)
        model.fit(X_train, y_train, priors=new_priors)

    return model
