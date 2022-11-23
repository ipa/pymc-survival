import os
import utils
import yaml
import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sksurv.datasets import get_x_y
from sksurv.linear_model import CoxPHSurvivalAnalysis
from skopt.space import Real, Categorical, Integer


def preprocess_data(dataset, config):
    X = dataset[config['features']]
    _, y = get_x_y(dataset,
                   attr_labels=[config['outcome']['event'], config['outcome']['time']],
                   pos_label=config['outcome']['pos_label'])
    return X, y


def train_model(X_train, y_train, config, train_kwargs):
    pipeline = Pipeline(
        [
            ('selector', SelectKBest(utils.mutual_info_surv)),
            ('model', CoxPHSurvivalAnalysis())
        ]
    )

    parameters = {
        'model__alpha': Real(1e-5, 0.9),
        'selector__k': Integer(1, X_train.shape[1]),
    }

    fit_params = {}

    return pipeline, parameters, fit_params

def save(save_dir, model, c_index, params, data):
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, "model.pkl"))
    joblib.dump(data, os.path.join(save_dir, "data.pkl"))
    metadata = {
        'c_index': c_index,
        'params': params,
    }
    with open(os.path.join(save_dir, "params.yaml"), 'w') as f:
        yaml.dump(metadata, f)
